"""
Task 2 Post-Processing: Relabel Oncogenic Predictions

After LLM classification, ground-truth labels in the output CSVs can drift from
the source mutation database due to annotation updates (OncoKB re-annotations,
AlphaMissense re-scoring, etc.). This script re-joins each output row back to the
source mutation database using a 3-key (SAMPLE_ID, Hugo_Symbol, HGVSp_Short) and
updates mutation_id, the oncogenic label, and AlphaMissense_label where a unique
unambiguous match exists.

Rows with ambiguous matches (same 3-key maps to >1 mutation_id) are skipped and
reported. Rows with no match are left unchanged.

Usage:
    python relabel_oncogenic_predictions.py \\
        --mutations  /path/to/maf_annotated_final_dbSNP.csv \\
        --input-dir  /path/to/llm_output_dir \\
        --output-dir /path/to/corrected_output_dir

All CSV files in --input-dir are processed; corrected files are written to
--output-dir with the same filenames.
"""

import argparse
import logging
import os

import pandas as pd


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


# ============================================================================
# Core relabeling logic
# ============================================================================

def _detect_columns(df: pd.DataFrame):
    """Return the oncogenic-label and AlphaMissense column names present in df."""
    onc_col = (
        'ONCOGENIC' if 'ONCOGENIC' in df.columns
        else 'OncoKB_label' if 'OncoKB_label' in df.columns
        else 'ground_truth' if 'ground_truth' in df.columns
        else None
    )
    am_col = 'AlphaMissense_label' if 'AlphaMissense_label' in df.columns else None
    gene_col = 'Hugo_Symbol' if 'Hugo_Symbol' in df.columns else 'gene' if 'gene' in df.columns else None
    return onc_col, am_col, gene_col


def build_source_index(mutations: pd.DataFrame):
    """
    Build a lookup dict: (SAMPLE_ID, Hugo_Symbol, HGVSp_Short) -> row info.
    Only entries with a unique 3-key are indexed; ambiguous entries are flagged.
    """
    key_cols = ['SAMPLE_ID', 'Hugo_Symbol', 'HGVSp_Short']
    for col in key_cols + ['mutation_id']:
        if col not in mutations.columns:
            raise ValueError(f"Source mutations file is missing required column: '{col}'")

    src = mutations[key_cols + ['mutation_id', 'ONCOGENIC', 'AlphaMissense_label']].copy()
    src = src.drop_duplicates(subset=key_cols + ['mutation_id'])

    counts = src.groupby(key_cols)['mutation_id'].nunique()
    unique_keys = set(counts[counts == 1].index.tolist())
    ambiguous_keys = set(counts[counts > 1].index.tolist())

    index = {}
    for _, row in src[src.set_index(key_cols).index.isin(unique_keys)].iterrows():
        key = (row['SAMPLE_ID'], row['Hugo_Symbol'], row['HGVSp_Short'])
        index[key] = {
            'mutation_id': row['mutation_id'],
            'ONCOGENIC': row['ONCOGENIC'],
            'AlphaMissense_label': row.get('AlphaMissense_label'),
        }

    logging.info(f"Source index: {len(index)} unique keys, {len(ambiguous_keys)} ambiguous keys")
    return index, ambiguous_keys


def relabel_dataframe(df: pd.DataFrame, source_index: dict, ambiguous_keys: set):
    """
    Re-join df against source_index and update mutation_id, oncogenic label,
    and AlphaMissense_label where a unique match is found.

    Returns (updated_df, n_mid_updates, n_onc_updates, n_am_updates, n_ambiguous, n_notfound).
    """
    onc_col, am_col, gene_col = _detect_columns(df)

    if gene_col is None:
        logging.warning("No gene column (Hugo_Symbol / gene) found — skipping file")
        return df, 0, 0, 0, 0, 0

    if 'HGVSp_Short' not in df.columns:
        logging.warning("No HGVSp_Short column found — skipping file")
        return df, 0, 0, 0, 0, 0

    df = df.copy()

    n_mid = n_onc = n_am = n_amb = n_notfound = 0

    for idx, row in df.iterrows():
        key = (str(row['SAMPLE_ID']).strip(),
               str(row[gene_col]).strip(),
               str(row['HGVSp_Short']).strip())

        if key in ambiguous_keys:
            n_amb += 1
            continue

        src = source_index.get(key)
        if src is None:
            n_notfound += 1
            continue

        # Update mutation_id
        if 'mutation_id' in df.columns and str(df.at[idx, 'mutation_id']) != str(src['mutation_id']):
            df.at[idx, 'mutation_id'] = src['mutation_id']
            n_mid += 1

        # Update oncogenic label
        if onc_col and str(df.at[idx, onc_col]) != str(src['ONCOGENIC']):
            df.at[idx, onc_col] = src['ONCOGENIC']
            n_onc += 1

        # Update AlphaMissense label
        if am_col and src['AlphaMissense_label'] is not None:
            if str(df.at[idx, am_col]) != str(src['AlphaMissense_label']):
                df.at[idx, am_col] = src['AlphaMissense_label']
                n_am += 1

    return df, n_mid, n_onc, n_am, n_amb, n_notfound


# ============================================================================
# File-level processing
# ============================================================================

def process_directory(input_dir: str, output_dir: str, source_index: dict, ambiguous_keys: set):
    os.makedirs(output_dir, exist_ok=True)

    csv_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
    if not csv_files:
        logging.warning(f"No CSV files found in {input_dir}")
        return

    logging.info(f"Found {len(csv_files)} CSV files to process")

    total_mid = total_onc = total_am = total_amb = total_nf = 0

    for fname in sorted(csv_files):
        input_path = os.path.join(input_dir, fname)
        output_path = os.path.join(output_dir, fname)

        try:
            df = pd.read_csv(input_path)
        except Exception as e:
            logging.error(f"Could not read {fname}: {e}")
            continue

        updated, n_mid, n_onc, n_am, n_amb, n_nf = relabel_dataframe(df, source_index, ambiguous_keys)

        updated.to_csv(output_path, index=False)

        logging.info(
            f"{fname}: mutation_id={n_mid}, label={n_onc}, AlphaMissense={n_am} "
            f"updates | ambiguous={n_amb}, not_found={n_nf}"
        )

        total_mid += n_mid
        total_onc += n_onc
        total_am += n_am
        total_amb += n_amb
        total_nf += n_nf

    logging.info("=" * 60)
    logging.info(
        f"Total: mutation_id={total_mid}, label={total_onc}, "
        f"AlphaMissense={total_am} updates | "
        f"ambiguous={total_amb}, not_found={total_nf}"
    )


# ============================================================================
# Entry point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Re-join Task 2 LLM output files against source mutation database to fix labels.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python relabel_oncogenic_predictions.py \\
      --mutations /data1/morrisq/yuj13/llm_genomics/msk_solid_heme/maf_annotated_final_dbSNP.csv \\
      --input-dir /data1/morrisq/yuj13/llm_genomics/mutation_output/impact_oncogenic_sample \\
      --output-dir /data1/morrisq/yuj13/llm_genomics/mutation_output/impact_oncogenic_sample_relabeled
        """
    )
    parser.add_argument('--mutations', type=str, required=True,
                        help='Path to source mutation database CSV (maf_annotated_final_dbSNP.csv)')
    parser.add_argument('--input-dir', type=str, required=True,
                        help='Directory containing LLM output CSVs to relabel')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Directory to write relabeled CSVs (created if needed)')

    args = parser.parse_args()

    logging.info("=" * 60)
    logging.info("Task 2 Prediction Relabeling")
    logging.info("=" * 60)
    logging.info(f"Source mutations: {args.mutations}")
    logging.info(f"Input dir:        {args.input_dir}")
    logging.info(f"Output dir:       {args.output_dir}")

    logging.info("Loading source mutation database...")
    mutations = pd.read_csv(args.mutations, low_memory=False)
    logging.info(f"Loaded {len(mutations)} mutations")

    source_index, ambiguous_keys = build_source_index(mutations)

    process_directory(args.input_dir, args.output_dir, source_index, ambiguous_keys)

    logging.info("Done.")


if __name__ == '__main__':
    main()

import yaml
import pandas as pd
import numpy as np
import logging
import argparse
import os
import sys

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')


def load_config(path):
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)

    # Allow templated path formatting (e.g., {data_directory})
    def fmt(v):
        if isinstance(v, str):
            try:
                return v.format(**cfg)
            except Exception:
                return v
        return v

    def recurse(o):
        if isinstance(o, dict):
            return {k: recurse(fmt(v)) for k, v in o.items()}
        elif isinstance(o, list):
            return [recurse(fmt(v)) for v in o]
        else:
            return fmt(o)

    return recurse(cfg)


def read_clinical_sample_and_patient(sample_path, patient_path):
    sample_df = pd.read_csv(sample_path, sep='\t', comment='#')
    patient_df = pd.read_csv(patient_path, sep='\t', comment='#')
    patient_df['GENDER'] = patient_df.get('SEX', pd.Series(dtype=str))
    merged = pd.merge(
        patient_df[['PATIENT_ID', 'GENDER']],
        sample_df,
        on='PATIENT_ID',
        how='left'
    )
    merged['sex_report_str'] = "Sex of patient: " + merged['GENDER'].fillna('Unknown')
    return merged


def create_mutation_report_str(mutations_df):
    mutations_df['Exon_Number'] = mutations_df['Exon_Number'].fillna('NA')
    mutations_df['report_str'] = (
        mutations_df['Hugo_Symbol']
        + ' exon '
        + mutations_df['Exon_Number'].str.split('/').str[0]
        + ' '
        + mutations_df['Consequence'].str.replace('_', ' ')
        + ' '
        + mutations_df['HGVSp_Short']
        + ' ('
        + mutations_df['Reference_Allele']
        + '>'
        + mutations_df['Tumor_Seq_Allele2']
        + ')'
    )
    grouped = (
        mutations_df[['Tumor_Sample_Barcode', 'report_str']]
        .fillna('')
        .groupby('Tumor_Sample_Barcode')['report_str']
        .apply(lambda rows: '\n'.join(rows))
        .reset_index()
    )
    return grouped.rename(columns={'Tumor_Sample_Barcode': 'SAMPLE_ID'})


def create_gene_level_cna(df):
    numeric_cols = df.select_dtypes(include=['number']).columns
    result = []

    for col in numeric_cols:
        col_series = df[col]
        amplifications = (col_series > 0)
        deletions = (col_series < 0)
        result.append(
            np.where(
                amplifications, f"{col} amplification",
                np.where(deletions, f"{col} deletion", "")
            )
        )

    stacked = np.vstack(result).T
    df['alteration'] = ['\n'.join(filter(None, row)) for row in stacked]
    df = df.reset_index().rename(columns={'index': 'SAMPLE_ID'})
    report = df[df['alteration'].str.strip().astype(bool)][['SAMPLE_ID', 'alteration']]
    return report.rename(columns={'alteration': 'cna_report_str'})


def create_chr_level_report(df, prefix_amp='Amplification in Chr ', prefix_del='Deletion in Chr '):
    numeric_cols = df.select_dtypes(include=['number']).columns
    result = []

    for col in numeric_cols:
        col_series = df[col]
        has_cna = (col_series > 0)
        result.append(np.where(has_cna, col, ""))

    stacked = np.vstack(result).T
    df['alteration'] = ['\n'.join(filter(None, row)) for row in stacked]
    df['alteration'] = df['alteration'].str.replace('Amp_', prefix_amp, regex=False)
    df['alteration'] = df['alteration'].str.replace('Del_', prefix_del, regex=False)
    df = df.reset_index().rename(columns={'index': 'SAMPLE_IDX'})
    return df[['SAMPLE_IDX', 'alteration']].rename(columns={'alteration': 'chr_report_str'})


def write_batched(df, base_path, batch_size, suffix=""):
    """
    Write DataFrame to CSV(s). If batch_size is None, write a single file. Otherwise chunk.
    """
    dirname = os.path.dirname(base_path)
    os.makedirs(dirname, exist_ok=True)

    base_name, ext = os.path.splitext(os.path.basename(base_path))
    if not ext:
        ext = '.csv'

    if batch_size is None:
        target = os.path.join(dirname, f"{base_name}{suffix}{ext}")
        df.to_csv(target, index=False)
        logging.info(f"Wrote report to {target} ({len(df)} rows).")
        return [target]
    else:
        total = len(df)
        num_batches = (total + batch_size - 1) // batch_size
        paths = []
        for i in range(num_batches):
            chunk = df.iloc[i * batch_size:(i + 1) * batch_size]
            batch_name = f"{base_name}{suffix}_batch_{i+1}{ext}"
            target = os.path.join(dirname, batch_name)
            chunk.to_csv(target, index=False)
            logging.info(f"Wrote batch {i+1}/{num_batches} to {target} ({len(chunk)} rows).")
            paths.append(target)
        return paths


def main(config_path):
    cfg = load_config(config_path)
    logging.info("Loaded configuration.")

    batching_cfg = cfg.get('batching', {})
    batching_size = batching_cfg.get('size', None)
    if isinstance(batching_size, str) and batching_size.lower() in ('none', 'null', ''):
        batching_size = None
    logging.info(f"Effective batching size: {batching_size}")

    # Load base data
    sample_gddens = pd.read_csv(cfg['sample_gddens_csv'], skiprows=1)
    suffix = cfg.get('sample_id_suffix_to_strip', '')
    if suffix:
        sample_gddens['PATIENT_ID'] = sample_gddens.SAMPLE_ID.str.replace(f"{suffix}$", "", regex=True)
    else:
        sample_gddens['PATIENT_ID'] = sample_gddens.SAMPLE_ID

    tumor_types = pd.read_csv(cfg['tumor_types_file'], sep='\t')
    clinical_merged = read_clinical_sample_and_patient(cfg['clinical_sample_file'], cfg['clinical_patient_file'])
    patient_impact = pd.read_csv(cfg['patient_impact_file'], sep='\t', comment='#')
    patient_impact['GENDER'] = patient_impact.get('SEX', pd.Series(dtype=str))
    gdd_ens_fts = pd.read_csv(cfg['genie_features_file'])
    samples_df = pd.read_csv(cfg['clinical_sample_file'], sep='\t', comment='#')
    sample_val = samples_df[
        ~samples_df.PATIENT_ID.isin(sample_gddens.PATIENT_ID.unique())
        & samples_df.CANCER_TYPE_DETAILED.isin(tumor_types['CANCER_TYPE_DETAILED'].unique())
    ]
    val_set = set(sample_val['SAMPLE_ID'].unique())

    # Only get samples that intersect with samples present in feature files
    gdd_sample_ids = set(gdd_ens_fts['SAMPLE_ID'].unique())
    val_set = val_set & gdd_sample_ids
    logging.info(f"Validation set size (unique samples, intersected with gdd_ens_fts): {len(val_set)}")


    # Mutations
    mutations = pd.read_csv(cfg['mutations_file'], sep='\t', comment='#')
    mutations_report_str = create_mutation_report_str(mutations)
    logging.info("Created mutation report strings.")

    # Gene-level CNA
    cna = pd.read_csv(cfg['cna_file'], sep='\t')
    cna = cna.set_index('Hugo_Symbol').T
    cna_report_str = create_gene_level_cna(cna)
    logging.info("Created gene-level CNA report strings.")

    # Fusions
    fusions = pd.read_csv(cfg['fusions_file'], sep='\t')
    fusions_report_str = (
        fusions[['Sample_Id', 'Annotation']]
        .dropna()
        .groupby('Sample_Id')['Annotation']
        .apply(lambda rows: '\n'.join(rows))
        .reset_index()
        .rename(columns={'Sample_Id': 'SAMPLE_ID', 'Annotation': 'fusion_report_str'})
    )
    logging.info("Created fusion report strings.")

    # Chromosomal arm-level
    chr_cols = cfg.get('chromosomal_columns', [])
    chr_df = gdd_ens_fts.loc[:, chr_cols] if chr_cols else pd.DataFrame()
    chr_df = chr_df[(chr_df > 0).any(axis=1)]
    chr_report = create_chr_level_report(chr_df)
    chr_report['SAMPLE_ID'] = gdd_ens_fts['SAMPLE_ID'].loc[chr_report['SAMPLE_IDX']].values
    chr_report = chr_report[['SAMPLE_ID', 'chr_report_str']]
    logging.info("Created chromosomal arm-level report strings.")

    # Merge for validation samples
    val_reports = clinical_merged[clinical_merged.SAMPLE_ID.isin(val_set)].copy()
    val_reports = val_reports.merge(mutations_report_str, on='SAMPLE_ID', how='left')
    val_reports = val_reports.merge(cna_report_str, on='SAMPLE_ID', how='left')
    val_reports = val_reports.merge(fusions_report_str, on='SAMPLE_ID', how='left')
    val_reports = val_reports.merge(chr_report, on='SAMPLE_ID', how='left')

    # Compose REPORT column
    val_reports['REPORT'] = (
        val_reports['sex_report_str'].fillna('') + '\n'
        + val_reports['report_str'].fillna('') + '\n\n'
        + 'The following somatic copy number abnormalities were detected:\n'
        + val_reports['cna_report_str'].fillna('') + '\n\n'
        + 'The following somatic fusions were detected:\n'
        + val_reports['fusion_report_str'].fillna('') + '\n\n'
        + 'The following chromosomal arm-level changes were detected:\n'
        + val_reports['chr_report_str'].fillna('') + '\n\n'
    )

    out_df = val_reports[[
        'SAMPLE_ID',
        'REPORT',
        'CANCER_TYPE',
        'CANCER_TYPE_DETAILED',
        'SAMPLE_TYPE_DETAILED'
    ]]

    # Write full reports with batching behavior
    full_paths = write_batched(out_df, cfg['output_report_csv'], batching_size)

    # Subset if enabled
    subset_cfg = cfg.get('subset', {})
    if subset_cfg.get('enabled', False):
        group_by = subset_cfg.get('group_by', 'CANCER_TYPE')
        frac = subset_cfg.get('frac', 0.05)
        seed = subset_cfg.get('shuffle_seed', 42)
        if group_by not in out_df.columns:
            logging.warning(f"Requested subset group_by '{group_by}' not in columns; skipping subsetting.")
        else:
            val_subset = out_df.groupby(group_by, group_keys=False).sample(
                frac=frac, random_state=seed
            )
            val_subset = val_subset.sample(frac=1, random_state=seed)
            subset_base = os.path.splitext(cfg['output_report_csv'])[0] + "_subset.csv"
            subset_paths = write_batched(val_subset, subset_base, batching_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate genomic reports with batching and YAML config.")
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    args = parser.parse_args()

    try:
        main(args.config)
    except Exception:
        logging.exception("Failed to generate reports.")
        sys.exit(1)
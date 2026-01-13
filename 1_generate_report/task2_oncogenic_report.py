"""
Task 2: Oncogenic Mutation Report Generation
Processes oncogenic mutation data from MSK IMPACT solid/heme dataset.

Configured via YAML file. See configs/task2_oncogenic_config.yaml for details.
"""

import pandas as pd
import numpy as np
import os
import argparse
import yaml
import logging
from pathlib import Path


def load_config(config_path):
    """Load and parse YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_logging(config):
    """Setup logging based on config."""
    log_config = config.get('logging', {})
    logging.basicConfig(
        level=getattr(logging, log_config.get('level', 'INFO')),
        format=log_config.get('format', '[%(asctime)s] %(levelname)s: %(message)s')
    )


def load_and_process_mutations(config):
    """Load and process mutation data with oncogenic annotations."""
    logging.info("=" * 80)
    logging.info("Loading and processing mutation data")
    logging.info("=" * 80)

    data_dir = config['data_dir']
    input_files = config['input_files']

    # Load sample data
    sample_data = pd.read_csv(
        f'{data_dir}/{input_files["clinical_sample"]}',
        sep='\t',
        comment='#',
        low_memory=False
    )

    patient_data = pd.read_csv(
        f'{data_dir}/{input_files["clinical_patient"]}',
        sep='\t',
        comment='#',
        low_memory=False
    )

    # Handle GENDER/SEX column
    if 'SEX' in patient_data.columns:
        patient_data['GENDER'] = patient_data['SEX']

    # Load mutations
    mutations = pd.read_csv(
        f'{data_dir}/{input_files["mutations_annotated"]}',
        comment='#',
        low_memory=False
    )

    logging.info(f"Loaded {len(mutations)} mutations")

    # Add mutation_id if not present
    if "mutation_id" not in mutations.columns:
        mutations['mutation_id'] = mutations.index

    # Process ONCOGENIC labels
    logging.info("Processing oncogenic labels...")

    oncogenic_config = config['oncogenic_labels']

    # Drop resistance mutations
    for label in oncogenic_config['drop']:
        mutations = mutations[mutations['ONCOGENIC'].str.strip().str.casefold() != label.lower()]

    # Normalize ONCOGENIC labels
    mutations['ONCOGENIC'] = mutations['ONCOGENIC'].replace(oncogenic_config['normalize'])

    # Use ClinVar and dbSNP to reclassify VUS as Benign
    if oncogenic_config['reclassify_vus']['clinvar_benign']:
        clinvar_benign_flag = mutations['ClinVar_label'].astype(str).str.strip().str.casefold().eq('benign')
        mask_clinvar = mutations['ONCOGENIC'].eq('VUS') & clinvar_benign_flag
        mutations.loc[mask_clinvar, 'ONCOGENIC'] = 'Benign'

    if oncogenic_config['reclassify_vus']['dbsnp_benign']:
        dbSNP_benign_flag = mutations['dbSNP_class'].astype(str).str.strip().str.casefold().eq('benign')
        mask_dbSNP = mutations['ONCOGENIC'].eq('VUS') & dbSNP_benign_flag
        mutations.loc[mask_dbSNP, 'ONCOGENIC'] = 'Benign'

    # Rename and merge with clinical data
    mutations = mutations.rename(columns={"Tumor_Sample_Barcode": "SAMPLE_ID"})

    features_in_report = config['report_features']
    mutations = mutations.merge(
        sample_data[["SAMPLE_ID", "PATIENT_ID"] + features_in_report],
        on=["SAMPLE_ID"],
        how="left"
    )

    # Filter by AlphaMissense if configured
    if config['alphamissense']['filter_vus']:
        logging.info(f"Before AlphaMissense filter: {len(mutations)} mutations")
        mutations = mutations[mutations['AlphaMissense_label'] != 'VUS']
        logging.info(f"After AlphaMissense filter: {len(mutations)} mutations")

    logging.info(f"\nONCOGENIC distribution:")
    logging.info(mutations['ONCOGENIC'].value_counts())

    return mutations, sample_data


def filter_genes_with_both_types(mutations, config):
    """Filter for genes that have both Benign and Oncogenic mutations."""
    logging.info("\n" + "=" * 80)
    logging.info("Filtering genes with both Benign and Oncogenic mutations")
    logging.info("=" * 80)

    balanced_config = config['balanced_dataset']

    # Deduplicate first
    dedup_cols = balanced_config['dedup_columns']
    mutations_dedup = mutations.drop_duplicates(subset=dedup_cols)

    # Find genes with both types if required
    if balanced_config['require_both_types']:
        gene_oncogenic = mutations_dedup.groupby('Hugo_Symbol')['ONCOGENIC'].apply(set).reset_index()
        genes_with_both = gene_oncogenic[
            gene_oncogenic['ONCOGENIC'].apply(lambda x: 'Benign' in x and 'Oncogenic' in x)
        ]['Hugo_Symbol'].tolist()

        logging.info(f"Found {len(genes_with_both)} genes with both Benign and Oncogenic mutations")

        mutations_filtered = mutations_dedup[mutations_dedup['Hugo_Symbol'].isin(genes_with_both)]
    else:
        mutations_filtered = mutations_dedup

    logging.info(f"\nFiltered mutations distribution:")
    logging.info(mutations_filtered['ONCOGENIC'].value_counts())

    return mutations_filtered


def create_balanced_dataset(mutations_filtered, config):
    """Create balanced dataset with configurable Oncogenic:Benign ratio."""
    logging.info("\n" + "=" * 80)
    logging.info("Creating balanced dataset")
    logging.info("=" * 80)

    balanced_config = config['balanced_dataset']
    ratio = balanced_config['oncogenic_to_benign_ratio']
    seed = balanced_config['random_seed']

    benign_df = mutations_filtered[mutations_filtered['ONCOGENIC'] == 'Benign']
    oncogenic_df = mutations_filtered[mutations_filtered['ONCOGENIC'] == 'Oncogenic']

    n_benign = len(benign_df)
    n_oncogenic_target = n_benign * ratio

    logging.info(f"Benign samples: {n_benign}")
    logging.info(f"Oncogenic samples available: {len(oncogenic_df)}")
    logging.info(f"Oncogenic samples target (ratio {ratio}:1): {n_oncogenic_target}")

    # Calculate distribution of Benign per gene-cancer combination
    benign_per_combo = benign_df.groupby(['Hugo_Symbol', 'CANCER_TYPE']).size().reset_index(name='count')

    # Sample Oncogenic proportionally
    sampled_oncogenic = []

    for _, row in benign_per_combo.iterrows():
        gene = row['Hugo_Symbol']
        cancer = row['CANCER_TYPE']
        n_benign_combo = row['count']

        # Calculate proportion
        proportion = n_benign_combo / n_benign
        target_for_combo = int(n_oncogenic_target * proportion)

        # Get available Oncogenic for this combination
        combo_oncogenic = oncogenic_df[
            (oncogenic_df['Hugo_Symbol'] == gene) &
            (oncogenic_df['CANCER_TYPE'] == cancer)
        ]

        # Sample
        n_to_sample = min(target_for_combo, len(combo_oncogenic))

        if n_to_sample > 0:
            sampled = combo_oncogenic.sample(n=n_to_sample, random_state=seed)
            sampled_oncogenic.append(sampled)

    # Combine
    oncogenic_balanced = pd.concat(sampled_oncogenic, ignore_index=True)

    # Fill remaining if needed
    if len(oncogenic_balanced) < n_oncogenic_target:
        remaining = oncogenic_df[~oncogenic_df.index.isin(oncogenic_balanced.index)]
        n_additional = min(n_oncogenic_target - len(oncogenic_balanced), len(remaining))
        if n_additional > 0:
            additional = remaining.sample(n=n_additional, random_state=seed)
            oncogenic_balanced = pd.concat([oncogenic_balanced, additional], ignore_index=True)

    # Combine all
    balanced_df = pd.concat([benign_df, oncogenic_balanced], ignore_index=True)

    logging.info(f"\nBalanced dataset distribution:")
    logging.info(balanced_df['ONCOGENIC'].value_counts())

    return balanced_df


def create_report_strings(balanced_df, config):
    """Create report strings from mutation data."""
    logging.info("\n" + "=" * 80)
    logging.info("Creating report strings")
    logging.info("=" * 80)

    balanced_df['Mutation_Status'] = balanced_df['Mutation_Status'].str.upper()

    # Required columns
    required_cols = config['required_columns']
    balanced_df = balanced_df.dropna(subset=required_cols).reset_index(drop=True)

    # Create report string
    report_config = config['report_format']

    if report_config['include_exon']:
        balanced_df['report_str'] = (
            balanced_df['Hugo_Symbol'] + ' exon '
            + balanced_df['Exon_Number'].fillna('NA').str.split('/').str[0] + ' '
            + balanced_df['Variant_Classification'].str.replace('_', ' ' if report_config['replace_underscore'] else '_') + ' '
            + balanced_df['HGVSp_Short']
        )
    else:
        balanced_df['report_str'] = (
            balanced_df['Hugo_Symbol'] + ' '
            + balanced_df['Variant_Classification'].str.replace('_', ' ' if report_config['replace_underscore'] else '_') + ' '
            + balanced_df['HGVSp_Short']
        )

    features_in_report = config['report_features']
    balanced_df = balanced_df.dropna(subset=features_in_report + ['ONCOGENIC']).reset_index(drop=True)

    balanced_df["report_str"] = (
        balanced_df["report_str"]
        + "\nCANCER_TYPE: " + balanced_df["CANCER_TYPE"].fillna("").astype(str)
        + "\nCANCER_TYPE_DETAILED: " + balanced_df["CANCER_TYPE_DETAILED"].fillna("").astype(str)
        + "\n"
    )

    logging.info(f"Created {len(balanced_df)} report strings")

    return balanced_df


def save_in_batches(df, output_dir, config, prefix="batch"):
    """Save dataframe in batches."""
    os.makedirs(output_dir, exist_ok=True)

    chunk_size = config['chunk_size']['balanced']
    output_cols = config['output_columns']

    num_chunks = (len(df) + chunk_size - 1) // chunk_size

    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(df))
        chunk = df.iloc[start_idx:end_idx]

        filename = f"{output_dir}/{prefix}_{i + 1}.csv"
        chunk[output_cols].to_csv(filename, index=False)

        logging.info(f"Saved chunk {i + 1}/{num_chunks} to {filename} ({len(chunk)} rows)")


def extract_nsclc_subset(mutations, sample_data, config):
    """Extract Non-Small Cell Lung Cancer subset."""
    logging.info("\n" + "=" * 80)
    logging.info("Extracting NSCLC subset")
    logging.info("=" * 80)

    cancer_type_filter = config['nsclc_filter']['cancer_type']

    # Filter for NSCLC
    nsclc_samples = sample_data[sample_data['CANCER_TYPE'] == cancer_type_filter]
    nsclc_mutations = mutations[mutations['SAMPLE_ID'].isin(nsclc_samples['SAMPLE_ID'])]

    logging.info(f"Found {len(nsclc_mutations)} NSCLC mutations from {len(nsclc_samples)} samples")

    # Create report strings
    nsclc_report = create_report_strings(nsclc_mutations.copy(), config)

    # Save
    os.makedirs(config['output_nsclc'], exist_ok=True)
    chunk_size = config['chunk_size']['nsclc']
    output_cols = config['output_columns']

    num_chunks = (len(nsclc_report) + chunk_size - 1) // chunk_size

    prefix = config['output_files']['batch_prefix']
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(nsclc_report))
        chunk = nsclc_report.iloc[start_idx:end_idx]

        filename = f"{config['output_nsclc']}/{prefix}_{i + 1}.csv"
        chunk[output_cols].to_csv(filename, index=False)

        logging.info(f"Saved NSCLC chunk {i + 1}/{num_chunks} to {filename} ({len(chunk)} rows)")


def main():
    parser = argparse.ArgumentParser(description='Generate oncogenic mutation reports')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to YAML configuration file')

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    setup_logging(config)

    logging.info("Configuration loaded successfully")

    # Load data
    mutations, sample_data = load_and_process_mutations(config)

    # Generate balanced dataset
    if not config['skip_steps']['balanced']:
        mutations_filtered = filter_genes_with_both_types(mutations, config)
        balanced_df = create_balanced_dataset(mutations_filtered, config)
        balanced_report = create_report_strings(balanced_df.copy(), config)
        save_in_batches(balanced_report, config['output_balanced'], config,
                       prefix=config['output_files']['batch_prefix'])
    else:
        logging.info("Skipping balanced dataset generation")

    # Generate NSCLC subset
    if not config['skip_steps']['nsclc']:
        extract_nsclc_subset(mutations, sample_data, config)
    else:
        logging.info("Skipping NSCLC subset generation")

    logging.info("\n" + "=" * 80)
    logging.info("All oncogenic reports generated successfully!")
    logging.info("=" * 80)


if __name__ == "__main__":
    main()

"""
Task 1: Mutation Report Generation
Processes mutation data from MSK and GENIE datasets to generate mutation reports.

Configured via YAML file. See configs/task1_mutation_config.yaml for details.
"""

import pandas as pd
import numpy as np
import os
import re
import math
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


def normalize_sample_id(sample_id, pattern=r'-IM\d*$'):
    """
    Remove -IM suffix from sample IDs for comparison.
    E.g., P-0061179-N01-IM7 -> P-0061179-N01
    """
    if pd.isna(sample_id):
        return sample_id
    return re.sub(pattern, '', str(sample_id))


def genie_to_msk_format(patient_id, prefix='GENIE-MSK-'):
    """
    Convert GENIE-MSK patient ID to MSK format.
    E.g., GENIE-MSK-P-0000004 -> P-0000004
    """
    if pd.isna(patient_id):
        return patient_id
    patient_id_str = str(patient_id)
    if patient_id_str.startswith(prefix):
        return patient_id_str.replace(prefix, '', 1)
    return patient_id_str


def msk_to_genie_format(patient_id, prefix='GENIE-MSK-'):
    """
    Convert MSK patient ID to GENIE format.
    E.g., P-0000004 -> GENIE-MSK-P-0000004
    """
    if pd.isna(patient_id):
        return patient_id
    patient_id_str = str(patient_id)
    if not patient_id_str.startswith(prefix):
        return f'{prefix}{patient_id_str}'
    return patient_id_str


def filter_msk_new_samples(config):
    """Filter MSK 2023 samples to get only new samples not in 2020."""
    logging.info("=" * 80)
    logging.info("STEP 1: Filtering MSK 2023 for new samples")
    logging.info("=" * 80)

    msk_ch_2020_dir = config['msk_ch_2020_dir']
    msk_ch_2023_dir = config['msk_ch_2023_dir']
    output_dir = config['msk_new_only_dir']
    input_files = config['input_files']

    # Read 2020 data
    logging.info(f"Reading 2020 data from {msk_ch_2020_dir}...")
    clinical_2020 = pd.read_csv(
        os.path.join(msk_ch_2020_dir, input_files['clinical_sample']),
        sep="\t",
        comment="#"
    )
    mutations_2020 = pd.read_csv(
        os.path.join(msk_ch_2020_dir, input_files['mutations']),
        sep="\t",
        comment="#"
    )

    samples_2020_clinical = set(clinical_2020['SAMPLE_ID'].dropna())
    samples_2020_mutations = set(mutations_2020['Tumor_Sample_Barcode'].dropna())
    samples_2020 = samples_2020_clinical.union(samples_2020_mutations)

    logging.info(f"2020 dataset:")
    logging.info(f"  - Samples from clinical file: {len(samples_2020_clinical)}")
    logging.info(f"  - Samples from mutations file: {len(samples_2020_mutations)}")
    logging.info(f"  - Total unique samples: {len(samples_2020)}")

    # Read 2023 data
    logging.info(f"\nReading 2023 data from {msk_ch_2023_dir}...")
    clinical_2023 = pd.read_csv(
        os.path.join(msk_ch_2023_dir, input_files['clinical_sample']),
        sep="\t",
        comment="#"
    )
    mutations_2023 = pd.read_csv(
        os.path.join(msk_ch_2023_dir, input_files['mutations']),
        sep="\t",
        comment="#"
    )

    # Normalize sample IDs if enabled
    if config['normalize_sample_ids']['enabled']:
        pattern = config['normalize_sample_ids']['pattern']
        clinical_2023['SAMPLE_ID_normalized'] = clinical_2023['SAMPLE_ID'].apply(
            lambda x: normalize_sample_id(x, pattern)
        )
        mutations_2023['Tumor_Sample_Barcode_normalized'] = mutations_2023['Tumor_Sample_Barcode'].apply(
            lambda x: normalize_sample_id(x, pattern)
        )
    else:
        clinical_2023['SAMPLE_ID_normalized'] = clinical_2023['SAMPLE_ID']
        mutations_2023['Tumor_Sample_Barcode_normalized'] = mutations_2023['Tumor_Sample_Barcode']

    samples_2023_clinical = set(clinical_2023['SAMPLE_ID'].dropna())
    samples_2023_mutations = set(mutations_2023['Tumor_Sample_Barcode'].dropna())
    samples_2023 = samples_2023_clinical.union(samples_2023_mutations)

    samples_2023_clinical_normalized = set(clinical_2023['SAMPLE_ID_normalized'].dropna())
    samples_2023_mutations_normalized = set(mutations_2023['Tumor_Sample_Barcode_normalized'].dropna())

    logging.info(f"2023 dataset:")
    logging.info(f"  - Samples from clinical file: {len(samples_2023_clinical)}")
    logging.info(f"  - Samples from mutations file: {len(samples_2023_mutations)}")
    logging.info(f"  - Total unique samples: {len(samples_2023)}")

    # Filter for new samples
    new_samples_clinical = clinical_2023[~clinical_2023['SAMPLE_ID_normalized'].isin(samples_2020)]
    new_sample_ids = set(new_samples_clinical['SAMPLE_ID'])

    new_samples_mutations_normalized = samples_2023_mutations_normalized - samples_2020 - samples_2023_clinical_normalized
    new_samples_mutations_only_original = set(
        mutations_2023[mutations_2023['Tumor_Sample_Barcode_normalized'].isin(new_samples_mutations_normalized)]['Tumor_Sample_Barcode']
    )

    logging.info(f"\nNew samples in 2023 (not in 2020):")
    logging.info(f"  - From clinical file: {len(new_samples_clinical)}")
    logging.info(f"  - Only in mutations file: {len(new_samples_mutations_only_original)}")
    logging.info(f"  - Total unique new samples: {len(new_sample_ids) + len(new_samples_mutations_only_original)}")

    # Get all new sample IDs
    all_new_sample_ids = new_sample_ids.union(new_samples_mutations_only_original)
    new_mutations = mutations_2023[mutations_2023['Tumor_Sample_Barcode'].isin(all_new_sample_ids)]

    logging.info(f"Mutations from new samples: {len(new_mutations)}")

    # Save filtered data
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"\nSaving filtered data to {output_dir}...")

    new_samples_clinical_to_save = new_samples_clinical.drop(columns=['SAMPLE_ID_normalized'])
    new_samples_clinical_to_save.to_csv(
        os.path.join(output_dir, input_files['clinical_sample']),
        sep="\t",
        index=False
    )

    new_mutations_to_save = new_mutations.drop(columns=['Tumor_Sample_Barcode_normalized'])
    new_mutations_to_save.to_csv(
        os.path.join(output_dir, input_files['mutations']),
        sep="\t",
        index=False
    )

    logging.info(f"Done! Files saved to {output_dir}/")


def filter_genie_for_msk_patients(config):
    """Filter GENIE data to only include MSK patients from new samples."""
    logging.info("\n" + "=" * 80)
    logging.info("STEP 2: Filtering GENIE data for MSK new patients")
    logging.info("=" * 80)

    msk_new_dir = config['msk_new_only_dir']
    genie_dir = config['genie_dir']
    output_dir = config['genie_msk_new_only_dir']
    input_files = config['input_files']
    prefix = config['patient_id_conversion']['genie_prefix']

    # Read MSK new only data
    logging.info(f"Reading MSK new only data from {msk_new_dir}...")
    msk_clinical = pd.read_csv(
        os.path.join(msk_new_dir, input_files['clinical_sample']),
        sep="\t",
        comment="#"
    )

    msk_patients = set(msk_clinical['PATIENT_ID'].dropna())
    logging.info(f"MSK new only dataset: {len(msk_patients)} unique patients")

    # Convert to GENIE format
    msk_patients_genie_format = {msk_to_genie_format(pid, prefix) for pid in msk_patients}

    # Read GENIE data
    logging.info(f"\nReading GENIE data from {genie_dir}...")
    genie_clinical = pd.read_csv(
        os.path.join(genie_dir, input_files['clinical_sample']),
        sep="\t",
        comment="#"
    )
    genie_mutations = pd.read_csv(
        os.path.join(genie_dir, input_files['mutations_extended']),
        sep="\t",
        comment="#",
        low_memory=False
    )

    logging.info(f"GENIE dataset: {len(genie_clinical)} samples, {len(genie_mutations)} mutations")

    # Filter GENIE data
    genie_clinical_msk = genie_clinical[genie_clinical['PATIENT_ID'].isin(msk_patients_genie_format)]

    # Map sample to patient for mutations
    genie_sample_to_patient = dict(zip(genie_clinical['SAMPLE_ID'], genie_clinical['PATIENT_ID']))
    genie_mutations['PATIENT_ID'] = genie_mutations['Tumor_Sample_Barcode'].map(genie_sample_to_patient)
    genie_mutations_msk = genie_mutations[genie_mutations['PATIENT_ID'].isin(msk_patients_genie_format)]

    logging.info(f"\nMatched {len(genie_clinical_msk['PATIENT_ID'].unique())} GENIE-MSK patients")
    logging.info(f"  - {len(genie_clinical_msk)} samples")
    logging.info(f"  - {len(genie_mutations_msk)} mutations")

    # Save
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"\nSaving filtered GENIE data to {output_dir}...")

    genie_clinical_msk.to_csv(
        os.path.join(output_dir, input_files['clinical_sample']),
        sep="\t",
        index=False
    )

    genie_mutations_msk_to_save = genie_mutations_msk.drop(columns=['PATIENT_ID'])
    genie_mutations_msk_to_save.to_csv(
        os.path.join(output_dir, input_files['mutations_extended']),
        sep="\t",
        index=False
    )

    logging.info(f"Done!")


def generate_mutation_reports(data_directory, sample_data, output_directory, config,
                              mutations_file=None, include_vaf=True, aggregate=True,
                              mutation_status='TUMOR-SOMATIC'):
    """Generate mutation reports from data."""
    logging.info("\n" + "=" * 80)
    logging.info(f"Generating mutation reports (VAF={include_vaf}, Aggregate={aggregate})")
    logging.info("=" * 80)

    chunk_size = config['chunk_size']
    required_cols = config['required_columns']
    report_features = config['report_features']

    if mutations_file is None:
        mutations_file = config["input_files"]["mutations_extended"]

    # Read mutations
    mutations = pd.read_csv(
        f'{data_directory}/{mutations_file}',
        sep='\t',
        comment='#',
        low_memory=False
    )
    mutations['Mutation_Status'] = mutations['Mutation_Status'].str.upper()

    # Calculate tumor VAF
    mutations['t_VAF'] = (
        mutations['t_alt_count'] /
        (mutations['t_alt_count'] + mutations['t_ref_count'])
    )

    # Cast position columns to int for locus string building
    mutations['Start_Position'] = mutations['Start_Position'].astype(int)
    mutations['End_Position'] = mutations['End_Position'].astype(int)

    # Clean data
    mutations = mutations.dropna(subset=required_cols).reset_index(drop=True)
    mutations = mutations.drop_duplicates(subset=required_cols).reset_index(drop=True)

    # Build locus string: chr17:7577120 for SNVs, chr17:7577120-7577125 for indels
    locus = (
        'chr' + mutations['Chromosome'].astype(str)
        + ':' + mutations['Start_Position'].astype(str)
        + np.where(
            mutations['Start_Position'] == mutations['End_Position'],
            '',
            '-' + mutations['End_Position'].astype(str),
        )
    )

    # Create report string — locus is always included; VAF is optional
    if include_vaf:
        mutations["report_str"] = (
            mutations["Hugo_Symbol"] + " "
            + mutations['Variant_Classification'].str.replace('_', ' ') + " "
            + mutations['Variant_Type']
            + " (" + mutations["Reference_Allele"] + ">" + mutations["Tumor_Seq_Allele2"] + ")"
            + " at " + locus
            + ", Tumor Variant Allele Fraction: " + mutations["t_VAF"].astype(str)
        )
    else:
        mutations["report_str"] = (
            mutations["Hugo_Symbol"] + " "
            + mutations['Variant_Classification'].str.replace('_', ' ') + " "
            + mutations['Variant_Type']
            + " (" + mutations["Reference_Allele"] + ">" + mutations["Tumor_Seq_Allele2"] + ")"
            + " at " + locus
        )

    mutations["PATIENT_ID"] = mutations["Tumor_Sample_Barcode"].str.split("-", n=2).str[:2].str.join("-")
    mutations = mutations.rename(columns={"Tumor_Sample_Barcode": "SAMPLE_ID"})

    # Filter mutations
    mutations = (
        mutations
        .loc[mutations['Mutation_Status'].notna()
             & (mutations['Mutation_Status'] != 'UNKNOWN')]
        .reset_index(drop=True)
    )
    mutations = mutations[mutations["t_VAF"] != 0]

    # Check for mixed mutation status
    status_counts = mutations.groupby('SAMPLE_ID')['Mutation_Status'].nunique()
    mixed_status_samples = status_counts[status_counts > 1].index.tolist()
    logging.info(f"Samples with mixed Mutation_Status: {len(mixed_status_samples)}")

    mutations = mutations.loc[~mutations['SAMPLE_ID'].isin(mixed_status_samples)].reset_index(drop=True)

    # Aggregate or not
    if aggregate:
        mutations_report_str = (
            mutations[['SAMPLE_ID', 'report_str']]
            .fillna('')
            .groupby('SAMPLE_ID')['report_str']
            .apply(lambda rows: '\n'.join(rows))
            .reset_index()
        )
    else:
        mutations_report_str = mutations[['SAMPLE_ID', 'report_str'] + required_cols]

    # Merge with clinical data
    mutations_report_str = mutations_report_str.merge(
        sample_data[["SAMPLE_ID", "PATIENT_ID"] + report_features],
        on=["SAMPLE_ID"],
        how="left"
    )
    mutations_report_str = mutations_report_str.dropna(subset=report_features).reset_index(drop=True)

    # Append report features dynamically from config
    for feat in report_features:
        mutations_report_str["report_str"] += f"\n{feat}: " + mutations_report_str[feat].fillna("").astype(str)
    mutations_report_str["report_str"] += "\n"

    mutations_report_str['Mutation_Status'] = mutation_status

    logging.info(f"Generated {len(mutations_report_str)} mutation reports")

    # Save in chunks
    os.makedirs(output_directory, exist_ok=True)
    num_chunks = math.ceil(len(mutations_report_str) / chunk_size)

    # Determine prefix based on mutation status
    prefix = config['output_files']['genie_prefix'] if 'SOMATIC' in mutation_status else config['output_files']['msk_ch_prefix']

    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(mutations_report_str))
        chunk = mutations_report_str.iloc[start_idx:end_idx]

        filename = f'{output_directory}/{prefix}_{chunk_size}_{i+1}.csv'
        chunk.to_csv(filename, index=False)
        logging.info(f"Saved chunk {i+1}/{num_chunks} ({len(chunk)} samples) to {filename}")


def main():
    parser = argparse.ArgumentParser(description='Generate mutation reports from MSK and GENIE data')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to YAML configuration file')

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    setup_logging(config)

    logging.info("Configuration loaded successfully")

    # Step 1 & 2: Filter data
    if not config.get('skip_filtering', False):
        filter_msk_new_samples(config)
        filter_genie_for_msk_patients(config)
    else:
        logging.info("Skipping filtering steps (using existing intermediate files)")

    # Get variant configurations
    generate_variants = config['generate_variants']
    output_base_dir = config['output_base_dir']

    variants = []
    if generate_variants.get('with_vaf_aggregate'):
        variants.append(('yes', 'yes', ''))
    if generate_variants.get('with_vaf_noaggregate'):
        variants.append(('yes', 'no', '_aggregate'))
    if generate_variants.get('no_vaf_aggregate'):
        variants.append(('no', 'yes', '_noVAF_aggregate'))
    if generate_variants.get('no_vaf_noaggregate'):
        variants.append(('no', 'no', '_noVAF'))

    # Step 3: Generate reports for GENIE data
    logging.info("\n" + "=" * 80)
    logging.info("Processing GENIE data")
    logging.info("=" * 80)

    genie_msk_new_dir = config['genie_msk_new_only_dir']
    sample_data = pd.read_csv(
        f'{genie_msk_new_dir}/{config["input_files"]["clinical_sample"]}',
        sep='\t',
        comment='#'
    )

    for vaf, agg, suffix in variants:
        include_vaf = (vaf == 'yes')
        do_aggregate = (agg == 'yes')

        output_dir = f"{output_base_dir}/mutation_status_final{suffix}"

        generate_mutation_reports(
            genie_msk_new_dir,
            sample_data,
            output_dir,
            config,
            mutations_file=config['input_files']['mutations_extended'],
            include_vaf=include_vaf,
            aggregate=do_aggregate,
            mutation_status=config['mutation_status']['genie']
        )

    # Step 4: Generate reports for MSK CH data
    logging.info("\n" + "=" * 80)
    logging.info("Processing MSK CH (CHIP) data")
    logging.info("=" * 80)

    msk_new_dir = config['msk_new_only_dir']
    sample_data_ch = pd.read_csv(
        f'{msk_new_dir}/{config["input_files"]["clinical_sample"]}',
        sep='\t',
        comment='#'
    )

    for vaf, agg, suffix in variants:
        include_vaf = (vaf == 'yes')
        do_aggregate = (agg == 'yes')

        output_dir = f"{output_base_dir}/mutation_status_final{suffix}"

        generate_mutation_reports(
            msk_new_dir,
            sample_data_ch,
            output_dir,
            config,
            mutations_file=config['input_files']['mutations'],
            include_vaf=include_vaf,
            aggregate=do_aggregate,
            mutation_status=config['mutation_status']['msk_ch']
        )

    logging.info("\n" + "=" * 80)
    logging.info("All mutation reports generated successfully!")
    logging.info("=" * 80)


if __name__ == "__main__":
    main()

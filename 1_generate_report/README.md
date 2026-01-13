# Data Preprocessing and Report Generation Scripts

This directory contains refactored scripts for step 1 of the LLM-driven cancer type classification pipeline. These scripts were refactored from the original Jupyter notebooks in `/data1/morrisq/yuj13/llm_genomics/zeroshot_cancertype/`.

## Overview

Three main tasks for preprocessing genomic data and generating mutation reports:

1. **Task 1** ([task1_mutation_report.py](task1_mutation_report.py)) - Mutation report generation from MSK and GENIE data
2. **Task 2** ([task2_oncogenic_report.py](task2_oncogenic_report.py)) - Oncogenic mutation report generation
3. **Task 3** ([task3_cancer_type_report.py](task3_cancer_type_report.py)) - Comprehensive cancer type report generation

## Quick Start

### Task 1: Mutation Report Generation

Processes mutation data from MSK and GENIE datasets.

```bash
# Run with config file (recommended)
python task1_mutation_report.py --config ../configs/task1_mutation_config.yaml
```

**Configuration:** Edit [../configs/task1_mutation_config.yaml](../configs/task1_mutation_config.yaml) to customize:
- Input/output directories
- Which variants to generate (VAF/noVAF × aggregate/non-aggregate)
- Chunk size, filtering options
- Mutation status labels

**Output:** 4 report variants in `/data1/morrisq/yuj13/llm_genomics/mutation_report/mutation_status_final*`

### Task 2: Oncogenic Mutation Report Generation

Processes oncogenic mutation data from MSK IMPACT dataset.

```bash
# Run with config file (recommended)
python task2_oncogenic_report.py --config ../configs/task2_oncogenic_config.yaml
```

**Configuration:** Edit [../configs/task2_oncogenic_config.yaml](../configs/task2_oncogenic_config.yaml) to customize:
- Input/output directories
- Skip balanced or NSCLC generation
- Oncogenic:Benign ratio
- Annotation reclassification rules
- Chunk sizes

**Output:**
- Balanced dataset: `/data1/morrisq/yuj13/llm_genomics/mutation_report/impact_oncogenic_sample_new/`
- NSCLC subset: `/data1/morrisq/yuj13/llm_genomics/mutation_report/impact_oncogenic_nsclc_final/`

### Task 3: Comprehensive Cancer Type Report Generation

Main report generator for cancer type classification.

```bash
# Run with config file (recommended)
python task3_cancer_type_report.py --config ../configs/task3_cancer_type_config.yaml
```

**Configuration:** Edit [../configs/task3_cancer_type_config.yaml](../configs/task3_cancer_type_config.yaml) to customize:
- Data directory and input files
- Feature table path
- Subsetting and batching options
- Output file path

**Output:** CSV with comprehensive genomic reports (mutations, CNAs, fusions, arm changes, signatures)

## Script Details

### [task1_mutation_report.py](task1_mutation_report.py)

**Original source:** `/data1/morrisq/yuj13/llm_genomics/zeroshot_cancertype/stats_wrangling_mutation.ipynb`

**What it does:**
1. Filters MSK 2023 data to get samples not in MSK 2020
2. Filters GENIE data to match MSK new patient IDs
3. Generates mutation reports for both GENIE (TUMOR-SOMATIC) and MSK CH (CHIP) data
4. Supports 4 output variants: ±VAF, ±aggregation

**Key configuration options:**
- `skip_filtering`: Skip data filtering steps
- `generate_variants`: Control which of 4 variants to generate
- `chunk_size`: Output file chunk size (default: 7500)
- `mutation_status`: Labels for GENIE vs MSK CH data

### [task2_oncogenic_report.py](task2_oncogenic_report.py)

**Original source:** `/data1/morrisq/yuj13/llm_genomics/zeroshot_cancertype/stats_wrangling_oncogenic.ipynb`

**What it does:**
1. Loads MSK IMPACT solid/heme data with oncogenic annotations
2. Processes ONCOGENIC labels (uses OncoKB, ClinVar, dbSNP, AlphaMissense)
3. Filters for genes with both Benign and Oncogenic mutations
4. Creates balanced dataset (Oncogenic:Benign = 2:1)
5. Extracts NSCLC-specific subset

**Key configuration options:**
- `data_dir`: Input data directory
- `output_balanced` / `output_nsclc`: Output directories
- `skip_steps`: Skip balanced or NSCLC generation
- `chunk_size`: Output file chunk sizes (separate for balanced/NSCLC)
- `balanced_dataset.oncogenic_to_benign_ratio`: Ratio for balanced dataset (default: 2)
- `oncogenic_labels`: Normalization and reclassification rules

### [task3_cancer_type_report.py](task3_cancer_type_report.py)

**Original source:** `/data1/morrisq/yuj13/llm_genomics/llm_cancer_type_classification/generate_report.py` (already refactored from `/data1/morrisq/yuj13/llm_genomics/zeroshot_cancertype/stats_wrangling.ipynb`)

**What it does:**
1. Defines validation set (excludes training samples)
2. Generates comprehensive reports including:
   - Patient sex and sample site
   - MSI and TMB scores
   - Somatic mutations
   - Copy number alterations (CNAs)
   - Gene fusions
   - Chromosomal arm-level changes
   - Mutational signatures

**Key configuration options:**
- `data_directory`: Base path with variable substitution support
- `subset`: Enable/disable subsetting by cancer type
- `batching`: Control output file chunking
- Input file paths for clinical, mutations, CNAs, fusions

## Dependencies

- Python 3.8+
- pandas
- numpy
- pyyaml (for task 3)

Install from parent directory:
```bash
cd ..
pip install -r requirements.txt
```

## Data Flow

```
Task 1: MSK/GENIE Data → Filtered Data → Mutation Reports (4 variants)
                                        → CHIP Reports

Task 2: MSK IMPACT → Oncogenic Processing → Balanced Dataset (2:1)
                                          → NSCLC Subset

Task 3: Multiple Sources → Comprehensive Reports → Cancer Type Classification
```

## Output Formats

All tasks output CSV files with batching support for large datasets.

**Task 1 output columns:**
- `SAMPLE_ID`, `PATIENT_ID`, `report_str`, `CANCER_TYPE`, `CANCER_TYPE_DETAILED`, `Mutation_Status`

**Task 2 output columns:**
- `mutation_id`, `SAMPLE_ID`, `ONCOGENIC`, `AlphaMissense_label`, `am_pathogenicity`, `Hugo_Symbol`, `HGVSp_Short`, `CANCER_TYPE`, `report_str`

**Task 3 output columns:**
- `SAMPLE_ID`, `REPORT`, `CANCER_TYPE`, `CANCER_TYPE_DETAILED`, `SAMPLE_TYPE_DETAILED`

## Notes

- All scripts support chunked output for handling large datasets
- Task 1 generates both GENIE and MSK CH (CHIP) reports
- Task 2 uses multiple annotation sources: AlphaMissense, OncoKB, ClinVar, dbSNP
- Task 3 integrates with the GDD-ENS feature generation workflow

## See Also

- Main README: [../README.md](../README.md)
- Configuration examples: [../configs/](../configs/)
- Original notebooks: `/data1/morrisq/yuj13/llm_genomics/zeroshot_cancertype/`

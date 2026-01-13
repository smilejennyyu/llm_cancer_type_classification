# Configuration Files

This directory contains YAML configuration files for all data preprocessing and report generation tasks.

## Configuration Files

### [task1_mutation_config.yaml](task1_mutation_config.yaml)
Configuration for Task 1: Mutation Report Generation

**Key Settings:**
- **Input directories**: MSK 2020, MSK 2023, GENIE v17.0 paths
- **Output variants**: Enable/disable 4 report variants (VAF/noVAF × aggregate/non-aggregate)
- **Processing options**: Filtering, chunk size, normalization settings
- **Mutation status labels**: Separate labels for GENIE and MSK CH data

**Usage:**
```bash
python 1_generate_report/task1_mutation_report.py --config configs/task1_mutation_config.yaml
```

**Customization Examples:**
```yaml
# Skip filtering if intermediate files exist
skip_filtering: true

# Generate only specific variants
generate_variants:
  with_vaf_aggregate: true
  with_vaf_noaggregate: false
  no_vaf_aggregate: false
  no_vaf_noaggregate: true

# Change chunk size
chunk_size: 10000
```

---

### [task2_oncogenic_config.yaml](task2_oncogenic_config.yaml)
Configuration for Task 2: Oncogenic Mutation Report Generation

**Key Settings:**
- **Input directory**: MSK IMPACT solid/heme data path
- **Output directories**: Balanced dataset and NSCLC subset paths
- **Oncogenic labels**: Normalization rules, reclassification settings
- **Balanced dataset**: Oncogenic:Benign ratio, random seed, deduplication criteria
- **Filtering**: AlphaMissense, ClinVar, dbSNP annotations

**Usage:**
```bash
python 1_generate_report/task2_oncogenic_report.py --config configs/task2_oncogenic_config.yaml
```

**Customization Examples:**
```yaml
# Skip specific processing steps
skip_steps:
  balanced: false
  nsclc: true   # Skip NSCLC subset generation

# Change oncogenic:benign ratio
balanced_dataset:
  oncogenic_to_benign_ratio: 3  # 3:1 instead of 2:1

# Disable external annotation reclassification
oncogenic_labels:
  reclassify_vus:
    clinvar_benign: false
    dbsnp_benign: false

# Change chunk sizes
chunk_size:
  balanced: 5000
  nsclc: 15000
```

---

### [task3_cancer_type_config.yaml](task3_cancer_type_config.yaml)
Configuration for Task 3: Comprehensive Cancer Type Report Generation

**Key Settings:**
- **Data directory**: Base path for GENIE/MSK data (use path templating)
- **Input files**: Clinical, mutations, CNA, fusions file paths
- **Features file**: GDD-ENS feature table path
- **Subsetting**: Enable/disable subsetting, grouping, fraction
- **Batching**: Control output file chunking

**Usage:**
```bash
python 1_generate_report/task3_cancer_type_report.py --config configs/task3_cancer_type_config.yaml
```

**Path Templating:**
The config supports variable substitution using `{variable_name}` syntax:
```yaml
data_directory: "/path/to/your/data"
clinical_sample_file: "{data_directory}/data_clinical_sample.txt"
mutations_file: "{data_directory}/data_mutations_extended.txt"
```

**Customization Examples:**
```yaml
# Enable subsetting
subset:
  enabled: true
  group_by: "CANCER_TYPE_DETAILED"  # or "CANCER_TYPE"
  frac: 0.1  # 10% of samples

# Enable batching
batching:
  size: 5000  # Split output into files of 5000 rows each

# Disable batching (single file output)
batching:
  size: null
```

---

## Configuration File Structure

All configuration files follow a similar structure:

### Common Sections

1. **Input/Output Paths**
   ```yaml
   data_dir: "/path/to/input/data"
   output_dir: "/path/to/output"
   ```

2. **Processing Options**
   ```yaml
   skip_filtering: false
   chunk_size: 7500
   ```

3. **File Naming Conventions**
   ```yaml
   input_files:
     clinical_sample: "data_clinical_sample.txt"
     mutations: "data_mutations.txt"
   output_files:
     batch_prefix: "batch"
   ```

4. **Logging**
   ```yaml
   logging:
     level: "INFO"  # DEBUG, INFO, WARNING, ERROR
     format: "[%(asctime)s] %(levelname)s: %(message)s"
   ```

---

## Tips for Configuration

### 1. Path Management
- Use absolute paths for reliability
- Use path templating (`{variable}`) to avoid repetition
- Ensure output directories have write permissions

### 2. Performance Tuning
- **Chunk size**: Larger chunks = faster processing, but more memory
  - Small datasets: 5000-10000
  - Large datasets: 2000-5000
- **Skip flags**: Use when intermediate files exist to save time

### 3. Reproducibility
- Set `random_seed` for reproducible sampling (Task 2)
- Document any deviations from default configs
- Version control your config files

### 4. Testing
- Start with a subset of data using `subset.enabled: true`
- Use `logging.level: DEBUG` for troubleshooting
- Validate output paths before running full dataset

---

## Common Configuration Patterns

### Pattern 1: Development/Testing
```yaml
# Use smaller chunk sizes and enable subsetting
chunk_size: 1000
subset:
  enabled: true
  frac: 0.01  # 1% of data
logging:
  level: "DEBUG"
```

### Pattern 2: Production Run
```yaml
# Optimize for performance
chunk_size: 7500
skip_filtering: true  # If intermediate files exist
subset:
  enabled: false
logging:
  level: "INFO"
```

### Pattern 3: Custom Analysis
```yaml
# Task 2: Focus on specific cancer type
nsclc_filter:
  cancer_type: "Breast Cancer"  # Change from NSCLC

# Task 1: Generate only VAF reports
generate_variants:
  with_vaf_aggregate: true
  with_vaf_noaggregate: true
  no_vaf_aggregate: false
  no_vaf_noaggregate: false
```

---

## Environment Variables

You can use environment variables in your configs (requires code modification):

```yaml
data_directory: "${DATA_ROOT}/genie_data_v17.0"
output_base_dir: "${OUTPUT_ROOT}/mutation_report"
```

Then set in your shell:
```bash
export DATA_ROOT="/data1/morrisq/yuj13/llm_genomics"
export OUTPUT_ROOT="/data1/morrisq/yuj13/llm_genomics"
```

---

## Validation

Before running, validate your config:

1. Check all paths exist:
   ```bash
   ls /data1/morrisq/yuj13/llm_genomics/msk_ch_2020
   ```

2. Ensure output directories are writable:
   ```bash
   touch /data1/morrisq/yuj13/llm_genomics/mutation_report/test.txt
   rm /data1/morrisq/yuj13/llm_genomics/mutation_report/test.txt
   ```

3. Validate YAML syntax:
   ```python
   import yaml
   with open('configs/task1_mutation_config.yaml') as f:
       config = yaml.safe_load(f)
   print(config)
   ```

---

## Troubleshooting

**Issue**: `FileNotFoundError` when running script
- **Solution**: Check all paths in config are absolute and correct

**Issue**: Out of memory errors
- **Solution**: Reduce `chunk_size` in config

**Issue**: Config not loading
- **Solution**: Validate YAML syntax (indentation matters!)

**Issue**: Wrong output location
- **Solution**: Check `output_*` settings in config

---

## See Also

- Main README: [../README.md](../README.md)
- Task-specific READMEs: [../1_generate_report/README.md](../1_generate_report/README.md)
- Example workflows: [../docs/](../docs/) (if available)

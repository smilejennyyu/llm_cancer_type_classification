# Step 2: LLM Classification

This directory contains the unified LLM classifier for genomic data analysis. The classifier supports three distinct classification tasks using large language models.

## Overview

The `llm_classifier.py` script is a unified implementation that handles:

1. **Task 1: Mutation Status Classification** - Distinguishes between TUMOR-SOMATIC and CHIP (Clonal Hematopoiesis) mutations
2. **Task 2: Oncogenic Classification** - Classifies mutations as Oncogenic or Benign
3. **Task 3: Cancer Type Classification** - Predicts cancer type from comprehensive genomic reports

## Features

- **Multiple LLM Providers**: Azure OpenAI, OpenAI, Claude, and local HuggingFace models
  - **API-based**: Azure OpenAI (gpt-4o, o3-mini), OpenAI, Claude via LiteLLM
  - **Local models**: MedGemma, Qwen, DeepSeek, and other HuggingFace chat models
  - **Automatic quantization**: 4-bit quantization for efficient local inference
- **Robust Error Handling**: Exponential backoff retry logic for API rate limits
- **Reprompting Logic**: Automatically retries with simplified prompts on parsing errors
- **Resume Support**: Continue processing from where it left off
- **Pydantic Validation**: Type-safe response parsing and validation
- **Configurable**: All settings managed via YAML configuration files

## Quick Start

### Task 1: Mutation Status Classification

Classify mutations as TUMOR-SOMATIC vs CHIP:

```bash
# Using Azure OpenAI (default in config)
python llm_classifier.py --config ../configs/task1_mutation_status_llm_config.yaml

# Using local MedGemma model (edit config to set llm_provider: "local")
python llm_classifier.py --config ../configs/task1_mutation_status_llm_config.yaml
```

**Input**: CSV files with columns `SAMPLE_ID`, `Mutation_Status`, `report_str`
**Output**: Predictions with `prediction`, `prob`, and optional `explanation`

### Task 2: Oncogenic Classification

Classify mutations as Oncogenic vs Benign:

```bash
python llm_classifier.py --config ../configs/task2_oncogenic_llm_config.yaml
```

**Input**: CSV files with columns `SAMPLE_ID`, `ONCOGENIC`, `report_str`
**Output**: Predictions with `prediction` and `prob`

### Task 3: Cancer Type Classification

Predict cancer type from comprehensive genomic reports:

```bash
python llm_classifier.py --config ../configs/task3_cancer_type_llm_config.yaml
```

**Input**: CSV files with columns `SAMPLE_ID`, `CANCER_TYPE`, `REPORT`
**Output**: Top 2 predictions with probabilities, sex prediction, and key genomic features

## Configuration

All tasks are configured via YAML files in the [`../configs/`](../configs/) directory:

- [`task1_mutation_status_llm_config.yaml`](../configs/task1_mutation_status_llm_config.yaml)
- [`task2_oncogenic_llm_config.yaml`](../configs/task2_oncogenic_llm_config.yaml)
- [`task3_cancer_type_llm_config.yaml`](../configs/task3_cancer_type_llm_config.yaml)

### Key Configuration Sections

#### LLM Provider Selection

```yaml
llm_provider: "azure"  # Options: "azure", "openai", "claude", "local"

azure:
  api_key_env: "AZURE_OPENAI_API_KEY"
  endpoint_env: "AZURE_OPENAI_ENDPOINT"
  api_version: "2025-01-01-preview"
  model_name: "gpt-4o"
```

#### Input/Output Paths

```yaml
input_dir: "/path/to/input/csv/files"
output_dir: "/path/to/output/results"
prompt_template: "prompts/prompt_file.txt"
```

#### Processing Options

```yaml
include_explanation: true  # Include detailed reasoning (slower)
resume: true              # Resume from existing output
batch_size: null          # Process all files, or set to integer

retry:
  max_retries: 20
  initial_wait_time: 2
  max_wait_time: 60
  max_reprompt_attempts: 3

rate_limit:
  requests_per_minute: 60
  delay_between_calls: 1
```

## Workflow

### 1. Prepare Configuration

Edit the appropriate config file for your task:

```bash
# For mutation status classification
vim ../configs/task1_mutation_status_llm_config.yaml

# Update these fields:
# - llm_provider and model settings
# - input_dir (directory containing CSV files from Step 1)
# - output_dir (where to save predictions)
# - prompt_template (path to prompt file)
```

### 2. Set Environment Variables

For Azure OpenAI:

```bash
export AZURE_OPENAI_API_KEY="your-api-key"
export AZURE_OPENAI_ENDPOINT="https://your-endpoint.openai.azure.com/"
```

For OpenAI:

```bash
export OPENAI_API_KEY="your-api-key"
```

### 3. Run Classification

```bash
python llm_classifier.py --config ../configs/task1_mutation_status_llm_config.yaml
```

The script will:
- Load all CSV files from `input_dir`
- Process each sample through the LLM
- Save results incrementally to `output_dir`
- Resume automatically if interrupted

### 4. Monitor Progress

Progress is logged to the file specified in config:

```bash
# Watch mutation status classification progress
tail -f mutation_status_prediction.log

# Check for errors
grep ERROR mutation_status_prediction.log
```

## Task-Specific Details

### Task 1: Mutation Status Classification

**Purpose**: Distinguish tumor-specific somatic mutations from clonal hematopoiesis (CHIP) mutations.

**Input Requirements**:
- `SAMPLE_ID`: Unique sample identifier
- `Mutation_Status`: Ground truth label (TUMOR-SOMATIC or CHIP)
- `report_str`: Mutation report text

**Output Columns**:
- `prediction`: TUMOR-SOMATIC, CHIP, or UNKNOWN
- `prob`: Confidence probability (0-1)
- `explanation`: Detailed reasoning (if `include_explanation: true`)
- `ground_truth`: Original mutation status
- Additional columns from input preserved

**Valid Predictions**: `TUMOR-SOMATIC`, `CHIP`, `UNKNOWN`

**Example Config Modifications**:

```yaml
# Use faster inference without explanation
prompt_template: "prompts/prompt_mutation_prediction_CH_no_explain.txt"
include_explanation: false

# Use o3-mini for faster/cheaper inference
azure:
  model_name: "o3-mini"
```

### Task 2: Oncogenic Classification

**Purpose**: Classify mutations as oncogenic (cancer-driving) or benign.

**Input Requirements**:
- `SAMPLE_ID`: Unique sample identifier
- `ONCOGENIC`: Ground truth label
- `report_str`: Mutation report text

**Output Columns**:
- `prediction`: Oncogenic or Benign
- `prob`: Confidence probability (0-1)
- `ground_truth`: Original oncogenic status
- Additional columns from input preserved

**Valid Predictions**: `Oncogenic`, `Benign`

**Example Config Modifications**:

```yaml
# Faster inference without explanation (default)
include_explanation: false

# Increase rate limiting for faster processing
rate_limit:
  requests_per_minute: 120
  delay_between_calls: 0.5
```

### Task 3: Cancer Type Classification

**Purpose**: Predict cancer type from comprehensive genomic reports including mutations, CNAs, fusions, and arm-level changes.

**Input Requirements**:
- `SAMPLE_ID`: Unique sample identifier
- `CANCER_TYPE`: Ground truth cancer type
- `REPORT`: Comprehensive genomic report text

**Output Columns**:
- `prediction1`: Top predicted cancer type
- `prob1`: Confidence for top prediction
- `prediction2`: Second predicted cancer type
- `prob2`: Confidence for second prediction
- `sex`: Predicted patient sex
- `sex_used`: Whether sex influenced prediction
- `key_genes`: Top 3 influential genes
- `key_arm_changes`: Top 3 chromosomal arm changes
- `key_scnas`: Top 3 copy number alterations
- `key_fusions`: Top 3 gene fusions
- `important_data_part`: Most influential data type
- `explanation`: Detailed reasoning (if enabled)
- `ground_truth`: Original cancer type

**Valid Cancer Types**: See [config file](../configs/task3_cancer_type_llm_config.yaml) for full list of 34 cancer types.

**Example Config Modifications**:

```yaml
# Use extended prompt with detailed explanations
prompt_template: "prompts/prompt_genie_extended_with_explain.txt"
include_explanation: true

# Or use faster prompt without explanations
prompt_template: "prompts/prompt_genie_extended_no_explain.txt"
include_explanation: false
```

## Advanced Usage

### Changing LLM Provider

#### Use OpenAI (non-Azure)

```yaml
llm_provider: "openai"

openai:
  api_key_env: "OPENAI_API_KEY"
  model_name: "gpt-4"
```

#### Use Claude via LiteLLM Proxy

```yaml
llm_provider: "claude"

claude:
  base_url: "http://your-litellm-server:4000/v1"
  api_key: "litellm"
  model_name: "bedrock-sonnet-37"
```

#### Use Local HuggingFace Models

The classifier supports local models like MedGemma, Qwen, DeepSeek, and other HuggingFace models.

**Requirements:**
```bash
pip install transformers torch
# For quantization (recommended):
pip install bitsandbytes accelerate
```

**Configuration:**
```yaml
llm_provider: "local"

local:
  # Use HuggingFace model ID
  model_name: "google/medgemma-27b-it"

  # Or use local path to downloaded model
  # model_path: "/data/models/medgemma-27b-it"

  device: "cuda"  # or "cpu"
  quantization: true  # 4-bit quantization (CUDA only, reduces memory)

  # Generation parameters
  max_new_tokens: 256  # Max tokens to generate
  temperature: 0.7     # Sampling temperature
  top_p: 0.9          # Nucleus sampling
```

**Supported Models:**

| Model | HuggingFace ID | Use Case | Memory (4-bit) |
|-------|---------------|----------|----------------|
| MedGemma 27B | `google/medgemma-27b-it` | Medical domain tasks | ~14 GB |
| Qwen3 30B | `Qwen/Qwen3-30B-A3B-Instruct-2507` | General + medical | ~16 GB |
| DeepSeek R1 32B | `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B` | Reasoning tasks | ~17 GB |

**Example for Task 1 (Mutation Status):**
```yaml
llm_provider: "local"

local:
  model_name: "google/medgemma-27b-it"
  device: "cuda"
  quantization: true
  max_new_tokens: 256
  temperature: 0.7
  top_p: 0.9
```

**Example for Task 2 (Oncogenic):**
```yaml
llm_provider: "local"

local:
  model_name: "Qwen/Qwen3-30B-A3B-Instruct-2507"
  device: "cuda"
  quantization: true
  max_new_tokens: 256
  temperature: 0.7
  top_p: 0.9
```

**Example for Task 3 (Cancer Type):**
```yaml
llm_provider: "local"

local:
  model_name: "google/medgemma-27b-it"
  device: "cuda"
  quantization: true
  max_new_tokens: 512  # Longer for detailed features
  temperature: 0.7
  top_p: 0.9
```

**Performance Tips for Local Models:**

1. **Use quantization** - Reduces memory from ~60GB to ~15GB for 27B models
2. **Adjust batch size** - Process one sample at a time for large models
3. **Monitor GPU memory** - Use `nvidia-smi` to check CUDA memory usage
4. **CPU fallback** - Set `device: "cpu"` if GPU memory is insufficient (much slower)

**Troubleshooting Local Models:**

- **CUDA out of memory**: Enable quantization or use CPU
- **Slow inference**: Ensure CUDA is available, check GPU utilization
- **Model download issues**: Pre-download models or use local paths

### Adjusting Retry Behavior

For unstable connections:

```yaml
retry:
  max_retries: 40          # Increase max retries
  initial_wait_time: 5     # Start with longer wait
  max_wait_time: 120       # Allow longer backoff
  max_reprompt_attempts: 5 # More reprompt attempts
```

For faster inference (stable connection):

```yaml
retry:
  max_retries: 10
  initial_wait_time: 1
  max_wait_time: 30
  max_reprompt_attempts: 2
```

### Processing Specific Files

To process only specific CSV files, move them to a separate directory:

```bash
# Create subset directory
mkdir -p /tmp/subset_data

# Copy specific files
cp /path/to/input/batch_1.csv /tmp/subset_data/
cp /path/to/input/batch_5.csv /tmp/subset_data/

# Update config
# input_dir: "/tmp/subset_data"

# Run classification
python llm_classifier.py --config ../configs/task1_mutation_status_llm_config.yaml
```

### Resume After Interruption

The classifier automatically resumes from where it stopped:

```bash
# Start processing
python llm_classifier.py --config ../configs/task1_mutation_status_llm_config.yaml

# If interrupted (Ctrl+C or error), simply re-run the same command
python llm_classifier.py --config ../configs/task1_mutation_status_llm_config.yaml

# It will skip already processed samples
```

To disable resume and restart from scratch:

```yaml
resume: false
```

## Understanding Output Files

### Output File Structure

Each input CSV file produces a corresponding output file in `output_dir`:

```
input_dir/
├── batch_1.csv
├── batch_2.csv
└── batch_3.csv

output_dir/
├── batch_1.csv  (predictions for batch_1)
├── batch_2.csv  (predictions for batch_2)
└── batch_3.csv  (predictions for batch_3)
```

### Output CSV Format

**Task 1 & 2 Output** (mutation_status and oncogenic):
```csv
SAMPLE_ID,Hugo_Symbol,...,prediction,prob,ground_truth,explanation
SA12345,TP53,...,TUMOR-SOMATIC,0.95,TUMOR-SOMATIC,"High VAF and..."
```

**Task 3 Output** (cancer_type):
```csv
SAMPLE_ID,prediction1,prob1,prediction2,prob2,sex,sex_used,key_genes,...,ground_truth
SA12345,Breast Cancer,0.85,Ovarian Cancer,0.12,Female,Yes,TP53;PIK3CA;GATA3,...,Breast Cancer
```

## Troubleshooting

### Issue: Rate Limit Errors

**Symptom**: Logs show "429 rate limit hit" repeatedly

**Solutions**:
1. Increase wait time:
   ```yaml
   retry:
     initial_wait_time: 5
     max_wait_time: 120
   ```

2. Reduce request rate:
   ```yaml
   rate_limit:
     requests_per_minute: 30
     delay_between_calls: 2
   ```

### Issue: JSON Parsing Errors

**Symptom**: Many "JSON parsing error" messages in logs

**Solutions**:
1. Increase reprompt attempts:
   ```yaml
   retry:
     max_reprompt_attempts: 5
   ```

2. Use a more capable model:
   ```yaml
   azure:
     model_name: "gpt-4o"  # instead of o3-mini
   ```

### Issue: Invalid Predictions

**Symptom**: Predictions not in valid set (e.g., "Malignant" instead of "Oncogenic")

**Solutions**:
1. Check prompt template for clear instructions
2. Add normalization rules in config:
   ```yaml
   normalize_predictions:
     "Malignant": "Oncogenic"
     "Benign Variant": "Benign"
   ```

### Issue: Missing Environment Variables

**Symptom**: `KeyError: 'AZURE_OPENAI_API_KEY'`

**Solution**:
```bash
export AZURE_OPENAI_API_KEY="your-key-here"
export AZURE_OPENAI_ENDPOINT="https://your-endpoint.openai.azure.com/"
```

### Issue: Slow Processing

**Solutions**:
1. Disable explanations:
   ```yaml
   include_explanation: false
   prompt_template: "prompts/prompt_..._no_explain.txt"
   ```

2. Use faster model:
   ```yaml
   azure:
     model_name: "o3-mini"  # Faster than gpt-4o
   ```

3. Increase parallelism (if running multiple instances):
   - Split input files across multiple directories
   - Run multiple instances with different configs

## Performance Tips

### Speed Optimization

1. **Disable explanations** when not needed for analysis
2. **Use o3-mini** for routine classification (cheaper and faster)
3. **Increase rate limits** if your API quota allows
4. **Process in batches** using multiple instances

### Cost Optimization

1. **Use o3-mini** instead of gpt-4o for similar accuracy at lower cost
2. **Disable explanations** to reduce token usage
3. **Resume functionality** prevents re-processing on failures
4. **Validate prompts** on small subset before full run

### Accuracy Optimization

1. **Include explanations** for better reasoning
2. **Use gpt-4o or gpt-5** for highest accuracy
3. **Increase reprompt attempts** for better error recovery
4. **Validate results** and adjust prompts if needed

## File Organization

```
2_llm_classification/
├── llm_classifier.py          # Unified classifier script
├── README.md                  # This file
└── (output directories created at runtime)

../configs/
├── task1_mutation_status_llm_config.yaml
├── task2_oncogenic_llm_config.yaml
└── task3_cancer_type_llm_config.yaml

../prompts/
├── prompt_mutation_prediction_CH.txt
├── prompt_mutation_prediction_CH_no_explain.txt
├── prompt_mutation_prediction_oncogenic_absolute.txt
├── prompt_mutation_prediction_oncogenic_absolute_no_explain.txt
├── prompt_genie_extended_with_explain.txt
└── prompt_genie_extended_no_explain.txt
```

## Example Workflows

### Complete Workflow for Task 1

```bash
# 1. Set environment variables
export AZURE_OPENAI_API_KEY="your-api-key"
export AZURE_OPENAI_ENDPOINT="https://your-endpoint.openai.azure.com/"

# 2. Review configuration
cat ../configs/task1_mutation_status_llm_config.yaml

# 3. Run classification
python llm_classifier.py --config ../configs/task1_mutation_status_llm_config.yaml

# 4. Monitor progress
tail -f mutation_status_prediction.log

# 5. Check for errors
grep ERROR mutation_status_prediction.log

# 6. Verify output
ls -lh /path/to/output/dir/
head -5 /path/to/output/dir/batch_1.csv
```

### Testing on Subset

```bash
# 1. Create test subset
mkdir -p /tmp/test_input
cp /path/to/full/input/batch_1.csv /tmp/test_input/

# 2. Create test config
cp ../configs/task1_mutation_status_llm_config.yaml /tmp/test_config.yaml

# 3. Edit test config
vim /tmp/test_config.yaml
# Change:
#   input_dir: "/tmp/test_input"
#   output_dir: "/tmp/test_output"

# 4. Run test
python llm_classifier.py --config /tmp/test_config.yaml

# 5. Review results
cat /tmp/test_output/batch_1.csv
```

### Using Local Models (Complete Example)

```bash
# 1. Install dependencies
pip install transformers torch bitsandbytes accelerate

# 2. Create local model config
cp ../configs/task1_mutation_status_llm_config.yaml /tmp/local_config.yaml

# 3. Edit config to use local model
vim /tmp/local_config.yaml
# Change:
#   llm_provider: "local"
# Uncomment and configure the local section:
#   local:
#     model_name: "google/medgemma-27b-it"
#     device: "cuda"
#     quantization: true
#     max_new_tokens: 256
#     temperature: 0.7
#     top_p: 0.9

# 4. Run classification with local model
python llm_classifier.py --config /tmp/local_config.yaml

# 5. Monitor GPU usage (if using CUDA)
watch -n 1 nvidia-smi
```

## Related Documentation

- Main README: [../README.md](../README.md)
- Configuration README: [../configs/README.md](../configs/README.md)
- Step 1 Data Generation: [../1_generate_report/README.md](../1_generate_report/README.md)

## Citation

If you use this classifier in your research, please cite the original zeroshot_cancertype work and acknowledge the LLM model used.

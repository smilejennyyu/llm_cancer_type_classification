# Step 3: Ensemble Models

This directory contains ensemble models that combine predictions from task-specific models with LLM predictions using machine learning meta-models.

## Overview

The ensemble approach combines:
- **Task-specific models**: Domain-specific models trained for specific classification tasks (e.g., GDD for cancer type, AlphaMissense for mutation impact)
- **LLM models**: Large language model predictions (e.g., gpt-4o, gpt-5, o3-mini, MedGemma)
- **Meta-models**: Machine learning models (Logistic Regression, Random Forest, XGBoost) that learn to combine the predictions

This typically improves performance over using either model alone.

## Tasks

### Task 1: Mutation Status Ensemble ([task12_mutation_ensemble.py](task12_mutation_ensemble.py))

Combines predictions for TUMOR-SOMATIC vs CHIP classification.

**Example models to combine:**
- AlphaMissense + gpt-4o
- OncoKB + MedGemma
- Any two mutation status classifiers

**Usage:**
```bash
python task12_mutation_ensemble.py --config ../configs/task1_mutation_ensemble_config.yaml
```

### Task 2: Oncogenic Classification Ensemble ([task12_mutation_ensemble.py](task12_mutation_ensemble.py))

Combines predictions for Oncogenic vs Benign classification.

**Example models to combine:**
- AlphaMissense + gpt-4o
- ClinVar + MedGemma
- Any two oncogenic classifiers

**Usage:**
```bash
python task12_mutation_ensemble.py --config ../configs/task2_oncogenic_ensemble_config.yaml
```

### Task 3: Cancer Type Ensemble ([task3_cancer_type_ensemble.py](task3_cancer_type_ensemble.py))

Combines predictions for cancer type classification.

**Example models to combine:**
- GDD (Genomic Diagnosis of Disease) + gpt-5
- Any task-specific cancer type classifier + LLM

**Usage:**
```bash
python task3_cancer_type_ensemble.py --config ../configs/task3_cancer_type_ensemble_config.yaml
```

## Features

- **Multiple ensemble methods**: Logistic Regression, Random Forest, XGBoost
- **Patient-level cross-validation**: Splits by PATIENT_ID to avoid data leakage
- **K-fold cross-validation**: Default 5-fold, configurable
- **Comprehensive metrics**: Accuracy, F1 score, Precision, Recall
- **Automatic visualization**: Bar plots with error bars
- **Timestamped outputs**: All results saved with timestamps
- **Configuration-driven**: All settings in YAML files

## Quick Start

### Task 3: Cancer Type Ensemble (Complete Example)

```bash
# 1. Prepare your data
# Ensure you have:
# - Task-specific model predictions (e.g., GDD output from Step 1)
# - LLM predictions (e.g., gpt-5 output from Step 2)

# 2. Update configuration file
vim ../configs/task3_cancer_type_ensemble_config.yaml
# Set:
#   model1.predictions_dir: "/path/to/GDD/predictions"
#   model2.predictions_dir: "/path/to/gpt-5/predictions"

# 3. Run ensemble model
python task3_cancer_type_ensemble.py --config ../configs/task3_cancer_type_ensemble_config.yaml

# 4. View results
# Results saved to: /data1/morrisq/yuj13/llm_genomics/genie_meta_model_result/{timestamp}/
```

### Tasks 1 & 2: Mutation Ensemble (Complete Example)

```bash
# 1. Prepare merged predictions file
# Create a CSV with columns:
#   - SAMPLE_ID, PATIENT_ID, ground_truth
#   - Model1 prediction and probability columns
#   - Model2 prediction and probability columns

# 2. Update configuration
vim ../configs/task1_mutation_ensemble_config.yaml
# Set:
#   input.data_path: "/path/to/merged_predictions.csv"
#   model1.name: "AlphaMissense"
#   model1.prediction_column: "AlphaMissense_prediction"
#   model1.probability_column: "AlphaMissense_prob"
#   model2.name: "gpt-4o"
#   model2.prediction_column: "prediction"
#   model2.probability_column: "prob"

# 3. Run ensemble model
python task12_mutation_ensemble.py --config ../configs/task1_mutation_ensemble_config.yaml

# 4. View results
# Results saved to: /data1/morrisq/yuj13/llm_genomics/mutation_meta_model_result/{model1}_{model2}_{timestamp}/
```

## Configuration

All ensemble models are configured via YAML files in the [`../configs/`](../configs/) directory:

- [`task1_mutation_ensemble_config.yaml`](../configs/task1_mutation_ensemble_config.yaml) - Mutation status ensemble
- [`task2_oncogenic_ensemble_config.yaml`](../configs/task2_oncogenic_ensemble_config.yaml) - Oncogenic classification ensemble
- [`task3_cancer_type_ensemble_config.yaml`](../configs/task3_cancer_type_ensemble_config.yaml) - Cancer type ensemble

### Key Configuration Sections

#### Model Configuration

```yaml
# Task 3 example (two separate prediction directories)
model1:
  name: "GDD"
  predictions_dir: "/path/to/GDD/predictions"

model2:
  name: "gpt-5"
  predictions_dir: "/path/to/gpt-5/predictions"
  normalize_predictions: true

# Tasks 1 & 2 example (single merged file)
input:
  data_path: "/path/to/merged_predictions.csv"

model1:
  name: "AlphaMissense"
  prediction_column: "AlphaMissense_prediction"
  probability_column: "AlphaMissense_prob"

model2:
  name: "gpt-4o"
  prediction_column: "prediction"
  probability_column: "prob"
```

#### Cross-Validation Settings

```yaml
# Number of folds for k-fold cross-validation
k_fold: 5

# Random seed for reproducibility
random_seed: 42
```

#### Output Configuration

```yaml
# Task 3
output_dir: "/data1/morrisq/yuj13/llm_genomics/genie_meta_model_result/{timestamp}"

# Tasks 1 & 2
output_dir: "/data1/morrisq/yuj13/llm_genomics/mutation_meta_model_result"
```

## Output Files

After running an ensemble model, the following files are generated in the output directory:

### For K-Fold Cross-Validation

| File | Description |
|------|-------------|
| `5_fold_cv_detailed_results.csv` | Detailed metrics for each fold |
| `5_fold_cv_summary.csv` | Summary statistics (mean ± std) for each method |
| `5_fold_cv_accuracy_plot.png` | Bar plot comparing methods |
| `overall_summary.csv` | Best performing method summary |

### Detailed Results CSV Format

```csv
fold,GDD_accuracy,GDD_f1,GDD_precision,GDD_recall,gpt-5_accuracy,...,LogReg_accuracy,...,XGBoost_accuracy,...,RandomForest_accuracy,...
1,0.85,0.84,0.86,0.83,0.82,0.81,0.83,0.82,0.87,0.86,0.88,0.87,0.86,0.85,0.87,0.86
2,0.84,0.83,0.85,0.82,0.81,0.80,0.82,0.81,0.88,0.87,0.89,0.88,0.85,0.84,0.86,0.85
...
```

### Summary CSV Format

```csv
Method,Mean_Accuracy,Std_Accuracy,Mean_F1,Std_F1,Mean_Precision,Std_Precision,Mean_Recall,Std_Recall
GDD,0.847,0.015,0.835,0.018,0.855,0.020,0.825,0.016
gpt-5,0.815,0.012,0.808,0.014,0.825,0.015,0.810,0.013
Logistic Regression,0.875,0.010,0.868,0.012,0.880,0.011,0.865,0.013
XGBoost,0.860,0.012,0.852,0.014,0.865,0.013,0.850,0.015
Random Forest,0.855,0.013,0.848,0.015,0.860,0.014,0.845,0.016
```

## Workflow

### Typical Workflow for Task 3

```
1. Generate task-specific model predictions (e.g., GDD from Step 1)
   ↓
2. Generate LLM predictions (e.g., gpt-5 from Step 2)
   ↓
3. Configure ensemble model (edit config YAML)
   ↓
4. Run ensemble model
   ↓
5. Analyze results (CSV files and plots)
```

### Typical Workflow for Tasks 1 & 2

```
1. Generate predictions from Model 1 (e.g., AlphaMissense)
   ↓
2. Generate predictions from Model 2 (e.g., gpt-4o from Step 2)
   ↓
3. Merge predictions into single CSV file
   ↓
4. Configure ensemble model (edit config YAML)
   ↓
5. Run ensemble model
   ↓
6. Analyze results (CSV files and plots)
```

## Ensemble Methods

### Logistic Regression

- **When to use**: Simple, interpretable baseline
- **Pros**: Fast training, interpretable coefficients, works well with few features
- **Cons**: Assumes linear decision boundaries

### Random Forest

- **When to use**: Want robust performance without much tuning
- **Pros**: Handles non-linear relationships, robust to overfitting, provides feature importances
- **Cons**: Slower than logistic regression, less interpretable

### XGBoost

- **When to use**: Want best possible performance
- **Pros**: Often achieves best accuracy, handles complex patterns, built-in regularization
- **Cons**: Requires more tuning, can overfit on small datasets

## Advanced Usage

### Custom Number of Folds

```bash
# Run with 10-fold cross-validation instead of 5
python task3_cancer_type_ensemble.py --config ../configs/task3_cancer_type_ensemble_config.yaml --fold 10
```

### Multiple Model Comparisons

Run the same script with different model combinations:

```bash
# Compare different LLMs with GDD
python task3_cancer_type_ensemble.py --config ../configs/task3_gdd_gpt5_config.yaml
python task3_cancer_type_ensemble.py --config ../configs/task3_gdd_gpt4o_config.yaml
python task3_cancer_type_ensemble.py --config ../configs/task3_gdd_medgemma_config.yaml

# Then compare results across all runs
```

### Cancer Type-Specific Analysis

For Task 3, you can analyze performance by specific cancer types by examining the detailed results and filtering by `CANCER_TYPE` column in your data.

## Data Requirements

### Task 3 Requirements

**Model 1 (Task-specific) predictions must have:**
- `SAMPLE_ID` column
- `Pred1`, `Pred2` columns (top 2 predictions)
- `Conf1`, `Conf2` columns (confidence scores)
- `Cancer_Type` or similar ground truth column

**Model 2 (LLM) predictions must have:**
- `SAMPLE_ID` column
- `prediction1`, `prediction2` columns
- `prob1`, `prob2` columns
- `ground_truth` column

**Optional:**
- `PATIENT_ID` for patient-level cross-validation
- `SEQ_ASSAY_ID` and clinical data for center-based analysis

### Tasks 1 & 2 Requirements

**Merged predictions file must have:**
- `SAMPLE_ID` column
- `PATIENT_ID` column
- `ground_truth` column
- Model 1 prediction and probability columns (column names configurable)
- Model 2 prediction and probability columns (column names configurable)
- Optional: `CANCER_TYPE` for cancer type-specific analysis

## Troubleshooting

### Issue: FileNotFoundError for predictions

**Symptom**: `No CSV files found in /path/to/predictions`

**Solutions**:
1. Check that the predictions directory exists and contains CSV files
2. Verify the path in your config file
3. Ensure predictions were generated successfully in Steps 1 and 2

### Issue: Missing required columns

**Symptom**: `KeyError: 'PATIENT_ID'` or similar

**Solutions**:
1. Check that your input data has all required columns
2. For Task 3, ensure clinical data is loaded (if using patient-level CV)
3. For Tasks 1 & 2, ensure the merged predictions file has PATIENT_ID column
4. Update `required_columns` in config if needed

### Issue: Low ensemble performance

**Symptom**: Ensemble models don't improve over individual models

**Solutions**:
1. Check that the two models are complementary (different strengths/weaknesses)
2. Ensure sufficient data for training meta-models (>1000 samples recommended)
3. Try different k-fold values (e.g., 3 or 10 instead of 5)
4. Check for class imbalance issues in ground truth

### Issue: Memory errors

**Symptom**: `MemoryError` or `Out of memory`

**Solutions**:
1. Use smaller k (e.g., k=3 instead of k=5)
2. Process smaller subsets of data
3. Use Random Forest instead of XGBoost (lower memory footprint)

## Performance Tips

### Improving Ensemble Performance

1. **Use diverse models**: Combine models with different architectures (e.g., rule-based + LLM)
2. **Ensure data quality**: Clean predictions, handle missing values
3. **Sufficient training data**: >1000 samples recommended for reliable ensemble training
4. **Feature engineering**: Consider adding additional features beyond predictions
5. **Hyperparameter tuning**: Use GridSearchCV for optimal ensemble model parameters

### Computational Efficiency

1. **Parallel processing**: Run multiple configs in parallel on different machines
2. **Caching**: Reuse loaded data for multiple ensemble configurations
3. **Subset testing**: Test on small subset before full run

## Example Results Interpretation

```
===== Cross-Validation Metrics Summary =====
GDD:
  Accuracy: 0.8470 ± 0.0150
  F1 Score: 0.8350 ± 0.0180

gpt-5:
  Accuracy: 0.8150 ± 0.0120
  F1 Score: 0.8080 ± 0.0140

Logistic Regression:
  Accuracy: 0.8750 ± 0.0100
  F1 Score: 0.8680 ± 0.0120

XGBoost:
  Accuracy: 0.8600 ± 0.0120
  F1 Score: 0.8520 ± 0.0140

Random Forest:
  Accuracy: 0.8550 ± 0.0130
  F1 Score: 0.8480 ± 0.0150
```

**Interpretation:**
- **Best individual model**: GDD (84.7%)
- **Best ensemble**: Logistic Regression (87.5%)
- **Improvement**: +2.8% over best individual model
- **Low variance**: All methods have std < 2%, indicating stable performance
- **Conclusion**: Ensemble provides consistent improvement; Logistic Regression is best

## Related Documentation

- Main README: [../README.md](../README.md)
- Configuration README: [../configs/README.md](../configs/README.md)
- Step 1 Data Generation: [../1_generate_report/README.md](../1_generate_report/README.md)
- Step 2 LLM Classification: [../2_llm_classification/README.md](../2_llm_classification/README.md)

## Citation

If you use these ensemble models in your research, please cite the appropriate base models (GDD, AlphaMissense, etc.) and the LLM models used.

# VAF Comparison Analysis Summary

## Notebook: vaf_comparison_analysis.ipynb

This notebook now includes comprehensive statistical analysis comparing model performance WITH VAF vs WITHOUT VAF features.

## Statistical Tests Implemented

### 1. **McNemar's Test** (Cells 15-18)
- **Purpose**: Tests whether two paired classifiers have significantly different error rates
- **Implementation**: Cell 15 performs McNemar's test for all 6 models
- **Outputs**:
  - CSV: `vaf_mcnemar_test_results.csv`
  - Contingency table statistics (a, b, c, d counts)
  - Chi-square statistic and p-values

### 2. **Permutation Test** (Cells 19-24)
- **Purpose**: Non-parametric test to determine if observed performance difference could occur by chance
- **Implementation**:
  - Cell 20: Permutation test function (10,000 permutations)
  - Cell 21: Runs tests for all models on accuracy and F1 metrics
- **Outputs**:
  - CSV: `vaf_permutation_test_accuracy.csv`
  - CSV: `vaf_permutation_test_f1.csv`
  - Null distribution arrays for each model

## Visualizations Generated

### McNemar Test Visualizations (4 plots)

1. **mcnemar_pvalue_plot.pdf**
   - Horizontal bar chart showing p-values on log scale
   - Color-coded by significance level
   - Reference lines at p=0.001, 0.01, 0.05

2. **mcnemar_contingency_heatmaps.pdf**
   - 2x2 contingency tables for each model
   - Shows agreement/disagreement patterns
   - Displays counts and percentages

3. **mcnemar_disagreement_summary.pdf**
   - Bar chart comparing unique correct predictions
   - Green: With VAF correct, No VAF wrong
   - Red: No VAF correct, With VAF wrong
   - Significance markers above each pair

4. **mcnemar_chisq_vs_accuracy.pdf** (if added)
   - Scatter plot: chi-square vs accuracy difference
   - Shows effect size vs statistical significance

### Permutation Test Visualizations (6 plots)

5. **permutation_pvalue_accuracy.pdf**
   - P-value bar chart for accuracy metric
   - Includes observed differences in labels
   - Log scale with significance thresholds

6. **permutation_pvalue_f1.pdf**
   - P-value bar chart for F1 metric
   - Same format as accuracy plot

7. **permutation_null_distributions_accuracy.pdf**
   - 6 subplots showing null distributions
   - Red line: observed difference
   - Blue histogram: null hypothesis distribution
   - Includes Z-scores and normal fit

8. **permutation_null_distributions_f1.pdf**
   - Same as above for F1 metric

9. **permutation_forest_plot_accuracy.pdf** (if added)
   - Forest plot with 95% confidence intervals
   - Points outside CI indicate significance

10. **permutation_forest_plot_f1.pdf** (if added)
    - Same as above for F1 metric

## Key Interpretation

### Significance Levels
- `***` : p < 0.001 (highly significant)
- `**`  : p < 0.01 (very significant)
- `*`   : p < 0.05 (significant)
- `ns`  : p ≥ 0.05 (not significant)

### McNemar's Test Results
- Tests if error rates differ between With/Without VAF
- Low p-value → significant difference in predictions
- b > c → With VAF makes more unique correct predictions
- c > b → No VAF makes more unique correct predictions

### Permutation Test Results
- Tests if performance difference could occur by random chance
- Low p-value → observed difference is unlikely under null hypothesis
- Null distribution shows what differences would be expected by chance
- Observed value far from null mean → strong evidence for real effect

## Files Generated

### CSV Files
- `vaf_comparison_delta.csv` - Performance differences for all metrics
- `vaf_comparison_all_results.csv` - Complete metrics for both conditions
- `vaf_mcnemar_test_results.csv` - McNemar test statistics
- `vaf_permutation_test_accuracy.csv` - Permutation test results (accuracy)
- `vaf_permutation_test_f1.csv` - Permutation test results (F1)

### PDF Plots
All plots saved in: `vaf_comparison_plots/`

## Running the Notebook

1. Ensure all data is loaded (cells 1-7)
2. Run McNemar test section (cells 15-18)
3. Run Permutation test section (cells 19-24)
4. Check final summary (cell 25)

Note: Permutation tests take several minutes to complete (10,000 permutations per model per metric).

## Models Analyzed
- claude-sonnet-37
- deepseek
- gpt-4o
- gpt-5
- medgemma
- qwen3

## Metrics Evaluated
- Accuracy (%)
- F1 Score (macro average)
- Precision (macro average)
- Recall (macro average)

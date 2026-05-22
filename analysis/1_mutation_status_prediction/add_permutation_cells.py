#!/usr/bin/env python3
"""
Script to add remaining permutation test cells to the notebook
"""

import json

# The cells to add
cells_to_add = [
    # Cell: Run permutation tests
    {
        "cell_type": "code",
        "source": """# Run Permutation Tests for All Models
print("="*80)
print("RUNNING PERMUTATION TESTS")
print("="*80)
print("This may take several minutes...\\n")

permutation_results = {
    'accuracy': [],
    'f1': []
}

# Store null distributions for plotting
null_distributions = {
    'accuracy': {},
    'f1': {}
}

for metric in ['accuracy', 'f1']:
    print(f"\\n{'='*80}")
    print(f"METRIC: {metric.upper()}")
    print(f"{'='*80}\\n")

    for model in MODEL_NAMES:
        if model not in with_vaf_data or model not in no_vaf_data:
            continue

        print(f"  Testing {model}...")
        p_value, observed_diff, null_diffs = permutation_test(
            with_vaf_data, no_vaf_data, model, metric=metric, n_permutations=10000
        )

        if p_value is not None:
            result = {
                'model': model,
                'metric': metric,
                'observed_diff': observed_diff,
                'p_value': p_value,
                'null_mean': np.mean(null_diffs),
                'null_std': np.std(null_diffs)
            }
            permutation_results[metric].append(result)
            null_distributions[metric][model] = null_diffs

            # Print result
            if p_value < 0.001:
                sig = '***'
            elif p_value < 0.01:
                sig = '**'
            elif p_value < 0.05:
                sig = '*'
            else:
                sig = 'ns'

            if metric == 'accuracy':
                print(f"    Observed difference: {observed_diff:+.3f}%")
                print(f"    Null mean: {np.mean(null_diffs):.3f}%")
                print(f"    Null std: {np.std(null_diffs):.3f}%")
            else:
                print(f"    Observed difference: {observed_diff:+.6f}")
                print(f"    Null mean: {np.mean(null_diffs):.6f}")
                print(f"    Null std: {np.std(null_diffs):.6f}")

            print(f"    P-value: {p_value:.4f} {sig}")
            print()

# Create DataFrames for each metric
perm_acc_df = pd.DataFrame(permutation_results['accuracy']).sort_values('p_value')
perm_f1_df = pd.DataFrame(permutation_results['f1']).sort_values('p_value')

print("\\n" + "="*80)
print("PERMUTATION TEST RESULTS SUMMARY")
print("="*80)

print("\\nACCURACY:")
print(perm_acc_df[['model', 'observed_diff', 'p_value']].to_string(index=False))

print("\\n\\nF1 SCORE:")
print(perm_f1_df[['model', 'observed_diff', 'p_value']].to_string(index=False))

# Save results
perm_acc_df.to_csv(os.path.join(OUTPUT_DIR, 'vaf_permutation_test_accuracy.csv'), index=False)
perm_f1_df.to_csv(os.path.join(OUTPUT_DIR, 'vaf_permutation_test_f1.csv'), index=False)

print(f"\\n\\nSaved permutation test results to:")
print(f"  {os.path.join(OUTPUT_DIR, 'vaf_permutation_test_accuracy.csv')}")
print(f"  {os.path.join(OUTPUT_DIR, 'vaf_permutation_test_f1.csv')}")""",
        "metadata": {},
        "outputs": []
    }
]

print("Generated cells for permutation tests")
print(f"Number of cells: {len(cells_to_add)}")

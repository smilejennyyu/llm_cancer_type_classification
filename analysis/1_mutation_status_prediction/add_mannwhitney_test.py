#!/usr/bin/env python3
"""
Script to add Mann-Whitney U test cells to the vaf_comparison_analysis notebook
"""

import json

# Read the notebook
notebook_path = 'vaf_comparison_analysis.ipynb'
with open(notebook_path, 'r') as f:
    notebook = json.load(f)

# New cell 1: Mann-Whitney U Test Function
cell1_source = """# Mann-Whitney U Test for VAF Comparison
from scipy.stats import mannwhitneyu
import numpy as np

def mann_whitney_test_vaf(with_vaf_data, no_vaf_data, model, metric='accuracy'):
    \"\"\"
    Perform Mann-Whitney U test to compare with VAF vs no VAF for a single model.

    This is a non-parametric test that doesn't assume normal distribution.
    It tests whether the distributions are the same.

    Args:
        with_vaf_data: dict of dataframes with VAF predictions
        no_vaf_data: dict of dataframes without VAF predictions
        model: model name
        metric: 'accuracy' or 'f1'

    Returns:
        p_value: p-value from Mann-Whitney U test
        statistic: U statistic
        with_scores: individual scores with VAF
        no_scores: individual scores without VAF
    \"\"\"
    df_with = with_vaf_data[model].copy()
    df_with = df_with[df_with['prediction'] != 'UNKNOWN']

    df_no = no_vaf_data[model].copy()
    df_no = df_no[df_no['prediction'] != 'UNKNOWN']

    # Align samples - crucial for paired comparison
    common_ids = set(df_with.index) & set(df_no.index)
    df_with = df_with.loc[list(common_ids)]
    df_no = df_no.loc[list(common_ids)]

    # Calculate individual sample scores
    # For each sample, we'll calculate if it was correct (1) or incorrect (0)
    if metric == 'accuracy':
        with_scores = (df_with['ground_truth'] == df_with['prediction']).astype(int)
        no_scores = (df_no['ground_truth'] == df_no['prediction']).astype(int)
    else:  # f1
        # For F1, we need to calculate per-sample contribution
        # We'll use bootstrapping approach: sample with replacement and calculate F1
        from sklearn.metrics import f1_score

        # Get all labels
        all_labels_with = sorted(set(df_with['ground_truth'].unique()) | set(df_with['prediction'].unique()))
        all_labels_no = sorted(set(df_no['ground_truth'].unique()) | set(df_no['prediction'].unique()))

        # Bootstrap to get distribution of F1 scores
        n_bootstrap = 1000
        with_scores = []
        no_scores = []

        np.random.seed(42)
        for _ in range(n_bootstrap):
            # Sample with replacement
            sample_idx = np.random.choice(len(df_with), size=len(df_with), replace=True)

            sample_with = df_with.iloc[sample_idx]
            sample_no = df_no.iloc[sample_idx]

            f1_with = f1_score(sample_with['ground_truth'], sample_with['prediction'],
                              labels=all_labels_with, average='macro', zero_division=0)
            f1_no = f1_score(sample_no['ground_truth'], sample_no['prediction'],
                            labels=all_labels_no, average='macro', zero_division=0)

            with_scores.append(f1_with)
            no_scores.append(f1_no)

        with_scores = np.array(with_scores)
        no_scores = np.array(no_scores)

    # Perform Mann-Whitney U test
    statistic, p_value = mannwhitneyu(with_scores, no_scores, alternative='two-sided')

    return p_value, statistic, with_scores, no_scores

print("Mann-Whitney U test function defined")"""

# New cell 2: Run Mann-Whitney U tests for all models
cell2_source = """# Run Mann-Whitney U Tests for All Models
print("="*80)
print("MANN-WHITNEY U TEST RESULTS")
print("="*80)
print("\\nThis test compares the distributions of with-VAF vs no-VAF predictions")
print("It does not assume normal distribution (non-parametric test)\\n")

mannwhitney_results = {
    'accuracy': [],
    'f1': []
}

for metric in ['accuracy', 'f1']:
    print(f"\\n{'='*80}")
    print(f"METRIC: {metric.upper()}")
    print('='*80)

    for model in MODEL_NAMES:
        if model not in with_vaf_data or model not in no_vaf_data:
            continue

        print(f"  Testing {model}...")
        p_value, statistic, with_scores, no_scores = mann_whitney_test_vaf(
            with_vaf_data, no_vaf_data, model, metric=metric
        )

        # Calculate mean difference
        mean_with = np.mean(with_scores)
        mean_no = np.mean(no_scores)

        if metric == 'accuracy':
            mean_diff = (mean_with - mean_no) * 100  # Convert to percentage
            mean_with_pct = mean_with * 100
            mean_no_pct = mean_no * 100
        else:
            mean_diff = mean_with - mean_no
            mean_with_pct = mean_with
            mean_no_pct = mean_no

        mannwhitney_results[metric].append({
            'model': model,
            'p_value': p_value,
            'statistic': statistic,
            'mean_with_vaf': mean_with_pct,
            'mean_no_vaf': mean_no_pct,
            'mean_diff': mean_diff
        })

        # Determine significance
        if p_value < 0.001:
            sig = '***'
        elif p_value < 0.01:
            sig = '**'
        elif p_value < 0.05:
            sig = '*'
        else:
            sig = 'ns'

        if metric == 'accuracy':
            print(f"    p-value: {p_value:.4f} ({sig})")
            print(f"    Mean diff: {mean_diff:+.2f}% (With: {mean_with_pct:.2f}%, No: {mean_no_pct:.2f}%)")
        else:
            print(f"    p-value: {p_value:.4f} ({sig})")
            print(f"    Mean diff: {mean_diff:+.4f} (With: {mean_with_pct:.4f}, No: {mean_no_pct:.4f})")

# Create DataFrames and save results
mw_acc_df = pd.DataFrame(mannwhitney_results['accuracy']).sort_values('p_value')
mw_f1_df = pd.DataFrame(mannwhitney_results['f1']).sort_values('p_value')

print("\\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"Significant models for accuracy (p < 0.05): {(mw_acc_df['p_value'] < 0.05).sum()}/{len(mw_acc_df)}")
print(f"Significant models for F1 score (p < 0.05): {(mw_f1_df['p_value'] < 0.05).sum()}/{len(mw_f1_df)}")

# Save to CSV
mw_acc_df.to_csv(os.path.join(OUTPUT_DIR, 'vaf_mannwhitney_test_accuracy.csv'), index=False)
mw_f1_df.to_csv(os.path.join(OUTPUT_DIR, 'vaf_mannwhitney_test_f1.csv'), index=False)

print(f"\\nResults saved to:")
print(f"  {os.path.join(OUTPUT_DIR, 'vaf_mannwhitney_test_accuracy.csv')}")
print(f"  {os.path.join(OUTPUT_DIR, 'vaf_mannwhitney_test_f1.csv')}")"""

# New cell 3: Create bar plot with Mann-Whitney U test significance
cell3_source = """# Create Bar Plot with Mann-Whitney U Test Significance
def create_mannwhitney_barplot(results_df, mw_results_df, metric='accuracy'):
    '''
    Create a grouped bar plot comparing with VAF vs no VAF with Mann-Whitney U test significance.
    Similar to the style in the provided example image.
    '''
    fig, ax = plt.subplots(figsize=(14, 8), dpi=150)

    # Get models in the order they appear in results
    models = []
    for model in MODEL_NAMES:
        if model in results_df['model'].values:
            models.append(model)

    # Prepare data
    with_vaf_values = []
    no_vaf_values = []
    p_values = []

    for model in models:
        with_vaf = results_df[(results_df['model'] == model) &
                             (results_df['condition'] == 'With VAF')][metric].values
        no_vaf = results_df[(results_df['model'] == model) &
                           (results_df['condition'] == 'No VAF')][metric].values

        if len(with_vaf) > 0 and len(no_vaf) > 0:
            with_vaf_values.append(with_vaf[0])
            no_vaf_values.append(no_vaf[0])

            # Get p-value from Mann-Whitney results
            mw_row = mw_results_df[mw_results_df['model'] == model]
            if len(mw_row) > 0:
                p_values.append(mw_row['p_value'].values[0])
            else:
                p_values.append(1.0)
        else:
            with_vaf_values.append(0)
            no_vaf_values.append(0)
            p_values.append(1.0)

    # Set up bar positions
    x = np.arange(len(models))
    width = 0.35

    # Create bars
    bars1 = ax.bar(x - width/2, with_vaf_values, width, label='With VAF',
                   color='#4A90E2', edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, no_vaf_values, width, label='No VAF',
                   color='#E74C3C', edgecolor='black', linewidth=1.5)

    # Add value labels on bars
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        height1 = bar1.get_height()
        height2 = bar2.get_height()

        if metric == 'accuracy':
            ax.text(bar1.get_x() + bar1.get_width()/2., height1 + 0.3,
                   f'{height1:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
            ax.text(bar2.get_x() + bar2.get_width()/2., height2 + 0.3,
                   f'{height2:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        else:
            ax.text(bar1.get_x() + bar1.get_width()/2., height1 + 0.005,
                   f'{height1:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
            ax.text(bar2.get_x() + bar2.get_width()/2., height2 + 0.005,
                   f'{height2:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Add significance annotations
    max_height = max(max(with_vaf_values), max(no_vaf_values))

    for i, (model, p_val) in enumerate(zip(models, p_values)):
        # Determine significance level
        if p_val < 0.001:
            sig_text = '***'
        elif p_val < 0.01:
            sig_text = '**'
        elif p_val < 0.05:
            sig_text = '*'
        else:
            sig_text = 'ns'

        # Calculate bar heights for this pair
        h1 = with_vaf_values[i]
        h2 = no_vaf_values[i]
        y_max = max(h1, h2)

        # Draw bracket
        if metric == 'accuracy':
            bracket_height = y_max + 1.5
            line_height = y_max + 1.0
        else:
            bracket_height = y_max + 0.02
            line_height = y_max + 0.015

        # Horizontal line
        ax.plot([i - width/2, i + width/2], [bracket_height, bracket_height],
               color='black', linewidth=1.5)
        # Vertical lines
        ax.plot([i - width/2, i - width/2], [line_height, bracket_height],
               color='black', linewidth=1.5)
        ax.plot([i + width/2, i + width/2], [line_height, bracket_height],
               color='black', linewidth=1.5)

        # Add significance text
        ax.text(i, bracket_height + (0.3 if metric == 'accuracy' else 0.005),
               sig_text, ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Customize plot
    ax.set_xlabel('Model', fontsize=14, fontweight='bold')
    if metric == 'accuracy':
        ax.set_ylabel('Accuracy (%)', fontsize=14, fontweight='bold')
        ax.set_ylim([89, max_height + 4])
        title = 'Accuracy Comparison: With VAF vs No VAF'
    else:
        ax.set_ylabel('F1 Score', fontsize=14, fontweight='bold')
        ax.set_ylim([0, max_height + 0.05])
        title = 'F1 Score Comparison: With VAF vs No VAF'

    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right', fontsize=12)
    ax.legend(loc='upper right', fontsize=12, frameon=True, shadow=True)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add significance legend
    sig_text = '*** p<0.001  ** p<0.01  * p<0.05  ns p≥0.05'
    ax.text(0.02, 0.98, sig_text, transform=ax.transAxes,
           fontsize=11, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    return fig

# Create plots for accuracy and F1
print("Creating Mann-Whitney U test bar plots...\\n")

for metric in ['accuracy', 'f1']:
    if metric == 'accuracy':
        mw_df = mw_acc_df
    else:
        mw_df = mw_f1_df

    fig = create_mannwhitney_barplot(results_df, mw_df, metric=metric)

    output_path = os.path.join(PLOTS_DIR, f'vaf_mannwhitney_{metric}_barplot.pdf')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved Mann-Whitney {metric} plot:")
    print(f"  {output_path}")
    print(f"  {output_path.replace('.pdf', '.png')}")

print("\\nAll Mann-Whitney U test visualizations complete!")"""

# Create the new cells
new_cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## Mann-Whitney U Test Analysis\n",
            "\n",
            "The Mann-Whitney U test (also known as Wilcoxon rank-sum test) is a non-parametric test that doesn't assume normal distribution. It's appropriate for comparing two independent samples to determine if they come from the same distribution.\n",
            "\n",
            "This test is useful because:\n",
            "- It doesn't require normality assumptions\n",
            "- It's robust to outliers\n",
            "- It tests whether the distributions are different\n",
            "\n",
            "We'll use it to compare the performance distributions of models with VAF vs without VAF."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": cell1_source.split('\n')
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": cell2_source.split('\n')
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": cell3_source.split('\n')
    }
]

# Insert the new cells before the last empty cell
# Find the last empty cell
for i in range(len(notebook['cells']) - 1, -1, -1):
    if notebook['cells'][i]['cell_type'] == 'code' and not notebook['cells'][i]['source']:
        insert_position = i
        break

# Insert new cells
for cell in reversed(new_cells):
    notebook['cells'].insert(insert_position, cell)

# Write back to the notebook
with open(notebook_path, 'w') as f:
    json.dump(notebook, f, indent=1)

print(f"Successfully added {len(new_cells)} new cells to the notebook!")
print(f"Inserted at position {insert_position}")
print(f"Total cells in notebook: {len(notebook['cells'])}")

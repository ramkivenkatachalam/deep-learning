# Generate experiment results visualization from results.csv
# Usage: uv run python plot_results.py [assignment_folder]
# If no argument, uses current working directory.
# When framework column exists with multiple values, generates separate charts per framework.

import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch

# Determine assignment folder
if len(sys.argv) > 1:
    folder = sys.argv[1]
else:
    folder = os.getcwd()

results_path = os.path.join(folder, 'results.csv')
if not os.path.exists(results_path):
    print(f"No results.csv found in {folder}")
    sys.exit(1)

# Derive title from folder name (e.g. "1_heart-disease" -> "Heart Disease")
folder_name = os.path.basename(os.path.abspath(folder))
# Strip leading number prefix like "1_" or "2_"
title_base = folder_name.lstrip('0123456789').lstrip('_')
title = title_base.replace('-', ' ').replace('_', ' ').title()

# Detect column count from header; split accordingly to keep description intact
lines = open(results_path).readlines()
n_cols = len(lines[0].strip().split(','))
max_splits = n_cols - 1
header = lines[0].strip().split(',', max_splits)
rows = [line.strip().split(',', max_splits) for line in lines[1:] if line.strip()]
df = pd.DataFrame(rows, columns=header)
# Backfill missing columns for older results.csv files
for col in ['model', 'framework']:
    if col not in df.columns:
        df[col] = ''
df['test_accuracy'] = df['test_accuracy'].astype(float)

df = df[df['status'].isin(['keep', 'discard'])]
df = df.reset_index(drop=True)


def plot_chart(ax, sub_df, chart_title):
    """Plot a single experiment chart on the given axes."""
    sub_df = sub_df.reset_index(drop=True)
    sub_df.index += 1  # 1-based experiment numbering

    colors = ['#2ecc71' if s == 'keep' else '#cccccc' for s in sub_df['status']]
    ax.bar(sub_df.index, sub_df['test_accuracy'], color=colors, width=0.7, edgecolor='none')

    best_so_far = sub_df['test_accuracy'].cummax()
    ax.step(sub_df.index, best_so_far, where='mid', color='#e74c3c', linewidth=2)

    for i, row in sub_df[sub_df['status'] == 'keep'].iterrows():
        label = row['description']
        if len(label) > 35:
            label = label[:32] + '...'
        ax.annotate(
            label,
            xy=(i, row['test_accuracy']),
            xytext=(0, 12), textcoords='offset points',
            ha='center', va='bottom', fontsize=7,
            fontweight='bold', color='#2c3e50',
            arrowprops=dict(arrowstyle='-', color='#2c3e50', lw=0.8),
        )

    ax.set_xlabel('Experiment #', fontsize=11)
    ax.set_ylabel('Test Accuracy', fontsize=11)
    ax.set_title(chart_title, fontsize=13, fontweight='bold')
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    if len(sub_df) > 0:
        ax.axhline(y=sub_df['test_accuracy'].iloc[0], color='#3498db', linestyle='--', linewidth=1, alpha=0.7)
    ax.grid(axis='y', alpha=0.3)

    legend_elements = [
        Patch(facecolor='#2ecc71', label='Keep'),
        Patch(facecolor='#cccccc', label='Discard'),
        plt.Line2D([0], [0], color='#e74c3c', linewidth=2, label='Best so far'),
        plt.Line2D([0], [0], color='#3498db', linestyle='--', linewidth=1, label='Baseline'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)


# Determine frameworks present
frameworks = sorted(df['framework'].unique())
frameworks = [f for f in frameworks if f]  # drop empty strings

if len(frameworks) > 1:
    # Separate chart per framework
    fig, axes = plt.subplots(len(frameworks), 1, figsize=(14, 5 * len(frameworks)))
    if len(frameworks) == 1:
        axes = [axes]
    for ax, fw in zip(axes, frameworks):
        fw_df = df[df['framework'] == fw]
        plot_chart(ax, fw_df, f'Autoresearch: {title} — {fw.title()}')
    plt.tight_layout()
    output_path = os.path.join(folder, 'experiments.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved {output_path}")
else:
    # Single chart (no framework column or only one framework)
    fig, ax = plt.subplots(figsize=(14, 5))
    plot_chart(ax, df, f'Autoresearch: {title} Experiments')
    plt.tight_layout()
    output_path = os.path.join(folder, 'experiments.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved {output_path}")

# Generate per-model experiment charts from results.csv
# Usage: uv run python plot_results_by_model.py [assignment_folder]
# If no argument, uses current working directory.
#
# Automatically detects models from the description column (first word).
# Generates one chart per model: experiments_{model}.png

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

results_path = os.path.join(folder, "results.csv")
if not os.path.exists(results_path):
    print(f"No results.csv found in {folder}")
    sys.exit(1)

# Derive title from folder name
folder_name = os.path.basename(os.path.abspath(folder))
title_base = folder_name.lstrip("0123456789").lstrip("_")
project_title = title_base.replace("-", " ").replace("_", " ").title()

# Parse CSV (description may contain commas)
lines = open(results_path).readlines()
header = lines[0].strip().split(",", 8)
rows = [line.strip().split(",", 8) for line in lines[1:] if line.strip()]
df = pd.DataFrame(rows, columns=header)
df["test_accuracy"] = df["test_accuracy"].astype(float)
df = df[df["status"].isin(["keep", "discard"])].reset_index(drop=True)

# Extract model name from description (first word)
df["model"] = df["description"].str.split().str[0]
models = list(df["model"].unique())

if len(models) <= 1:
    print(f"Only {len(models)} model(s) found — use plot_results.py instead for single-model projects.")
    sys.exit(0)

print(f"Found {len(models)} models: {', '.join(models)}")

for model in models:
    mdf = df[df["model"] == model].reset_index(drop=True)
    if len(mdf) == 0:
        continue
    mdf.index += 1

    # Auto-generate display title from model name
    model_title = model.replace("_", " ").title()

    fig, ax = plt.subplots(figsize=(max(8, len(mdf) * 0.6), 5))

    colors = ["#2ecc71" if s == "keep" else "#cccccc" for s in mdf["status"]]
    ax.bar(mdf.index, mdf["test_accuracy"], color=colors, width=0.7, edgecolor="none")

    best_so_far = mdf["test_accuracy"].cummax()
    ax.step(mdf.index, best_so_far, where="mid", color="#e74c3c", linewidth=2)

    # Annotate kept experiments
    for i, row in mdf[mdf["status"] == "keep"].iterrows():
        # Strip model name prefix from label
        label = row["description"]
        if label.startswith(model + " "):
            label = label[len(model) + 1:]
        if len(label) > 30:
            label = label[:27] + "..."
        ax.annotate(
            label,
            xy=(i, row["test_accuracy"]),
            xytext=(0, 12),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
            color="#2c3e50",
            arrowprops=dict(arrowstyle="-", color="#2c3e50", lw=0.8),
        )

    ax.set_xlabel("Experiment #", fontsize=11)
    ax.set_ylabel("Test Accuracy", fontsize=11)
    ax.set_title(f"Autoresearch: {model_title}", fontsize=13, fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax.axhline(
        y=mdf["test_accuracy"].iloc[0],
        color="#3498db",
        linestyle="--",
        linewidth=1,
        alpha=0.7,
    )
    ax.set_xticks(mdf.index)
    ax.grid(axis="y", alpha=0.3)

    legend_elements = [
        Patch(facecolor="#2ecc71", label="Keep"),
        Patch(facecolor="#cccccc", label="Discard"),
        plt.Line2D([0], [0], color="#e74c3c", linewidth=2, label="Best so far"),
        plt.Line2D([0], [0], color="#3498db", linestyle="--", linewidth=1, label="Baseline"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

    plt.tight_layout()
    out = os.path.join(folder, f"experiments_{model}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")

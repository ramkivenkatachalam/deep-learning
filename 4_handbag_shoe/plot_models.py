# Generate model comparison chart for handbag-shoe classifier
# Usage: uv run python plot_models.py

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

models = ["CNN\n(baseline)", "CNN\n(tuned)", "CNN+Aug\n(baseline)", "CNN+Aug\n(tuned)", "ResNet50\n(feat.extract)", "ResNet50\n(end-to-end)"]
test_acc = [0.7692, 0.8205, 0.6154, 0.8205, 1.0000, 1.0000]
colors = ["#bdc3c7", "#2ecc71", "#bdc3c7", "#2ecc71", "#3498db", "#3498db"]

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(range(len(models)), test_acc, color=colors, width=0.6, edgecolor="none")

for bar, acc in zip(bars, test_acc):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f"{acc:.1%}", ha="center", va="bottom", fontsize=10, fontweight="bold")

ax.set_xticks(range(len(models)))
ax.set_xticklabels(models, fontsize=9)
ax.set_ylabel("Test Accuracy", fontsize=11)
ax.set_title("Handbag vs Shoe: Model Comparison", fontsize=13, fontweight="bold")
ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
ax.set_ylim(0, 1.12)
ax.grid(axis="y", alpha=0.3)

from matplotlib.patches import Patch
legend = [
    Patch(facecolor="#bdc3c7", label="Baseline"),
    Patch(facecolor="#2ecc71", label="After tuning"),
    Patch(facecolor="#3498db", label="Transfer learning"),
]
ax.legend(handles=legend, loc="upper left", fontsize=9)

plt.tight_layout()
plt.savefig("experiments.png", dpi=150, bbox_inches="tight")
print("Saved experiments.png")

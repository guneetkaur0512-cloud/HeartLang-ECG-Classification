import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path

out_dir = Path(r"D:\HeartLang_HPC_Backup\figures")
out_dir.mkdir(parents=True, exist_ok=True)

models = [
    "Linear",
    "MLP",
    "CNN",
    "GRU",
    "ResNet",
    "HeartLang",
    "HeartLang + ECG\nPreprocessing (Proposed)",
]

metrics = ["AUC", "Accuracy", "Precision", "Recall", "F1-score"]

data = np.array([
    [0.7360, 0.3458, 0.5140, 0.4865, 0.4967],
    [0.8000, 0.4384, 0.6562, 0.4740, 0.5391],
    [0.8749, 0.4838, 0.7297, 0.5921, 0.6407],
    [0.8827, 0.5545, 0.7450, 0.6000, 0.6516],
    [0.8868, 0.5479, 0.7584, 0.5845, 0.6439],
    [0.9020, 0.5618, 0.7358, 0.6662, 0.6792],
    [0.9126, 0.5779, 0.7608, 0.6549, 0.6791],
])

plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 12,
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

fig, ax = plt.subplots(figsize=(9, 5.8))

im = ax.imshow(
    data,
    cmap="YlGnBu",
    aspect="auto",
    vmin=0.30,
    vmax=0.95,
)

ax.set_xticks(np.arange(len(metrics)))
ax.set_yticks(np.arange(len(models)))
ax.set_xticklabels(metrics, fontsize=12)
ax.set_yticklabels(models, fontsize=12)

proposed_tick = ax.get_yticklabels()[-1]
proposed_tick.set_fontweight("bold")
proposed_tick.set_color("#1f4e79")

column_max = data.max(axis=0)

for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        value = data[i, j]
        text_color = "white" if value >= 0.72 else "black"
        is_best = column_max[j] - value < 0.0015

        ax.text(
            j,
            i,
            f"{value:.3f}",
            ha="center",
            va="center",
            color=text_color,
            fontsize=12,
            fontweight="bold" if is_best else "normal",
        )

ax.set_xticks(np.arange(data.shape[1] + 1) - 0.5, minor=True)
ax.set_yticks(np.arange(data.shape[0] + 1) - 0.5, minor=True)
ax.grid(which="minor", color="white", linestyle="-", linewidth=1.0)
ax.tick_params(which="minor", bottom=False, left=False)
ax.tick_params(axis="both", which="major", length=0, pad=8)

for spine in ax.spines.values():
    spine.set_visible(False)

highlight = Rectangle(
    (-0.5, data.shape[0] - 1.5),
    data.shape[1],
    1,
    fill=False,
    edgecolor="black",
    linewidth=1.2,
)
ax.add_patch(highlight)

cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.035)
cbar.set_label("Performance Score", fontsize=12)
cbar.ax.tick_params(labelsize=11)
cbar.outline.set_visible(False)

ax.set_title(
    "Comparison of Classification Performance Across Baseline Models and the Proposed HeartLang Framework",
    fontsize=15,
    fontweight="bold",
    pad=16,
)

plt.tight_layout(pad=1.2)

plt.savefig(
    out_dir / "performance_metrics_heatmap_ieee.png",
    dpi=600,
    bbox_inches="tight",
)

plt.close()
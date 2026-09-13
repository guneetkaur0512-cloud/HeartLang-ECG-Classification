import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


out_dir = Path(r"D:\HeartLang_HPC_Backup\figures")
out_dir.mkdir(parents=True, exist_ok=True)

data = [
    ["Linear", 0.7360, 0.3458, 0.5140, 0.4865, 0.4967],
    ["MLP", 0.8000, 0.4384, 0.6562, 0.4740, 0.5391],
    ["CNN", 0.8749, 0.4838, 0.7297, 0.5921, 0.6407],
    ["GRU", 0.8827, 0.5545, 0.7450, 0.6000, 0.6516],
    ["ResNet", 0.8868, 0.5479, 0.7584, 0.5845, 0.6439],
    ["HeartLang", 0.9020, 0.5618, 0.7358, 0.6662, 0.6792],
    ["HeartLang +\nPreprocessing", 0.9126, 0.5779, 0.7608, 0.6549, 0.6791],
]

columns = ["Model", "ROC-AUC", "Accuracy", "Precision", "Recall", "F1-score"]
df = pd.DataFrame(data, columns=columns)

metrics = ["ROC-AUC", "Accuracy", "Precision", "Recall", "F1-score"]
models = df["Model"].tolist()

x = np.arange(len(models))
bar_width = 0.14

normal_colors = {
    "ROC-AUC": "#4C78A8",
    "Accuracy": "#B279A2",
    "Precision": "#F58518",
    "Recall": "#54A24B",
    "F1-score": "#E45756",
}

highlight_colors = {
    "ROC-AUC": "#1f4e79",
    "Accuracy": "#7B3F98",
    "Precision": "#C45A00",
    "Recall": "#2E7D32",
    "F1-score": "#B22222",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.titlesize": 24,
    "axes.labelsize": 18,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 14,
})

fig, ax = plt.subplots(figsize=(18, 7.5))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

final_model_idx = len(models) - 1

for i, metric in enumerate(metrics):
    offset = (i - (len(metrics) - 1) / 2) * bar_width

    values = df[metric].values

    colors = [
        highlight_colors[metric] if j == final_model_idx else normal_colors[metric]
        for j in range(len(models))
    ]

    bars = ax.bar(
        x + offset,
        values,
        width=bar_width,
        label=metric,
        color=colors,
        edgecolor="black",
        linewidth=0.8,
    )

    if metric == "ROC-AUC":
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.012,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=12,
                rotation=0,
            )

ax.set_title(
    "Performance Comparison of Baseline Models and Proposed HeartLang Framework",
    fontsize=24,
    pad=18,
)

ax.set_xlabel("Classification Model", fontsize=18, labelpad=12)
ax.set_ylabel("Performance Score", fontsize=18, labelpad=12)

ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylim(0.3, 1.0)

ax.grid(
    axis="y",
    linestyle="--",
    color="lightgray",
    alpha=0.8,
)

ax.set_axisbelow(True)

ax.legend(
    loc="upper left",
    frameon=True,
    facecolor="white",
    edgecolor="lightgray",
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()

png_path = out_dir / "grouped_model_metric_comparison.png"
pdf_path = out_dir / "grouped_model_metric_comparison.pdf"

plt.savefig(png_path, dpi=600, bbox_inches="tight")
plt.savefig(pdf_path, bbox_inches="tight")

print("Saved:")
print(png_path)
print(pdf_path)
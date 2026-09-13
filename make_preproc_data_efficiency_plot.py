import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

out_dir = Path(r"D:\HeartLang_HPC_Backup\figures")
out_dir.mkdir(parents=True, exist_ok=True)

fractions = np.array([10, 25, 50, 100])
auc = np.array([
    0.8682134807829615,
    0.8772678525160202,
    0.8952348403150486,
    0.9126,
])

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.linewidth": 0.9,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

fig, ax = plt.subplots(figsize=(7.2, 4.6))

line_color = "#1f4e79"

ax.plot(
    fractions,
    auc,
    color=line_color,
    marker="o",
    markersize=6,
    linewidth=2.2,
)

for x, y in zip(fractions, auc):
    ax.text(
        x,
        y + 0.0014,
        f"{y:.4f}",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
    )

ax.set_title(
    "Data-Efficiency Analysis of HeartLang with ECG Preprocessing",
    fontsize=13,
    fontweight="bold",
    pad=8,
)

ax.set_xlabel("Training Data Fraction (%)", fontsize=12, fontweight="bold")
ax.set_ylabel("Test ROC-AUC", fontsize=12, fontweight="bold")

ax.set_xticks(fractions)
ax.set_xticklabels(["10%", "25%", "50%", "100%"], fontsize=11)
ax.tick_params(axis="y", labelsize=11)

ax.set_ylim(0.86, 0.92)
ax.set_xlim(6, 104)

ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.45)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()

png_path = out_dir / "data_efficiency_preprocessed_heartlang_ieee.png"
pdf_path = out_dir / "data_efficiency_preprocessed_heartlang_ieee.pdf"

plt.savefig(png_path, dpi=600, bbox_inches="tight")
plt.savefig(pdf_path, bbox_inches="tight")

print(f"Saved: {png_path}")
print(f"Saved: {pdf_path}")
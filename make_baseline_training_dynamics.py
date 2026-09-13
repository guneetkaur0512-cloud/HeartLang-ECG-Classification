import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

csv_path = Path(r"D:\HeartLang_HPC_Backup\results\baseline_training_curves.csv")
out_dir = Path(r"D:\HeartLang_HPC_Backup\figures")
out_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(csv_path)

model_order = ["linear", "mlp", "cnn", "gru", "resnet"]
model_labels = {
    "linear": "Linear",
    "mlp": "MLP",
    "cnn": "CNN",
    "gru": "GRU",
    "resnet": "ResNet",
}

colors = {
    "linear": "#6c757d",
    "mlp": "#9467bd",
    "cnn": "#1f77b4",
    "gru": "#ff7f0e",
    "resnet": "#2ca02c",
}

plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 12,
    "axes.linewidth": 0.9,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

summary = (
    df.groupby(["model", "epoch"])
    .agg(
        train_loss_mean=("train_loss", "mean"),
        train_loss_std=("train_loss", "std"),
        val_auc_mean=("val_auc", "mean"),
        val_auc_std=("val_auc", "std"),
    )
    .reset_index()
)

fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

ax_loss, ax_auc = axes

for model in model_order:
    sub = summary[summary["model"] == model].sort_values("epoch")
    epochs = sub["epoch"].values

    train_mean = sub["train_loss_mean"].values
    train_std = sub["train_loss_std"].fillna(0).values

    auc_mean = sub["val_auc_mean"].values
    auc_std = sub["val_auc_std"].fillna(0).values

    ax_loss.plot(
        epochs,
        train_mean,
        label=model_labels[model],
        color=colors[model],
        linewidth=2.2,
        marker="o",
        markersize=4.5,
    )
    ax_loss.fill_between(
        epochs,
        train_mean - train_std,
        train_mean + train_std,
        color=colors[model],
        alpha=0.12,
        linewidth=0,
    )

    ax_auc.plot(
        epochs,
        auc_mean,
        label=model_labels[model],
        color=colors[model],
        linewidth=2.2,
        marker="o",
        markersize=4.5,
    )
    ax_auc.fill_between(
        epochs,
        auc_mean - auc_std,
        auc_mean + auc_std,
        color=colors[model],
        alpha=0.12,
        linewidth=0,
    )

ax_loss.set_title("(a) Training Loss", fontsize=15)
ax_auc.set_title("(b) Validation ROC-AUC", fontsize=15)

ax_loss.set_xlabel("Training Epoch", fontsize=13)
ax_auc.set_xlabel("Training Epoch", fontsize=13)

ax_loss.set_ylabel("Training Loss", fontsize=13)
ax_auc.set_ylabel("Validation ROC-AUC", fontsize=13)

ax_loss.grid(True, linestyle="--", alpha=0.35)
ax_auc.grid(True, linestyle="--", alpha=0.35)

ax_loss.set_xticks(range(1, 11))
ax_auc.set_xticks(range(1, 11))

ax_auc.set_ylim(0.70, 0.91)

for ax in axes:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=11)

handles, labels = ax_auc.get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="lower center",
    ncol=5,
    frameon=False,
    fontsize=11,
    bbox_to_anchor=(0.5, -0.04),
)

fig.suptitle(
    "Training Dynamics of Baseline Models on PTB-XL Superdiagnostic Classification",
    fontsize=17,
    fontweight="bold",
    y=1.02,
)

plt.tight_layout(rect=[0, 0.07, 1, 0.96])

png_path = out_dir / "baseline_training_dynamics.png"
pdf_path = out_dir / "baseline_training_dynamics.pdf"

plt.savefig(png_path, dpi=600, bbox_inches="tight")
plt.savefig(pdf_path, bbox_inches="tight")

print(f"Saved: {png_path}")
print(f"Saved: {pdf_path}")
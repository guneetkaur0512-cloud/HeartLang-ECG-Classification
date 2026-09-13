import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

backup = Path(r"D:\HeartLang_HPC_Backup")
out_dir = backup / "figures"
out_dir.mkdir(parents=True, exist_ok=True)

baseline_csv = backup / "results" / "baseline_training_curves.csv"

heartlang_logs = {
    "HeartLang": [
        backup / "finetune_superdiagnostic_gpu_all_1.00_random_10ep_seed0" / "log.txt",
        backup / "finetune_superdiagnostic_gpu_all_1.00_random_10ep_seed1" / "log.txt",
        backup / "finetune_superdiagnostic_gpu_all_1.00_random_10ep_seed2" / "log.txt",
    ],
    "HeartLang + ECG Preprocessing": [
        backup / "finetune_superdiagnostic_gpu_all_preproc_1.00_random_10ep_seed0" / "log.txt",
        backup / "finetune_superdiagnostic_gpu_all_preproc_1.00_random_10ep_seed1" / "log.txt",
        backup / "finetune_superdiagnostic_gpu_all_preproc_1.00_random_10ep_seed2" / "log.txt",
    ],
}

model_order = [
    "Linear",
    "MLP",
    "CNN",
    "GRU",
    "ResNet",
    "HeartLang",
    "HeartLang + ECG Preprocessing",
]

colors = {
    "Linear": "#7a7a7a",
    "MLP": "#9467bd",
    "CNN": "#1f77b4",
    "GRU": "#ff7f0e",
    "ResNet": "#8c564b",
    "HeartLang": "#1f4e79",
    "HeartLang + ECG Preprocessing": "#2ca02c",
}

plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 12,
    "axes.linewidth": 0.9,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

records = []

# -----------------------------
# Load baseline curves
# -----------------------------
df_base = pd.read_csv(baseline_csv)

baseline_name_map = {
    "linear": "Linear",
    "mlp": "MLP",
    "cnn": "CNN",
    "gru": "GRU",
    "resnet": "ResNet",
}

for _, row in df_base.iterrows():
    records.append({
        "model": baseline_name_map[row["model"]],
        "seed": int(row["seed"]),
        "epoch": int(row["epoch"]),
        "train_loss": float(row["train_loss"]),
        "val_auc": float(row["val_auc"]),
    })

# -----------------------------
# Load HeartLang logs
# -----------------------------
for model_name, paths in heartlang_logs.items():
    for seed, path in enumerate(paths):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                row = json.loads(line)

                if "epoch" in row and "train_loss" in row and "val_roc_auc" in row:
                    records.append({
                        "model": model_name,
                        "seed": seed,
                        "epoch": int(row["epoch"]),
                        "train_loss": float(row["train_loss"]),
                        "val_auc": float(row["val_roc_auc"]),
                    })

df = pd.DataFrame(records)

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

summary["train_loss_std"] = summary["train_loss_std"].fillna(0)
summary["val_auc_std"] = summary["val_auc_std"].fillna(0)

# -----------------------------
# Plot
# -----------------------------
fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
ax_loss, ax_auc = axes

for model in model_order:
    sub = summary[summary["model"] == model].sort_values("epoch")
    epochs = sub["epoch"].values
    color = colors[model]

    is_main = model == "HeartLang + ECG Preprocessing"

    linewidth = 3.4 if is_main else 1.8
    markersize = 6.5 if is_main else 4.0
    alpha = 1.0 if is_main else 0.65
    linestyle = "-" if is_main else "--"
    zorder = 10 if is_main else 3

    label = model + " (Proposed)" if is_main else model

    ax_loss.plot(
        epochs,
        sub["train_loss_mean"].values,
        label=label,
        color=color,
        linewidth=linewidth,
        marker="o",
        markersize=markersize,
        linestyle=linestyle,
        alpha=alpha,
        zorder=zorder,
    )

    if is_main:
        ax_loss.fill_between(
            epochs,
            sub["train_loss_mean"].values - sub["train_loss_std"].values,
            sub["train_loss_mean"].values + sub["train_loss_std"].values,
            color=color,
            alpha=0.18,
            linewidth=0,
            zorder=1,
        )

    ax_auc.plot(
        epochs,
        sub["val_auc_mean"].values,
        label=label,
        color=color,
        linewidth=linewidth,
        marker="o",
        markersize=markersize,
        linestyle=linestyle,
        alpha=alpha,
        zorder=zorder,
    )

    if is_main:
        ax_auc.fill_between(
            epochs,
            sub["val_auc_mean"].values - sub["val_auc_std"].values,
            sub["val_auc_mean"].values + sub["val_auc_std"].values,
            color=color,
            alpha=0.18,
            linewidth=0,
            zorder=1,
        )

ax_loss.set_title("(a) Training Loss", fontsize=16)
ax_auc.set_title("(b) Validation ROC-AUC", fontsize=16)

ax_loss.set_xlabel("Training Epoch", fontsize=14)
ax_auc.set_xlabel("Training Epoch", fontsize=14)

ax_loss.set_ylabel("Training Loss", fontsize=14)
ax_auc.set_ylabel("Validation ROC-AUC", fontsize=14)

ax_loss.set_xticks(range(1, 11))
ax_auc.set_xticks(range(1, 11))

ax_auc.set_ylim(0.70, 0.93)

for ax in axes:
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=12)

handles, labels = ax_auc.get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc="lower center",
    ncol=4,
    frameon=False,
    fontsize=11,
    bbox_to_anchor=(0.5, -0.06),
)

fig.suptitle(
    "Training Dynamics Comparison Across Baseline Models and the Proposed HeartLang Framework",
    fontsize=18,
    fontweight="bold",
    y=1.03,
)

plt.tight_layout(rect=[0, 0.10, 1, 0.96])

png_path = out_dir / "all_models_training_dynamics_highlighted.png"
pdf_path = out_dir / "all_models_training_dynamics_highlighted.pdf"

plt.savefig(png_path, dpi=600, bbox_inches="tight")
plt.savefig(pdf_path, bbox_inches="tight")

print(f"Saved: {png_path}")
print(f"Saved: {pdf_path}")
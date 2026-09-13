import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


LOG_FILES = [
    Path(r"D:\HeartLang_HPC_Backup\finetune_superdiagnostic_gpu_all_preproc_1.00_random_10ep_seed0\log.txt"),
    Path(r"D:\HeartLang_HPC_Backup\finetune_superdiagnostic_gpu_all_preproc_1.00_random_10ep_seed1\log.txt"),
    Path(r"D:\HeartLang_HPC_Backup\finetune_superdiagnostic_gpu_all_preproc_1.00_random_10ep_seed2\log.txt"),
]

OUT_DIR = Path(r"D:\HeartLang_HPC_Backup\figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def parse_log(path):
    epochs = []
    train_loss = []
    val_auc = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            record = json.loads(line)

            if (
                "epoch" in record
                and "train_loss" in record
                and "val_roc_auc" in record
            ):
                epochs.append(int(record["epoch"]))
                train_loss.append(float(record["train_loss"]))
                val_auc.append(float(record["val_roc_auc"]))

    return {
        "epochs": np.array(epochs),
        "train_loss": np.array(train_loss),
        "val_auc": np.array(val_auc),
    }


runs = [parse_log(path) for path in LOG_FILES]
epochs = runs[0]["epochs"]

for path, run in zip(LOG_FILES, runs):
    if len(run["epochs"]) == 0:
        raise ValueError(f"No valid training records found in {path}")

    if not np.array_equal(run["epochs"], epochs):
        raise ValueError(f"Epoch mismatch found in {path}")


train_loss_values = np.stack([run["train_loss"] for run in runs], axis=0)
val_auc_values = np.stack([run["val_auc"] for run in runs], axis=0)

train_loss_mean = train_loss_values.mean(axis=0)

val_auc_mean = val_auc_values.mean(axis=0)
val_auc_std = val_auc_values.std(axis=0)

best_idx = int(np.argmax(val_auc_mean))
best_epoch = epochs[best_idx]
best_auc = val_auc_mean[best_idx]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.titlesize": 17,
    "axes.labelsize": 15,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "figure.titlesize": 20,
})

fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))

fig.suptitle(
    "Training Dynamics of HeartLang During Full Fine-Tuning",
    fontsize=20,
    y=1.03,
)

# ---------------------------------------------------------
# (a) Training Loss
# ---------------------------------------------------------
ax = axes[0]

ax.plot(
    epochs,
    train_loss_mean,
    color="#1f4e79",
    linewidth=3.0,
    marker="o",
    markersize=7,
)

ax.set_title("(a) Training Loss", fontsize=17)
ax.set_xlabel("Training Epoch", fontsize=15)
ax.set_ylabel("Training Loss", fontsize=15)
ax.grid(True, linestyle="--", color="lightgray", alpha=0.75)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# ---------------------------------------------------------
# (b) Validation ROC-AUC
# ---------------------------------------------------------
ax = axes[1]

# Individual seed curves
for idx, run in enumerate(runs):
    ax.plot(
        epochs,
        run["val_auc"],
        linestyle="--",
        linewidth=1.3,
        color="#2ca02c",
        alpha=0.35,
    )

# Mean curve
ax.plot(
    epochs,
    val_auc_mean,
    color="#2ca02c",
    linewidth=3.5,
    marker="o",
    markersize=8,
)

# Standard deviation band
ax.fill_between(
    epochs,
    val_auc_mean - val_auc_std,
    val_auc_mean + val_auc_std,
    color="#2ca02c",
    alpha=0.20,
)

# Selected checkpoint marker
ax.axvline(
    best_epoch,
    linestyle="--",
    color="gray",
    alpha=0.8,
    linewidth=1.5,
)

y_min = max(0.0, (val_auc_mean - val_auc_std).min() - 0.01)
y_max = min(1.0, (val_auc_mean + val_auc_std).max() + 0.012)
ax.set_ylim(y_min, y_max)

annotation_y = y_max - 0.003
ax.text(
    best_epoch + 0.18,
    annotation_y,
    "Selected checkpoint",
    color="gray",
    fontsize=12,
    rotation=90,
    va="top",
    ha="left",
)

ax.set_title("(b) Validation ROC-AUC", fontsize=17)
ax.set_xlabel("Training Epoch", fontsize=15)
ax.set_ylabel("Validation ROC-AUC", fontsize=15)
ax.grid(True, linestyle="--", color="lightgray", alpha=0.75)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()

png_path = OUT_DIR / "training_dynamics.png"
pdf_path = OUT_DIR / "training_dynamics.pdf"

plt.savefig(png_path, dpi=600, bbox_inches="tight")
plt.savefig(pdf_path, bbox_inches="tight")
plt.close()

print(f"Selected checkpoint epoch: {best_epoch}")
print(f"Mean validation ROC-AUC at selected checkpoint: {best_auc:.6f}")
print(f"Saved PNG: {png_path}")
print(f"Saved PDF: {pdf_path}")
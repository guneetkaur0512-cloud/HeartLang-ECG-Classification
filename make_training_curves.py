
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

log_files = [
    r"D:\HeartLang_HPC_Backup\finetune_superdiagnostic_gpu_all_preproc_1.00_random_10ep_seed0\log.txt",
    r"D:\HeartLang_HPC_Backup\finetune_superdiagnostic_gpu_all_preproc_1.00_random_10ep_seed1\log.txt",
    r"D:\HeartLang_HPC_Backup\finetune_superdiagnostic_gpu_all_preproc_1.00_random_10ep_seed2\log.txt",
]

out_dir = Path(r"D:\HeartLang_HPC_Backup\figures")
out_dir.mkdir(exist_ok=True)

def parse_log(path):
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            try:
                obj = json.loads(line)
            except Exception:
                continue

            epoch = obj.get("epoch")
            train_loss = obj.get("train_loss")
            val_loss = obj.get("val_loss")
            val_auc = obj.get("val_roc_auc", obj.get("val_auc", obj.get("test_auc")))

            if epoch is not None:
                rows.append({
                    "epoch": int(epoch),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_auc": val_auc,
                })

    return rows

all_rows = [parse_log(p) for p in log_files]

for path, rows in zip(log_files, all_rows):
    print(path)
    print("epochs found:", len(rows))
    if rows:
        print("first row:", rows[0])
        print("last row:", rows[-1])

epochs = np.array([r["epoch"] for r in all_rows[0]])

def collect(key):
    curves = []

    for rows in all_rows:
        values = [r[key] for r in rows if r[key] is not None]

        if len(values) == len(epochs):
            curves.append(values)

    if not curves:
        return None, None

    arr = np.array(curves, dtype=float)
    return arr.mean(axis=0), arr.std(axis=0)

train_loss, train_loss_std = collect("train_loss")
val_loss, val_loss_std = collect("val_loss")
val_auc, val_auc_std = collect("val_auc")

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
})

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

# --------------------------------------------------------
# Panel A: Loss curves
# --------------------------------------------------------

ax = axes[0]

if train_loss is not None:
    ax.plot(
        epochs,
        train_loss,
        marker="o",
        linewidth=2,
        color="#1f77b4",
        label="Training loss",
    )
    ax.fill_between(
        epochs,
        train_loss - train_loss_std,
        train_loss + train_loss_std,
        color="#1f77b4",
        alpha=0.15,
    )

if val_loss is not None:
    ax.plot(
        epochs,
        val_loss,
        marker="s",
        linewidth=2,
        color="#ff7f0e",
        label="Validation loss",
    )
    ax.fill_between(
        epochs,
        val_loss - val_loss_std,
        val_loss + val_loss_std,
        color="#ff7f0e",
        alpha=0.15,
    )

ax.set_title("(a) Loss Curves")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.grid(alpha=0.25)
ax.legend(frameon=False)

# --------------------------------------------------------
# Panel B: Validation ROC-AUC
# --------------------------------------------------------

ax = axes[1]

if val_auc is not None:
    ax.plot(
        epochs,
        val_auc,
        marker="o",
        linewidth=2,
        color="#2ca02c",
        label="Validation ROC-AUC",
    )
    ax.fill_between(
        epochs,
        val_auc - val_auc_std,
        val_auc + val_auc_std,
        color="#2ca02c",
        alpha=0.15,
    )

    ymin = max(0.0, val_auc.min() - 0.01)
    ymax = min(1.0, val_auc.max() + 0.01)
    ax.set_ylim(ymin, ymax)

else:
    ax.text(
        0.5,
        0.5,
        "Validation AUC not found",
        ha="center",
        va="center",
        transform=ax.transAxes,
    )

ax.set_title("(b) Validation ROC-AUC")
ax.set_xlabel("Epoch")
ax.set_ylabel("ROC-AUC")
ax.grid(alpha=0.25)
ax.legend(frameon=False)

fig.suptitle(
    "Training Curves for Preprocessed HeartLang Full Fine-Tuning",
    fontsize=13,
)

plt.tight_layout()

png_path = out_dir / "training_curves_combined.png"
pdf_path = out_dir / "training_curves_combined.pdf"

plt.savefig(png_path, dpi=300, bbox_inches="tight")
plt.savefig(pdf_path, bbox_inches="tight")
plt.close()

print("Saved:")
print(png_path)
print(pdf_path)

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

CLASS_NAMES = ["NORM", "MI", "STTC", "CD", "HYP"]

def load_csv(path):
    return pd.read_csv(path).values.astype(float)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--out_dir", default="results/per_class_auc")
    ap.add_argument("--name", default="heartlang")
    ap.add_argument("--logits", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    y_score = load_csv(args.pred)
    y_true = load_csv(args.target)

    if args.logits:
        y_score = 1 / (1 + np.exp(-y_score))

    rows = []
    plt.figure(figsize=(7, 6))

    for i, cls in enumerate(CLASS_NAMES):
        auc = roc_auc_score(y_true[:, i], y_score[:, i])
        fpr, tpr, _ = roc_curve(y_true[:, i], y_score[:, i])
        rows.append({"class": cls, "auc": auc})
        plt.plot(fpr, tpr, label=f"{cls} AUC={auc:.3f}")

    macro_auc = roc_auc_score(y_true, y_score, average="macro")
    rows.append({"class": "macro_average", "auc": macro_auc})

    df = pd.DataFrame(rows)
    csv_path = out_dir / f"{args.name}_per_class_auc.csv"
    fig_path = out_dir / f"{args.name}_roc_curves.png"

    df.to_csv(csv_path, index=False)

    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Per-Class ROC Curves: {args.name}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)

    print(df)
    print("Saved:", csv_path)
    print("Saved:", fig_path)

if __name__ == "__main__":
    main()

from scipy.stats import ttest_rel, wilcoxon
import numpy as np

# -----------------------------
# HeartLang (Without Preprocessing)
# -----------------------------
baseline = np.array([
    0.9005137286,
    0.9023325244,
    0.9030897799
])

# -----------------------------
# HeartLang + Proposed Preprocessing
# -----------------------------
preprocessed = np.array([
    0.9119584743,
    0.9120432294,
    0.9137408436
])

print("=" * 60)
print("STATISTICAL SIGNIFICANCE ANALYSIS")
print("=" * 60)

print("\nBaseline AUCs:")
print(baseline)

print("\nPreprocessed AUCs:")
print(preprocessed)

print("\nMean Baseline AUC      :", baseline.mean())
print("Mean Preprocessed AUC :", preprocessed.mean())
print("Mean Improvement      :", preprocessed.mean() - baseline.mean())

# -------------------------------------------------
# Paired t-test
# -------------------------------------------------
t_stat, p_t = ttest_rel(preprocessed, baseline)

print("\n" + "=" * 60)
print("Paired t-test")
print("=" * 60)
print(f"T-statistic : {t_stat:.6f}")
print(f"P-value     : {p_t:.6f}")

if p_t < 0.05:
    print("Result      : Significant improvement (p < 0.05)")
else:
    print("Result      : Not statistically significant")

# -------------------------------------------------
# Wilcoxon signed-rank test
# -------------------------------------------------
w_stat, p_w = wilcoxon(preprocessed, baseline)

print("\n" + "=" * 60)
print("Wilcoxon Signed-Rank Test")
print("=" * 60)
print(f"Statistic   : {w_stat:.6f}")
print(f"P-value     : {p_w:.6f}")

if p_w < 0.05:
    print("Result      : Significant improvement (p < 0.05)")
else:
    print("Result      : Not statistically significant")

print("\n" + "=" * 60)
print("Analysis Complete")
print("=" * 60)
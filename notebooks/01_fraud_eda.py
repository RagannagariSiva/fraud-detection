"""
notebooks/01_fraud_eda.py
==========================
Exploratory Data Analysis — Credit Card Fraud Detection

This script is formatted as a Jupytext percent-script so it can be run:
  - As a plain Python script:  python notebooks/01_fraud_eda.py
  - As a Jupyter notebook:     jupytext --to notebook notebooks/01_fraud_eda.py

Sections
--------
1. Dataset overview (shape, dtypes, missing values)
2. Class imbalance analysis
3. Feature distributions (V1–V28)
4. Amount and Time analysis
5. Correlation heatmap
6. Fraud vs legitimate feature comparison
7. Key insights summary
"""

# %% [markdown]
# # 🔍 Credit Card Fraud — Exploratory Data Analysis

# %%
# ── Imports ───────────────────────────────────────────────────────────────────
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Add project root to path so we can import src modules
sys.path.insert(0, str(Path(__file__).parent.parent))

# Plotting style
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")
FIGURE_DIR = Path("reports/figures")
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

print("✅ Imports OK")

# %% [markdown]
# ## 1. Load Dataset

# %%
RAW_PATH = "data/raw/creditcard.csv"

try:
    df = pd.read_csv(RAW_PATH)
    print(f"Dataset loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
except FileNotFoundError:
    print(f"❌ Dataset not found at {RAW_PATH}")
    print("   Download from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
    print("   Place the CSV at data/raw/creditcard.csv")
    sys.exit(0)

# %% [markdown]
# ## 2. Dataset Overview

# %%
print("\n── Shape ────────────────────────────────────────")
print(f"  Rows    : {df.shape[0]:,}")
print(f"  Columns : {df.shape[1]}")

print("\n── Data Types ───────────────────────────────────")
print(df.dtypes.value_counts())

print("\n── Missing Values ───────────────────────────────")
null_count = df.isnull().sum().sum()
print(f"  Total nulls: {null_count}")
if null_count == 0:
    print("  ✅ No missing values")

print("\n── Sample ───────────────────────────────────────")
print(df.head(3).to_string())

# %%
# Basic statistics for Amount and Time
print("\n── Amount Statistics ────────────────────────────")
print(df["Amount"].describe().round(4))
print("\n── Time Statistics ──────────────────────────────")
print(df["Time"].describe().round(4))

# %% [markdown]
# ## 3. Class Imbalance

# %%
class_counts = df["Class"].value_counts()
fraud_count  = int(class_counts[1])
legit_count  = int(class_counts[0])
fraud_pct    = fraud_count / len(df) * 100

print("── Class Distribution ───────────────────────────")
print(f"  Legitimate : {legit_count:>9,}  ({100 - fraud_pct:.3f}%)")
print(f"  Fraud      : {fraud_count:>9,}  ({fraud_pct:.3f}%)")
print(f"  Imbalance ratio  : {legit_count // fraud_count}:1")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Bar chart
axes[0].bar(
    ["Legitimate", "Fraud"],
    [legit_count, fraud_count],
    color=["#4CAF50", "#F44336"],
    edgecolor="white",
    linewidth=1.5,
)
axes[0].set_title("Class Distribution (Absolute)", fontweight="bold")
axes[0].set_ylabel("Count")
for bar, count in zip(axes[0].patches, [legit_count, fraud_count]):
    axes[0].text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 1000,
        f"{count:,}",
        ha="center", fontsize=11, fontweight="bold",
    )

# Pie chart (zoomed into fraud)
axes[1].pie(
    [legit_count, fraud_count],
    labels=["Legitimate\n(99.83%)", f"Fraud\n({fraud_pct:.2f}%)"],
    colors=["#4CAF50", "#F44336"],
    startangle=90,
    wedgeprops={"edgecolor": "white", "linewidth": 2},
    autopct="%1.2f%%",
)
axes[1].set_title("Class Proportion", fontweight="bold")

plt.tight_layout()
plt.savefig(FIGURE_DIR / "01_class_distribution.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"  ✅ Saved → {FIGURE_DIR}/01_class_distribution.png")

# %% [markdown]
# ## 4. Feature Distributions (V1–V28)

# %%
# Compare distribution of each V feature for fraud vs legitimate
fraud_df = df[df["Class"] == 1]
legit_df = df[df["Class"] == 0].sample(n=len(fraud_df) * 5, random_state=42)

v_features = [f"V{i}" for i in range(1, 29)]

# Find most discriminative features (largest mean difference)
feature_diff = {}
for feat in v_features:
    fraud_mean = fraud_df[feat].mean()
    legit_mean = legit_df[feat].mean()
    feature_diff[feat] = abs(fraud_mean - legit_mean)

top_features = sorted(feature_diff, key=feature_diff.get, reverse=True)[:12]

fig, axes = plt.subplots(3, 4, figsize=(18, 12))
axes_flat = axes.flatten()

for i, feat in enumerate(top_features):
    ax = axes_flat[i]
    ax.hist(legit_df[feat], bins=50, alpha=0.6, color="#4CAF50", label="Legitimate", density=True)
    ax.hist(fraud_df[feat], bins=50, alpha=0.7, color="#F44336", label="Fraud", density=True)
    ax.set_title(f"{feat}  (Δmean={feature_diff[feat]:.2f})", fontsize=10, fontweight="bold")
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    if i == 0:
        ax.legend(fontsize=8)

plt.suptitle("Top 12 Most Discriminative V Features — Fraud vs Legitimate",
             fontsize=14, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(FIGURE_DIR / "02_feature_distributions.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"  ✅ Saved → {FIGURE_DIR}/02_feature_distributions.png")

print("\nTop 10 most discriminative V features (by |mean difference|):")
for feat in top_features[:10]:
    print(f"  {feat:<5}  Δmean={feature_diff[feat]:.4f}")

# %% [markdown]
# ## 5. Amount & Time Analysis

# %%
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Amount: fraud vs legitimate
axes[0, 0].hist(legit_df["Amount"], bins=60, alpha=0.6, color="#4CAF50",
                label="Legitimate", density=True)
axes[0, 0].hist(fraud_df["Amount"], bins=60, alpha=0.7, color="#F44336",
                label="Fraud", density=True)
axes[0, 0].set_title("Transaction Amount Distribution", fontweight="bold")
axes[0, 0].set_xlabel("Amount ($)")
axes[0, 0].set_ylabel("Density")
axes[0, 0].legend()
axes[0, 0].set_xlim(0, 500)

# Log amount
axes[0, 1].hist(np.log1p(legit_df["Amount"]), bins=50, alpha=0.6, color="#4CAF50",
                label="Legitimate", density=True)
axes[0, 1].hist(np.log1p(fraud_df["Amount"]), bins=50, alpha=0.7, color="#F44336",
                label="Fraud", density=True)
axes[0, 1].set_title("Log(1 + Amount) Distribution", fontweight="bold")
axes[0, 1].set_xlabel("log(1 + Amount)")
axes[0, 1].set_ylabel("Density")
axes[0, 1].legend()

# Time of day: fraud rate per hour
df["hour"] = (df["Time"] % 86400) / 3600
hourly = df.groupby("hour").agg(
    total=("Class", "count"),
    fraud=("Class", "sum"),
).reset_index()
hourly["fraud_rate"] = hourly["fraud"] / hourly["total"]

axes[1, 0].plot(hourly["hour"], hourly["fraud_rate"], color="#F44336", linewidth=2)
axes[1, 0].fill_between(hourly["hour"], hourly["fraud_rate"], alpha=0.2, color="#F44336")
axes[1, 0].set_title("Fraud Rate by Hour of Day", fontweight="bold")
axes[1, 0].set_xlabel("Hour of Day")
axes[1, 0].set_ylabel("Fraud Rate")
axes[1, 0].axvline(x=2, color="gray", linestyle="--", alpha=0.5, label="2am")
axes[1, 0].legend()

# Amount statistics: fraud vs legitimate
amount_stats = pd.DataFrame({
    "Metric": ["Mean", "Median", "Max", "Std"],
    "Legitimate": [
        legit_df["Amount"].mean(),
        legit_df["Amount"].median(),
        legit_df["Amount"].max(),
        legit_df["Amount"].std(),
    ],
    "Fraud": [
        fraud_df["Amount"].mean(),
        fraud_df["Amount"].median(),
        fraud_df["Amount"].max(),
        fraud_df["Amount"].std(),
    ],
})
x = np.arange(len(amount_stats))
w = 0.35
axes[1, 1].bar(x - w/2, amount_stats["Legitimate"], w, color="#4CAF50", label="Legitimate", alpha=0.8)
axes[1, 1].bar(x + w/2, amount_stats["Fraud"],      w, color="#F44336", label="Fraud",      alpha=0.8)
axes[1, 1].set_xticks(x)
axes[1, 1].set_xticklabels(amount_stats["Metric"])
axes[1, 1].set_title("Amount Statistics: Fraud vs Legitimate", fontweight="bold")
axes[1, 1].set_ylabel("Amount ($)")
axes[1, 1].legend()

plt.suptitle("Transaction Amount & Time Analysis", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(FIGURE_DIR / "03_amount_time_analysis.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"  ✅ Saved → {FIGURE_DIR}/03_amount_time_analysis.png")

# %% [markdown]
# ## 6. Correlation Heatmap

# %%
numeric_cols = v_features + ["Amount", "Time"]
corr = df[numeric_cols].corr()

fig, ax = plt.subplots(figsize=(18, 14))
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
sns.heatmap(
    corr,
    ax=ax,
    cmap="RdBu_r",
    center=0,
    vmin=-0.5, vmax=0.5,
    square=True,
    linewidths=0.3,
    annot=False,
    cbar_kws={"shrink": 0.8, "label": "Pearson Correlation"},
)
ax.set_title("Feature Correlation Matrix (V1–V28, Amount, Time)",
             fontsize=14, fontweight="bold", pad=15)
plt.tight_layout()
plt.savefig(FIGURE_DIR / "04_correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"  ✅ Saved → {FIGURE_DIR}/04_correlation_heatmap.png")

# %% [markdown]
# ## 7. Key Insights Summary

# %%
print("""
╔══════════════════════════════════════════════════════════════╗
║              KEY EDA INSIGHTS                                ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  1. EXTREME IMBALANCE                                        ║
║     Fraud = 0.17% of transactions (492 out of 284,807).      ║
║     → Must use SMOTE + class-weighted loss function.         ║
║                                                              ║
║  2. HIGHLY DISCRIMINATIVE FEATURES                           ║
║     V14, V12, V10, V4, V11 show the largest mean shift       ║
║     between fraud and legitimate transactions.               ║
║     → XGBoost will naturally discover these.                 ║
║                                                              ║
║  3. AMOUNT PATTERNS                                          ║
║     Fraudulent amounts cluster more tightly (lower std).     ║
║     Log transformation reduces right skew — include          ║
║     log_amount as an engineered feature.                     ║
║                                                              ║
║  4. TIME-OF-DAY PATTERNS                                     ║
║     Fraud rate peaks between 00:00–04:00 (nighttime).        ║
║     → Include is_nighttime binary feature.                   ║
║                                                              ║
║  5. LOW FEATURE CORRELATION                                  ║
║     V1–V28 are PCA components → near-zero cross-correlation. ║
║     Amount and Time are slightly correlated with some Vs.    ║
║     → No multicollinearity issues for tree models.           ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
""")

print(f"All figures saved to: {FIGURE_DIR}/")
print("Run the full pipeline: python main.py")

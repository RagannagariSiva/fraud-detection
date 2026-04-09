"""
notebooks/02_model_comparison.py
==================================
Model Comparison & Threshold Analysis

Loads all trained models and produces side-by-side comparisons:
  1. PR curves for all models on the test set
  2. ROC curves for all models
  3. Threshold vs. Precision/Recall tradeoff
  4. Confusion matrices (side by side)
  5. Decision threshold sensitivity analysis

Run after training:
    python main.py                          # train first
    python notebooks/02_model_comparison.py
"""

# %%
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

plt.style.use("seaborn-v0_8-darkgrid")
FIGURE_DIR = Path("reports/figures")
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR = Path("models")

print("✅ Imports OK")

# %%
# ── Load models ───────────────────────────────────────────────────────────────
import joblib

models_available = {}
for name, pkl in [
    ("XGBoost",            "xgboost_model.pkl"),
    ("Random Forest",      "random_forest_model.pkl"),
    ("Decision Tree",      "decision_tree_model.pkl"),
]:
    path = MODEL_DIR / pkl
    if path.exists():
        models_available[name] = joblib.load(path)
        print(f"  Loaded: {name}")
    else:
        print(f"  Not found: {name} ({path})")

if not models_available:
    print("\n❌ No models found. Run: python main.py")
    sys.exit(0)

# ── Load scaler + test data ────────────────────────────────────────────────────
scaler_path = MODEL_DIR / "scaler.pkl"
feat_path   = MODEL_DIR / "feature_names.pkl"

if not scaler_path.exists():
    print("❌ Scaler not found. Run: python main.py")
    sys.exit(0)

scaler       = joblib.load(scaler_path)
feature_names = joblib.load(feat_path) if feat_path.exists() else None

# ── Load processed test data (if available) ───────────────────────────────────
test_data_path = Path("data/processed/X_test.csv")
test_labels_path = Path("data/processed/y_test.csv")

if test_data_path.exists() and test_labels_path.exists():
    X_test = pd.read_csv(test_data_path)
    y_test = pd.read_csv(test_labels_path).squeeze()
    print(f"\nTest set loaded: {len(X_test):,} rows")
else:
    # Fall back to loading raw data and splitting it
    print("Test set CSVs not found — loading raw data and splitting...")
    try:
        from src.data.preprocessing import preprocess_pipeline
        import yaml
        with open("config/config.yaml") as f:
            cfg = yaml.safe_load(f)
        *_, X_test, _, _, y_test, feature_names = preprocess_pipeline(
            cfg["data"]["raw_path"], {**cfg["data"], **cfg["preprocessing"]}
        )
        print(f"Test set extracted: {len(X_test):,} rows")
    except Exception as e:
        print(f"Could not load test data: {e}")
        sys.exit(0)

# Ensure numpy arrays
if hasattr(X_test, "values"):
    X_test_arr = X_test.values
else:
    X_test_arr = X_test

if hasattr(y_test, "values"):
    y_test_arr = y_test.values
else:
    y_test_arr = y_test

# %%
# ── 1. PR Curves (all models) ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 7))

COLOURS = {
    "XGBoost":       "#6c63ff",
    "Random Forest": "#ff6b6b",
    "Decision Tree": "#ffd93d",
}

pr_auc_scores = {}
for name, model in models_available.items():
    probs    = model.predict_proba(X_test_arr)[:, 1]
    pr_auc   = average_precision_score(y_test_arr, probs)
    prec, rec, _ = precision_recall_curve(y_test_arr, probs)
    pr_auc_scores[name] = pr_auc
    ax.plot(rec, prec, label=f"{name}  (PR-AUC={pr_auc:.4f})",
            color=COLOURS.get(name, "steelblue"), linewidth=2.5)

# Baseline (random classifier)
baseline = y_test_arr.mean()
ax.axhline(y=baseline, color="gray", linestyle="--", linewidth=1.5,
           label=f"Random baseline ({baseline:.4f})")

ax.set_xlabel("Recall",    fontsize=12)
ax.set_ylabel("Precision", fontsize=12)
ax.set_title("Precision-Recall Curves — All Models", fontsize=14, fontweight="bold")
ax.legend(fontsize=11)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.02)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(FIGURE_DIR / "05_pr_curves_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"✅ Saved → {FIGURE_DIR}/05_pr_curves_comparison.png")

# %%
# ── 2. ROC Curves ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 7))

for name, model in models_available.items():
    probs    = model.predict_proba(X_test_arr)[:, 1]
    roc_auc  = roc_auc_score(y_test_arr, probs)
    fpr, tpr, _ = roc_curve(y_test_arr, probs)
    ax.plot(fpr, tpr, label=f"{name}  (ROC-AUC={roc_auc:.4f})",
            color=COLOURS.get(name, "steelblue"), linewidth=2.5)

ax.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Random (0.50)")
ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate",  fontsize=12)
ax.set_title("ROC Curves — All Models", fontsize=14, fontweight="bold")
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(FIGURE_DIR / "06_roc_curves_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"✅ Saved → {FIGURE_DIR}/06_roc_curves_comparison.png")

# %%
# ── 3. Threshold sensitivity (XGBoost) ────────────────────────────────────────
primary_name  = "XGBoost" if "XGBoost" in models_available else list(models_available.keys())[0]
primary_model = models_available[primary_name]
probs         = primary_model.predict_proba(X_test_arr)[:, 1]

thresholds = np.linspace(0.05, 0.90, 100)
precisions, recalls, f1s = [], [], []
for t in thresholds:
    preds = (probs >= t).astype(int)
    precisions.append(precision_score(y_test_arr, preds, zero_division=0))
    recalls.append(   recall_score(   y_test_arr, preds, zero_division=0))
    f1s.append(       f1_score(       y_test_arr, preds, zero_division=0))

best_f1_idx   = int(np.argmax(f1s))
best_threshold = thresholds[best_f1_idx]

fig, ax = plt.subplots(figsize=(11, 6))
ax.plot(thresholds, precisions, label="Precision", color="#4CAF50", linewidth=2.5)
ax.plot(thresholds, recalls,    label="Recall",    color="#F44336", linewidth=2.5)
ax.plot(thresholds, f1s,        label="F1 Score",  color="#6c63ff", linewidth=2.5)

ax.axvline(x=0.40, color="orange", linestyle="--", linewidth=2,
           label="Default threshold (0.40)")
ax.axvline(x=best_threshold, color="purple", linestyle=":", linewidth=2,
           label=f"Best F1 threshold ({best_threshold:.2f})")

ax.set_xlabel("Decision Threshold", fontsize=12)
ax.set_ylabel("Score",              fontsize=12)
ax.set_title(f"Threshold Sensitivity — {primary_name}", fontsize=14, fontweight="bold")
ax.legend(fontsize=11)
ax.set_xlim(0.05, 0.90)
ax.set_ylim(0, 1.02)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(FIGURE_DIR / "07_threshold_sensitivity.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"✅ Saved → {FIGURE_DIR}/07_threshold_sensitivity.png")
print(f"\nThreshold analysis ({primary_name}):")
print(f"  At threshold=0.40:  Precision={precisions[np.argmin(abs(thresholds-0.40))]:.4f}  "
      f"Recall={recalls[np.argmin(abs(thresholds-0.40))]:.4f}")
print(f"  Best F1 threshold:  {best_threshold:.3f}  F1={f1s[best_f1_idx]:.4f}")

# %%
# ── 4. Metric comparison table ────────────────────────────────────────────────
print("\n── Model Comparison (threshold=0.40) ────────────────────────────")
rows = []
for name, model in models_available.items():
    probs_m = model.predict_proba(X_test_arr)[:, 1]
    preds_m = (probs_m >= 0.40).astype(int)
    rows.append({
        "Model":     name,
        "PR-AUC":    round(average_precision_score(y_test_arr, probs_m), 4),
        "ROC-AUC":   round(roc_auc_score(y_test_arr, probs_m), 4),
        "Precision": round(precision_score(y_test_arr, preds_m, zero_division=0), 4),
        "Recall":    round(recall_score(y_test_arr, preds_m, zero_division=0), 4),
        "F1":        round(f1_score(y_test_arr, preds_m, zero_division=0), 4),
    })

comparison_df = pd.DataFrame(rows).sort_values("PR-AUC", ascending=False)
print(comparison_df.to_string(index=False))

# Save results
results_path = Path("reports/model_results.csv")
results_path.parent.mkdir(parents=True, exist_ok=True)
comparison_df.to_csv(results_path, index=False)
print(f"\n✅ Results saved → {results_path}")

# %%
print(f"\nAll figures saved to: {FIGURE_DIR}/")
print("View experiments: make mlflow")

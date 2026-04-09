"""
src/models/evaluate_model.py
=============================
Comprehensive model evaluation and visualisation.

Why Recall is the primary metric
---------------------------------
A false negative (missed fraud) means real money stolen.
A false positive (blocking a legitimate transaction) costs a customer review.
The asymmetric cost means we optimise for high Recall while keeping Precision
acceptable.  ROC-AUC is reported but Precision-Recall AUC (PR-AUC) is the
headline metric because it is not inflated by the massive true-negative count.

Outputs
-------
reports/figures/confusion_matrix_<model>.png
reports/figures/roc_curves.png
reports/figures/pr_curves.png
reports/figures/feature_importance_<model>.png
reports/figures/class_distribution.png
reports/figures/correlation_heatmap.png
reports/figures/model_comparison.png
reports/model_results.csv
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)

PALETTE = ["#2980b9", "#27ae60", "#e74c3c", "#8e44ad", "#f39c12"]
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False})


# ── Metrics ────────────────────────────────────────────────────────────────────


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
    model_name: str = "",
) -> dict[str, float]:
    """
    Compute the full evaluation suite for one model.

    Returns a flat dict suitable for building a comparison DataFrame.

    Parameters
    ----------
    y_true:
        Ground-truth binary labels.
    y_pred:
        Hard predictions at decision threshold.
    y_prob:
        Predicted fraud probabilities (required for AUC metrics).
    model_name:
        String label used in the output dict.

    Returns
    -------
    dict
        Keys: Model, Accuracy, Precision, Recall, F1, FNR, AUC_ROC, PR_AUC.
    """
    metrics: dict[str, Any] = {
        "Model": model_name,
        "Accuracy": round(accuracy_score(y_true, y_pred), 6),
        "Precision": round(precision_score(y_true, y_pred, zero_division=0), 6),
        "Recall": round(recall_score(y_true, y_pred, zero_division=0), 6),
        "F1": round(f1_score(y_true, y_pred, zero_division=0), 6),
        "FNR": round(1 - recall_score(y_true, y_pred, zero_division=0), 6),
    }
    if y_prob is not None:
        metrics["AUC_ROC"] = round(roc_auc_score(y_true, y_prob), 6)
        metrics["PR_AUC"] = round(average_precision_score(y_true, y_prob), 6)

    logger.info(
        "[%s] Acc=%.4f  Prec=%.4f  Recall=%.4f  F1=%.4f  PR_AUC=%s",
        model_name,
        metrics["Accuracy"],
        metrics["Precision"],
        metrics["Recall"],
        metrics["F1"],
        f"{metrics.get('PR_AUC', 'N/A'):.4f}" if "PR_AUC" in metrics else "N/A",
    )
    print(classification_report(y_true, y_pred, target_names=["Legit", "Fraud"], digits=4))
    return metrics


def evaluate_all(
    models: dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> pd.DataFrame:
    """
    Evaluate every model and return a sorted comparison DataFrame.

    Sorted by PR_AUC descending — the metric that matters most for fraud.
    """
    rows = []
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        rows.append(compute_metrics(y_test, y_pred, y_prob, model_name=name))

    df = pd.DataFrame(rows).sort_values("PR_AUC", ascending=False).reset_index(drop=True)
    print("\n" + "=" * 72)
    print("  MODEL COMPARISON  (sorted by PR-AUC — primary fraud metric)")
    print("=" * 72)
    print(df.to_string(index=False))
    return df


# ── Visualisations ─────────────────────────────────────────────────────────────


def plot_class_distribution(
    y: pd.Series | np.ndarray,
    output_dir: str = "reports/figures",
) -> None:
    """Bar chart of fraud vs legitimate class counts."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    counts = pd.Series(y).value_counts().sort_index()
    labels = ["Legitimate (0)", "Fraud (1)"]
    colors = ["#2980b9", "#e74c3c"]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, counts.values, color=colors, edgecolor="white", linewidth=0.8, width=0.5)
    for bar, count in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + counts.max() * 0.01,
            f"{count:,}\n({count / counts.sum() * 100:.3f}%)",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
        )
    ax.set_title("Class Distribution — Fraud vs Legitimate", fontsize=13, fontweight="bold")
    ax.set_ylabel("Transaction Count")
    ax.set_ylim(0, counts.max() * 1.18)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = Path(output_dir) / "class_distribution.png"
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info("Saved → %s", path)


def plot_correlation_heatmap(
    X: pd.DataFrame,
    output_dir: str = "reports/figures",
    n_features: int = 20,
) -> None:
    """Triangular correlation heatmap for the top N highest-variance features."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    top_cols = X.var().nlargest(n_features).index.tolist()
    corr = X[top_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(14, 11))
    sns.heatmap(
        corr, mask=mask, cmap="coolwarm", center=0,
        linewidths=0.4, annot=False,
        cbar_kws={"shrink": 0.8}, ax=ax,
    )
    ax.set_title(
        f"Feature Correlation Heatmap (top {n_features} by variance)",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    path = Path(output_dir) / "correlation_heatmap.png"
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info("Saved → %s", path)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    output_dir: str = "reports/figures",
) -> None:
    """Annotated confusion matrix with TP/TN/FP/FN counts."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Legitimate", "Fraud"],
        yticklabels=["Legitimate", "Fraud"],
        linewidths=0.5, ax=ax,
    )
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=13, fontweight="bold")
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")
    plt.tight_layout()
    safe = model_name.replace(" ", "_").lower()
    path = Path(output_dir) / f"confusion_matrix_{safe}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info("Saved → %s", path)


def plot_roc_curves(
    models: dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
    output_dir: str = "reports/figures",
) -> None:
    """Overlay ROC curves for all models on a single axes."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 7))
    for (name, model), color in zip(models.items(), PALETTE):
        if not hasattr(model, "predict_proba"):
            continue
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc = roc_auc_score(y_test, y_prob)
        ax.plot(fpr, tpr, lw=2, color=color, label=f"{name}  (AUC={roc:.4f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random classifier")
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("ROC Curves — All Models", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = Path(output_dir) / "roc_curves.png"
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info("Saved → %s", path)


def plot_precision_recall_curves(
    models: dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
    output_dir: str = "reports/figures",
) -> None:
    """
    Precision-Recall curves — the correct primary chart for imbalanced data.

    A random classifier sits at Precision = fraud_rate (~0.17%), not at 0.5.
    All serious imbalanced-class papers report this chart, not just ROC.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    fraud_rate = y_test.mean()
    fig, ax = plt.subplots(figsize=(9, 7))

    for (name, model), color in zip(models.items(), PALETTE):
        if not hasattr(model, "predict_proba"):
            continue
        y_prob = model.predict_proba(X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        pr = average_precision_score(y_test, y_prob)
        ax.plot(recall, precision, lw=2, color=color, label=f"{name}  (PR-AUC={pr:.4f})")

    ax.axhline(
        y=fraud_rate, linestyle="--", color="gray", lw=1,
        label=f"Random baseline (fraud rate={fraud_rate:.4f})",
    )
    ax.set_xlabel("Recall", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_title(
        "Precision-Recall Curves — All Models\n"
        "(Primary metric for imbalanced fraud detection)",
        fontsize=12, fontweight="bold",
    )
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = Path(output_dir) / "pr_curves.png"
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info("Saved → %s", path)


def plot_model_comparison(
    df_results: pd.DataFrame,
    output_dir: str = "reports/figures",
) -> None:
    """Grouped bar chart comparing key metrics across models."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    metrics = [m for m in ["Precision", "Recall", "F1", "AUC_ROC", "PR_AUC"] if m in df_results.columns]
    x = np.arange(len(df_results))
    width = 0.15
    fig, ax = plt.subplots(figsize=(13, 6))
    for i, metric in enumerate(metrics):
        offset = (i - len(metrics) / 2) * width + width / 2
        ax.bar(
            x + offset, df_results[metric], width,
            label=metric, color=PALETTE[i % len(PALETTE)],
            edgecolor="white", linewidth=0.5, alpha=0.9,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(df_results["Model"], fontsize=10)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = Path(output_dir) / "model_comparison.png"
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info("Saved → %s", path)


def plot_feature_importance(
    model: Any,
    feature_names: list[str],
    model_name: str,
    output_dir: str = "reports/figures",
    top_n: int = 20,
) -> None:
    """Horizontal bar chart of top-N feature importances."""
    if not hasattr(model, "feature_importances_"):
        return
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    importances = model.feature_importances_
    idx = np.argsort(importances)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(
        range(top_n), importances[idx][::-1],
        color="#2980b9", edgecolor="white", linewidth=0.5,
    )
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_names[i] for i in idx][::-1], fontsize=9)
    ax.set_xlabel("Importance (mean decrease in impurity / gain)")
    ax.set_title(f"Top-{top_n} Feature Importances — {model_name}", fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    safe = model_name.replace(" ", "_").lower()
    path = Path(output_dir) / f"feature_importance_{safe}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    logger.info("Saved → %s", path)


def generate_full_report(
    models: dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list[str],
    output_dir: str = "reports/figures",
) -> pd.DataFrame:
    """
    Master evaluation runner — metrics + all plots for every model.

    Returns
    -------
    pd.DataFrame
        Comparison table sorted by PR_AUC.
    """
    df_results = evaluate_all(models, X_test, y_test)

    for name, model in models.items():
        y_pred = model.predict(X_test)
        plot_confusion_matrix(y_test, y_pred, name, output_dir)
        plot_feature_importance(model, feature_names, name, output_dir)

    plot_roc_curves(models, X_test, y_test, output_dir)
    plot_precision_recall_curves(models, X_test, y_test, output_dir)
    plot_model_comparison(df_results, output_dir)

    return df_results


# ── Cost-sensitive evaluation ─────────────────────────────────────────────────
# In a real fraud system the goal is not to maximise F1 but to maximise
# financial recovery minus operational cost.  These numbers are illustrative
# but grounded in published fraud operations research.

# Cost assumptions (USD)
COST_FALSE_NEGATIVE = 80.0   # average fraud amount not caught (industry ~$80–120)
COST_FALSE_POSITIVE = 5.0    # customer review / call-centre cost per false block
COST_TRUE_POSITIVE  = 2.0    # investigation cost for confirmed fraud (manual review)


def compute_business_impact(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    amounts: np.ndarray | None = None,
    cost_fn: float = COST_FALSE_NEGATIVE,
    cost_fp: float = COST_FALSE_POSITIVE,
    cost_tp: float = COST_TRUE_POSITIVE,
) -> dict[str, float]:
    """
    Translate model errors into estimated financial cost/benefit.

    Parameters
    ----------
    y_true:   Ground-truth labels.
    y_pred:   Model predictions at chosen threshold.
    amounts:  Per-transaction amounts.  If None, uses *cost_fn* as a flat penalty.
    cost_fn:  Cost of a missed fraud (false negative) — default $80.
    cost_fp:  Cost of a false alert (false positive) — default $5.
    cost_tp:  Cost of reviewing confirmed fraud — default $2.

    Returns
    -------
    dict with estimated_loss_caught, estimated_loss_missed, net_benefit,
    total_review_cost, and roi.
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    if amounts is not None:
        fraud_amounts   = amounts[y_true == 1]
        caught_amount   = float(np.sum(fraud_amounts[y_pred[y_true == 1] == 1]))
        missed_amount   = float(np.sum(fraud_amounts[y_pred[y_true == 1] == 0]))
    else:
        caught_amount   = float(tp) * cost_fn
        missed_amount   = float(fn) * cost_fn

    review_cost     = float(tp) * cost_tp + float(fp) * cost_fp
    net_benefit     = caught_amount - review_cost

    roi = (net_benefit / max(review_cost, 1.0)) * 100  # percent

    result = {
        "true_positives":        int(tp),
        "false_positives":       int(fp),
        "false_negatives":       int(fn),
        "true_negatives":        int(tn),
        "estimated_loss_caught": round(caught_amount, 2),
        "estimated_loss_missed": round(missed_amount, 2),
        "total_review_cost":     round(review_cost, 2),
        "net_benefit":           round(net_benefit, 2),
        "roi_pct":               round(roi, 1),
    }

    logger.info(
        "Business impact | Caught=$%,.0f  Missed=$%,.0f  "
        "Review cost=$%,.0f  Net benefit=$%,.0f  ROI=%.1f%%",
        caught_amount, missed_amount, review_cost, net_benefit, roi,
    )
    return result

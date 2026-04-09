"""
src/monitoring/explainability.py
=================================
SHAP-based model explainability.

Why explainability matters for fraud detection
------------------------------------------------
1. Regulatory: GDPR Article 22 requires explainable automated decisions.
2. Operational: fraud investigators need to understand *why* a transaction
   was flagged to decide whether to block it.
3. Debugging: SHAP reveals if the model is learning spurious correlations.

Two explanation modes
---------------------
Global  — SHAP summary + beeswarm plots showing which features drive fraud
          across the entire test set.  Used for model reporting.

Local   — Waterfall / force plot for a single transaction.  Used in the
          API and dashboard to explain individual predictions.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import shap

    SHAP_OK = True
except ImportError:
    SHAP_OK = False
    logger.warning("shap not installed.  Run: pip install shap")

try:
    import matplotlib
    import matplotlib.pyplot as plt

    MPL_OK = True
except ImportError:
    MPL_OK = False


def compute_shap_values(
    model: Any,
    X: pd.DataFrame | np.ndarray,
    model_type: str = "tree",
) -> tuple["shap.Explainer", np.ndarray]:
    """
    Compute SHAP values for a fitted tree-based model.

    Parameters
    ----------
    model:
        Fitted XGBoost or RandomForest estimator.
    X:
        Feature matrix (typically X_test or a representative sample).
        Using more than ~5000 rows will be slow for TreeExplainer.
    model_type:
        ``"tree"`` (default) for XGBoost / RandomForest.

    Returns
    -------
    tuple[explainer, shap_values]
        ``shap_values`` shape: (n_samples, n_features) — fraud class values.
    """
    if not SHAP_OK:
        raise ImportError("Install shap: pip install shap")

    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X)

    # For binary classifiers: shap_values[1] is the fraud class
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]

    logger.info(
        "SHAP values computed | shape=%s | base_value=%.4f",
        shap_vals.shape,
        explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray))
        else explainer.expected_value,
    )
    return explainer, shap_vals


def plot_shap_summary(
    model: Any,
    X: pd.DataFrame,
    output_dir: str = "reports/figures",
    max_display: int = 20,
) -> None:
    """
    Generate a SHAP beeswarm summary plot showing global feature importance.

    Each dot is one transaction.  Colour = feature value (red = high, blue = low).
    Position on X axis = SHAP value (impact on fraud probability).

    Parameters
    ----------
    model:
        Fitted tree model.
    X:
        Test or validation DataFrame.
    output_dir:
        Where to save the PNG.
    max_display:
        Number of top features to display.
    """
    if not SHAP_OK or not MPL_OK:
        logger.warning("Skipping SHAP summary — shap or matplotlib not installed")
        return

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    _, shap_vals = compute_shap_values(model, X)

    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_vals, X, max_display=max_display, show=False)
    plt.title("SHAP Feature Importance (Fraud Class)", fontweight="bold", pad=14)
    plt.tight_layout()
    path = Path(output_dir) / "shap_summary.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("SHAP summary saved → %s", path)


def plot_shap_bar(
    model: Any,
    X: pd.DataFrame,
    output_dir: str = "reports/figures",
    max_display: int = 20,
) -> None:
    """Bar chart of mean absolute SHAP values (global feature importance)."""
    if not SHAP_OK or not MPL_OK:
        return

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    _, shap_vals = compute_shap_values(model, X)
    mean_abs = np.abs(shap_vals).mean(axis=0)
    idx = np.argsort(mean_abs)[::-1][:max_display]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(
        range(max_display),
        mean_abs[idx][::-1],
        color="#2980b9", edgecolor="white",
    )
    ax.set_yticks(range(max_display))
    ax.set_yticklabels([X.columns[i] for i in idx][::-1], fontsize=9)
    ax.set_xlabel("Mean |SHAP value| (average impact on fraud probability)")
    ax.set_title("SHAP Global Feature Importance", fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    path = Path(output_dir) / "shap_bar.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("SHAP bar chart saved → %s", path)


def explain_single_prediction(
    model: Any,
    transaction_df: pd.DataFrame,
    feature_names: list[str],
    top_n: int = 10,
) -> dict[str, Any]:
    """
    Compute per-feature SHAP contributions for a single transaction.

    This is what the API returns alongside the fraud probability to give
    investigators an explanation of *why* this transaction was flagged.

    Parameters
    ----------
    model:
        Fitted tree model.
    transaction_df:
        Single-row DataFrame with feature columns.
    feature_names:
        Ordered list of feature names.
    top_n:
        Number of top contributing features to return.

    Returns
    -------
    dict with keys:
        base_value           — model's expected output (prior)
        prediction_value     — model's output for this transaction
        top_features         — list of (feature_name, shap_value) sorted by |impact|
        explanation_text     — human-readable summary
    """
    if not SHAP_OK:
        return {
            "error": "shap not installed",
            "top_features": [],
            "base_value": None,
            "prediction_value": None,
            "explanation_text": "SHAP not available.",
        }

    explainer, shap_vals = compute_shap_values(model, transaction_df)
    contributions = dict(zip(feature_names, shap_vals[0].tolist()))

    top_features = sorted(
        contributions.items(),
        key=lambda x: abs(x[1]),
        reverse=True,
    )[:top_n]

    base_val = (
        float(explainer.expected_value[1])
        if isinstance(explainer.expected_value, (list, np.ndarray))
        else float(explainer.expected_value)
    )
    pred_val = base_val + shap_vals[0].sum()

    # Build a short text summary for the dashboard
    fraud_drivers = [(f, v) for f, v in top_features if v > 0]
    legit_drivers = [(f, v) for f, v in top_features if v < 0]
    lines = ["Top fraud signals: " + ", ".join(f"{f}(+{v:.3f})" for f, v in fraud_drivers[:3])]
    if legit_drivers:
        lines.append("Top legitimate signals: " + ", ".join(f"{f}({v:.3f})" for f, v in legit_drivers[:3]))

    return {
        "base_value": round(base_val, 6),
        "prediction_value": round(pred_val, 6),
        "top_features": [{"feature": f, "shap_value": round(v, 6)} for f, v in top_features],
        "explanation_text": "\n".join(lines),
    }

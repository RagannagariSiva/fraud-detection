"""
scripts/evaluate.py
====================
Standalone Model Evaluator

Loads a trained model from disk and evaluates it against fresh data without
running the full training pipeline.  Useful for:

  - Verifying a model before promoting it to production
  - Running evaluation on a new test batch that arrived after training
  - Comparing model versions offline before a Champion/Challenger test

Usage
-----
    # Evaluate using the test partition from the last training run:
    python scripts/evaluate.py

    # Evaluate against a custom CSV:
    python scripts/evaluate.py --data path/to/test_data.csv

    # Evaluate a specific model:
    python scripts/evaluate.py --model random_forest_model

    # Include business impact estimate:
    python scripts/evaluate.py --business-impact
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Project root on path so src.* imports work when called directly
sys.path.insert(0, str(Path(__file__).parent.parent))


def _load_test_data(
    data_path: str | None,
    config_path: str = "config/config.yaml",
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Load test data from a CSV path or re-run preprocessing on the original
    raw dataset to recover the same test split used during training.
    """
    import yaml
    import joblib

    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    feat_path = cfg["preprocessing"].get("feature_names_path", "models/feature_names.pkl")
    feature_names: list[str] = joblib.load(feat_path)

    if data_path:
        df = pd.read_csv(data_path)
        if "Class" not in df.columns:
            raise ValueError("CSV must have a 'Class' column for evaluation.")
        X = df[feature_names].values
        y = df["Class"].values
        logger.info("Loaded custom test data: %d rows from %s", len(df), data_path)
        return X, y, feature_names

    # Re-derive the test split with the same random state as training
    from src.data.preprocessing import preprocess_pipeline
    data_cfg = cfg["data"]
    prep_cfg = {**data_cfg, **cfg["preprocessing"]}
    _, _, X_test, _, _, y_test, _ = preprocess_pipeline(
        raw_path=data_cfg["raw_path"],
        cfg=prep_cfg,
    )
    from src.features.feature_engineering import build_features
    X_test_eng = build_features(X_test, cfg.get("features", {}))
    logger.info(
        "Re-derived test split: %d rows, %d features",
        len(X_test_eng), X_test_eng.shape[1],
    )
    return X_test_eng.values, y_test.values, list(X_test_eng.columns)


def evaluate(
    model_name: str = "xgboost_model",
    data_path: str | None = None,
    model_dir: str = "models",
    output_path: str | None = None,
    business_impact: bool = False,
    threshold: float = 0.40,
) -> dict:
    """
    Load a trained model and evaluate it on test data.

    Returns a dict with all computed metrics and, optionally, business impact.
    """
    import joblib
    from sklearn.metrics import (
        average_precision_score, roc_auc_score,
        precision_score, recall_score, f1_score,
        classification_report,
    )

    # ── Load model ────────────────────────────────────────────────────────────
    model_path = Path(model_dir) / f"{model_name}.pkl"
    if not model_path.exists():
        logger.error("Model not found: %s", model_path)
        logger.error("Run 'python main.py' to train first.")
        sys.exit(1)

    model = joblib.load(model_path)
    logger.info("Loaded model: %s", model_path)

    # ── Load test data ─────────────────────────────────────────────────────────
    X_test, y_test, feature_names = _load_test_data(data_path)

    # ── Score ─────────────────────────────────────────────────────────────────
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= threshold).astype(int)

    metrics = {
        "model":     model_name,
        "threshold": threshold,
        "n_samples": int(len(y_test)),
        "n_fraud":   int(y_test.sum()),
        "fraud_rate": float(y_test.mean()),
        "pr_auc":    round(float(average_precision_score(y_test, probs)), 6),
        "roc_auc":   round(float(roc_auc_score(y_test, probs)), 6),
        "precision": round(float(precision_score(y_test, preds, zero_division=0)), 6),
        "recall":    round(float(recall_score(y_test, preds, zero_division=0)), 6),
        "f1":        round(float(f1_score(y_test, preds, zero_division=0)), 6),
        "fnr":       round(float(1 - recall_score(y_test, preds, zero_division=0)), 6),
    }

    # ── Print results ─────────────────────────────────────────────────────────
    bar = "═" * 60
    print(f"\n{bar}")
    print(f"  EVALUATION: {model_name}  (threshold={threshold:.2f})")
    print(bar)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:<20} {v:.6f}")
        else:
            print(f"  {k:<20} {v}")

    print(f"\n{classification_report(y_test, preds, target_names=['Legitimate', 'Fraud'], digits=4)}")

    # ── Business impact ───────────────────────────────────────────────────────
    if business_impact:
        from src.models.evaluate_model import compute_business_impact
        impact = compute_business_impact(y_test, preds)
        metrics["business_impact"] = impact

        print(f"\n  BUSINESS IMPACT ESTIMATE")
        print(f"  {'─' * 40}")
        print(f"  Fraud caught:     ${impact['estimated_loss_caught']:>10,.2f}")
        print(f"  Fraud missed:     ${impact['estimated_loss_missed']:>10,.2f}")
        print(f"  Review cost:      ${impact['total_review_cost']:>10,.2f}")
        print(f"  Net benefit:      ${impact['net_benefit']:>10,.2f}")
        print(f"  ROI:              {impact['roi_pct']:>10.1f}%")

    print(bar + "\n")

    # ── Save results ──────────────────────────────────────────────────────────
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        logger.info("Evaluation results saved → %s", output_path)

    return metrics


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate a trained FraudGuard model on test data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model",    default="xgboost_model",
                   help="Model name (stem of .pkl file in models/)")
    p.add_argument("--data",     default=None,
                   help="Path to CSV with Class column (default: re-derive test split)")
    p.add_argument("--output",   default="reports/evaluation_results.json",
                   help="Path to save JSON results")
    p.add_argument("--threshold", type=float, default=0.40,
                   help="Decision threshold for hard predictions")
    p.add_argument("--business-impact", action="store_true",
                   help="Include estimated financial impact in output")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    evaluate(
        model_name      =args.model,
        data_path       =args.data,
        output_path     =args.output,
        threshold       =args.threshold,
        business_impact =args.business_impact,
    )

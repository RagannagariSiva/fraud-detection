"""
src/training/pipeline.py
=========================
Full offline training pipeline orchestrator.

This module owns the end-to-end training workflow. Each step is a discrete,
named phase so failed runs produce useful logs. The pipeline is intentionally
not a class — it is a pure function that takes a config dict and returns a
results DataFrame. This makes it trivial to call from scripts, notebooks,
or CI jobs without side effects from object state.

Pipeline phases
---------------
1.  Preprocess     — clean, scale (RobustScaler), stratified split
2.  EDA plots      — class distribution, correlation heatmap
3.  Feature eng.   — time features, interaction terms
4.  Resampling     — SMOTE on training set *only*
5.  Tuning         — optional Optuna hyperparameter search
6.  Training       — XGBoost + Random Forest + Decision Tree baseline
7.  Drift baseline — save feature statistics for production monitoring
8.  Evaluation     — ROC/PR curves, confusion matrices, comparison table
9.  Threshold      — Youden's J optimal threshold on validation set
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


# ── Logging setup (called from main.py before anything else) ──────────────────

def setup_logging(log_path: str = "logs/training.log") -> None:
    """Configure root logger to write to both stdout and a rotating log file."""
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s  %(levelname)-8s  %(name)s — %(message)s"

    # Prevent adding duplicate handlers on re-import
    root = logging.getLogger()
    if root.handlers:
        return

    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.FileHandler(log_path, mode="a", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def load_config(path: str = "config/config.yaml") -> dict[str, Any]:
    """
    Load and validate the YAML configuration file.

    Raises
    ------
    FileNotFoundError
        When the config file does not exist at *path*.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p.resolve()}")
    with open(p, encoding="utf-8") as f:
        cfg: dict[str, Any] = yaml.safe_load(f)
    logger.info("Config loaded from %s", p)
    return cfg


# ── Main pipeline ──────────────────────────────────────────────────────────────

def run_pipeline(cfg: dict[str, Any]) -> pd.DataFrame:
    """
    Execute the complete training pipeline and return a model comparison table.

    All intermediate artifacts (scaler, feature names, drift baseline, model
    pkl files, evaluation plots) are written to disk as side effects. The
    return value is a DataFrame with one row per trained model and columns for
    every evaluation metric.

    Parameters
    ----------
    cfg:
        Fully-loaded config dict from :func:`load_config`.

    Returns
    -------
    pd.DataFrame
        Model comparison table (columns: Model, PR-AUC, ROC-AUC, F1, …).
    """
    from src.data.preprocessing import preprocess_pipeline
    from src.features.feature_engineering import build_features
    from src.features.resampling import resample
    from src.models.evaluate_model import (
        generate_full_report,
        plot_class_distribution,
        plot_correlation_heatmap,
    )
    from src.training.train_model import train_all_models

    # Unpack config sections with clear local names
    data_cfg   = cfg["data"]
    prep_cfg   = {**data_cfg, **cfg["preprocessing"]}
    feat_cfg   = cfg.get("features", {})
    resamp_cfg = {**cfg.get("resampling", {}), "random_state": data_cfg["random_state"]}
    model_cfg  = cfg.get("models", {})
    train_cfg  = cfg["training"]
    mlflow_cfg = cfg.get("mlflow", {})
    drift_cfg  = cfg.get("drift_detection", {})

    model_dir  = train_cfg["model_dir"]
    report_dir = train_cfg["report_dir"]
    Path(report_dir).mkdir(parents=True, exist_ok=True)
    Path(model_dir).mkdir(parents=True, exist_ok=True)

    # ── Phase 1: Preprocess ───────────────────────────────────────────────────
    _phase("1 — PREPROCESSING")
    X_train, X_val, X_test, y_train, y_val, y_test, feature_names = preprocess_pipeline(
        raw_path=data_cfg["raw_path"],
        cfg=prep_cfg,
    )

    # ── Phase 2: EDA plots ────────────────────────────────────────────────────
    _phase("2 — EDA PLOTS")
    plot_class_distribution(y_train, report_dir)
    plot_correlation_heatmap(X_train, report_dir)

    # ── Phase 3: Feature engineering ─────────────────────────────────────────
    _phase("3 — FEATURE ENGINEERING")
    X_train = build_features(X_train, feat_cfg)
    X_val   = build_features(X_val,   feat_cfg)
    X_test  = build_features(X_test,  feat_cfg)

    # Re-save feature names now that engineering has added columns
    feature_names = list(X_train.columns)
    feat_names_path = prep_cfg.get("feature_names_path", "models/feature_names.pkl")
    joblib.dump(feature_names, feat_names_path)
    logger.info("Feature list updated: %d features → %s", len(feature_names), feat_names_path)

    # ── Phase 4: Resampling (training set only) ───────────────────────────────
    _phase("4 — RESAMPLING  [train set only]")
    X_train_res, y_train_res = resample(
        X_train, y_train,
        strategy=resamp_cfg.get("strategy", "smote"),
        cfg=resamp_cfg,
    )

    # ── Phase 5: Optional hyperparameter tuning ───────────────────────────────
    tune_cfg = cfg.get("tuning", {})
    if tune_cfg.get("enabled", False):
        _phase("5 — HYPERPARAMETER TUNING  [Optuna]")
        from src.training.tuning import tune_xgboost
        best_params = tune_xgboost(
            X_train_res, y_train_res,
            n_trials=int(tune_cfg.get("n_trials", 60)),
            cv_folds=int(tune_cfg.get("cv_folds", 5)),
            scoring=tune_cfg.get("scoring", "average_precision"),
            random_state=data_cfg["random_state"],
        )
        model_cfg["xgboost"] = best_params
        logger.info("Tuned XGBoost params: %s", best_params)
    else:
        logger.info("Hyperparameter tuning skipped  (set tuning.enabled: true to run)")

    # ── Phase 6: Training ─────────────────────────────────────────────────────
    _phase("6 — MODEL TRAINING")
    models = train_all_models(
        X_train_res, y_train_res,
        X_val.values, y_val.values,
        feature_names=feature_names,
        cfg=model_cfg,
        model_dir=model_dir,
        mlflow_enabled=mlflow_cfg.get("enabled", True),
        experiment_name=mlflow_cfg.get("experiment_name", "fraud-detection"),
    )

    # ── Phase 7: Save drift detection baseline ────────────────────────────────
    _phase("7 — DRIFT BASELINE")
    _save_drift_baseline(X_train, drift_cfg)

    # ── Phase 8: Evaluation ───────────────────────────────────────────────────
    _phase("8 — EVALUATION")
    df_results = generate_full_report(
        models,
        X_test.values, y_test.values,
        feature_names, report_dir,
    )

    results_path = Path(train_cfg["results_path"])
    results_path.parent.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(results_path, index=False)
    logger.info("Model comparison table → %s", results_path)

    # ── Phase 9: Decision threshold tuning ───────────────────────────────────
    _phase("9 — THRESHOLD OPTIMISATION  [Youden's J on validation set]")
    _tune_threshold(train_cfg, model_dir, X_val.values, y_val.values)

    # ── Phase 10: SHAP explainability plots ───────────────────────────────────
    _phase("10 — SHAP EXPLAINABILITY")
    primary_key = "XGBoost" if train_cfg.get("primary_model", "xgboost") == "xgboost" else "Random Forest"
    _run_shap_analysis(models, X_test, report_dir, primary_model_key=primary_key)

    # ── Phase 11: Business impact estimate ────────────────────────────────────
    _phase("11 — BUSINESS IMPACT ESTIMATE")
    _log_business_impact(models, X_test.values, y_test.values, primary_model_key=primary_key)

    _phase("PIPELINE COMPLETE ✓")
    return df_results


# ── Private helpers ────────────────────────────────────────────────────────────

def _phase(label: str) -> None:
    """Print a clearly visible phase separator to the log."""
    bar = "─" * 60
    logger.info("\n%s\n  %s\n%s", bar, label, bar)


def _save_drift_baseline(X_train: pd.DataFrame, drift_cfg: dict[str, Any]) -> None:
    """
    Compute and persist feature distribution statistics for the training set.

    Called immediately after training so the saved baseline reflects exactly
    what the model was trained on. The :class:`~src.monitoring.drift_detector.DriftDetector`
    loads this file in production to detect covariate shift in incoming data.
    """
    try:
        from src.monitoring.drift_detector import DriftDetector
        baseline_path = drift_cfg.get("baseline_path", "models/drift_baseline.json")
        detector = DriftDetector.from_training_data(X_train)
        detector.save(baseline_path)
        logger.info("Drift baseline saved → %s", baseline_path)
    except Exception as exc:
        # Non-fatal — drift detection is monitoring, not a training requirement
        logger.warning("Could not save drift baseline (non-fatal): %s", exc)


def _tune_threshold(
    train_cfg: dict[str, Any],
    model_dir: str,
    X_val: "np.ndarray",  # noqa: F821
    y_val: "np.ndarray",
) -> None:
    """
    Find the decision threshold that maximises Youden's J statistic on the
    validation set and update the saved predictor.

    Youden's J = Sensitivity + Specificity − 1. This is the point on the ROC
    curve that maximises the distance from the diagonal, giving a natural
    operating point for a fraud system where both recall and specificity matter.
    """
    primary = train_cfg.get("primary_model", "xgboost")
    model_key = "XGBoost" if primary == "xgboost" else "Random Forest"

    try:
        from src.inference.predictor import FraudPredictor
        predictor = FraudPredictor(
            model_name=f"{primary}_model",
            model_dir=model_dir,
            threshold=0.5,
        )
        # Only tune if we have the right model loaded
        best_threshold = predictor.set_threshold_youden(X_val, y_val)
        logger.info(
            "Threshold tuned (%s): %.4f  → saved to predictor",
            model_key, best_threshold,
        )
    except FileNotFoundError:
        logger.warning("Could not load %s for threshold tuning.", model_key)
    except Exception as exc:
        logger.warning("Threshold tuning failed (non-fatal): %s", exc)


def _run_shap_analysis(
    models: dict[str, Any],
    X_test: pd.DataFrame,
    report_dir: str,
    primary_model_key: str = "XGBoost",
) -> None:
    """
    Generate SHAP summary and bar plots for the primary model.

    Called after evaluation — adds global explainability artifacts to
    ``reports/figures/`` so the dashboard can display them immediately.
    Non-fatal: if shap is not installed, logs a warning and continues.
    """
    model = models.get(primary_model_key)
    if model is None:
        return

    try:
        from src.monitoring.explainability import plot_shap_bar, plot_shap_summary

        # Use a sample for speed (TreeExplainer is O(n) but still slow on 50k rows)
        sample = X_test if len(X_test) <= 2000 else X_test.sample(2000, random_state=42)
        logger.info("Running SHAP analysis on %d samples...", len(sample))
        plot_shap_summary(model, sample, output_dir=report_dir)
        plot_shap_bar(model, sample, output_dir=report_dir)
        logger.info("SHAP plots saved to %s", report_dir)
    except Exception as exc:
        logger.warning("SHAP analysis skipped (non-fatal): %s", exc)


def _log_business_impact(
    models: dict[str, Any],
    X_test: "np.ndarray",
    y_test: "np.ndarray",
    primary_model_key: str = "XGBoost",
) -> None:
    """
    Compute and log the estimated financial impact of the primary model's
    predictions on the test set.

    Uses the cost assumptions defined in ``src/models/evaluate_model`` —
    $80 average fraud loss per missed transaction, $5 per false alert.
    """
    import numpy as np
    from src.models.evaluate_model import compute_business_impact

    model = models.get(primary_model_key)
    if model is None:
        return

    try:
        threshold = 0.40  # default; will reflect Youden-tuned value after Phase 9
        probs     = model.predict_proba(X_test)[:, 1]
        preds     = (probs >= threshold).astype(int)
        impact    = compute_business_impact(y_test, preds)

        logger.info(
            "\n%s\n  BUSINESS IMPACT ESTIMATE  (test set, threshold=%.2f)\n%s\n"
            "  Fraud caught:    $%,.0f\n"
            "  Fraud missed:    $%,.0f\n"
            "  Review cost:     $%,.0f\n"
            "  Net benefit:     $%,.0f\n"
            "  ROI:             %.1f%%\n%s",
            "─" * 56, threshold, "─" * 56,
            impact["estimated_loss_caught"],
            impact["estimated_loss_missed"],
            impact["total_review_cost"],
            impact["net_benefit"],
            impact["roi_pct"],
            "─" * 56,
        )
    except Exception as exc:
        logger.warning("Business impact calculation skipped: %s", exc)

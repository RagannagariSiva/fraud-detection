"""
src/training/train_model.py
============================
Model Training with MLflow Experiment Tracking

Trains Random Forest and XGBoost classifiers, saves each model to disk,
and logs every experiment to MLflow for full reproducibility.

What gets tracked in MLflow
----------------------------
Parameters   — all hyperparameters (n_estimators, learning_rate, etc.)
Metrics      — val PR-AUC, val ROC-AUC, precision, recall, F1, training time
Artifacts    — model .pkl, confusion matrix PNG, feature importance PNG,
               classification report JSON, metadata JSON
Tags         — model type, dataset info, Python / library versions

Run MLflow UI
-------------
    mlflow ui --port 5000
    # Then open http://localhost:5000

Quick start without MLflow
---------------------------
    Set MLFLOW_ENABLED=false in environment or pass mlflow_enabled=False
    to train_all_models(). All training still works; just no tracking.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")  # headless backend — no display required
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.tree import DecisionTreeClassifier

logger = logging.getLogger(__name__)

# ── Optional MLflow import ─────────────────────────────────────────────────────

try:
    import mlflow
    import mlflow.sklearn

    MLFLOW_OK = True
except ImportError:
    MLFLOW_OK = False
    logger.warning(
        "MLflow not installed — experiment tracking disabled. Install with: pip install mlflow"
    )

# ── Optional XGBoost import ────────────────────────────────────────────────────

try:
    from xgboost import XGBClassifier

    XGB_OK = True
except ImportError:
    XGB_OK = False
    logger.warning("XGBoost not installed. Run: pip install xgboost")


# ══════════════════════════════════════════════════════════════════════════════
#  MLflow context manager
# ══════════════════════════════════════════════════════════════════════════════


class _MlflowRun:
    """
    Context manager that wraps an MLflow active run.

    If MLflow is not available or MLFLOW_ENABLED=false, all log_* calls
    become no-ops so training code can be written once for both cases.
    """

    def __init__(
        self,
        experiment_name: str = "fraud-detection",
        run_name: str | None = None,
        tracking_uri: str | None = None,
        enabled: bool = True,
    ):
        self._enabled = enabled and MLFLOW_OK
        self._experiment = experiment_name
        self._run_name = run_name
        self._uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", "mlruns")
        self._run = None

    # ── Context protocol ──────────────────────────────────────────────────────

    def __enter__(self) -> _MlflowRun:
        if not self._enabled:
            return self
        mlflow.set_tracking_uri(self._uri)
        mlflow.set_experiment(self._experiment)
        self._run = mlflow.start_run(run_name=self._run_name)
        logger.info(
            "MLflow run started | experiment=%s | run_id=%s",
            self._experiment,
            self._run.info.run_id,
        )
        return self

    def __exit__(self, *_) -> None:
        if self._enabled and self._run is not None:
            mlflow.end_run()
            logger.info("MLflow run ended | run_id=%s", self._run.info.run_id)

    # ── Logging helpers ───────────────────────────────────────────────────────

    def log_params(self, params: dict) -> None:
        if not self._enabled:
            return
        # MLflow requires string values for params
        str_params = {k: str(v) for k, v in params.items()}
        mlflow.log_params(str_params)

    def log_metrics(self, metrics: dict) -> None:
        if not self._enabled:
            return
        mlflow.log_metrics({k: float(v) for k, v in metrics.items() if v is not None})

    def log_tags(self, tags: dict) -> None:
        if not self._enabled:
            return
        mlflow.set_tags({k: str(v) for k, v in tags.items()})

    def log_model(self, model: Any, artifact_path: str) -> None:
        if not self._enabled:
            return
        mlflow.sklearn.log_model(model, artifact_path)

    def log_artifact(self, local_path: str) -> None:
        if not self._enabled:
            return
        mlflow.log_artifact(local_path)

    def log_dict_as_artifact(self, data: dict, filename: str) -> None:
        """Write dict to a temp JSON file and log it as an artifact."""
        if not self._enabled:
            return
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, prefix=filename
        ) as f:
            json.dump(data, f, indent=2)
            tmp = f.name
        mlflow.log_artifact(tmp, artifact_path="reports")
        os.unlink(tmp)

    @property
    def run_id(self) -> str | None:
        return self._run.info.run_id if self._run else None


# ══════════════════════════════════════════════════════════════════════════════
#  Evaluation helpers
# ══════════════════════════════════════════════════════════════════════════════


def _evaluate(
    model: Any,
    X_val: np.ndarray,
    y_val: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """
    Run full evaluation on the validation set.

    Returns a flat dict of metrics ready to be logged to MLflow.
    """
    y_prob = model.predict_proba(X_val)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    pr_auc = average_precision_score(y_val, y_prob)
    roc_auc = roc_auc_score(y_val, y_prob)
    prec = precision_score(y_val, y_pred, zero_division=0)
    rec = recall_score(y_val, y_pred, zero_division=0)
    f1 = f1_score(y_val, y_pred, zero_division=0)

    return {
        "val_pr_auc": round(pr_auc, 6),
        "val_roc_auc": round(roc_auc, 6),
        "val_precision": round(prec, 6),
        "val_recall": round(rec, 6),
        "val_f1": round(f1, 6),
    }


def _plot_confusion_matrix(
    model: Any,
    X_val: np.ndarray,
    y_val: np.ndarray,
    model_name: str,
    output_dir: str,
    threshold: float = 0.5,
) -> str:
    """Plot and save a confusion matrix PNG. Returns the file path."""
    y_prob = model.predict_proba(X_val)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_val, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set(
        title=f"Confusion Matrix — {model_name}",
        xlabel="Predicted label",
        ylabel="True label",
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["Legitimate", "Fraud"],
        yticklabels=["Legitimate", "Fraud"],
    )
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=13, fontweight="bold")

    plt.tight_layout()
    path = Path(output_dir) / f"{model_name.lower().replace(' ', '_')}_confusion_matrix.png"
    plt.savefig(path, dpi=120)
    plt.close(fig)
    return str(path)


def _plot_feature_importance(
    model: Any,
    feature_names: list[str],
    model_name: str,
    output_dir: str,
    top_n: int = 20,
) -> str | None:
    """Plot and save a feature importance bar chart PNG. Returns path or None."""
    importances = None
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        return None

    indices = np.argsort(importances)[-top_n:][::-1]
    top_names = [feature_names[i] for i in indices]
    top_values = importances[indices]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(top_names)), top_values[::-1], color="steelblue")
    ax.set_yticks(range(len(top_names)))
    ax.set_yticklabels(top_names[::-1])
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {top_n} Feature Importances — {model_name}")
    plt.tight_layout()
    path = Path(output_dir) / f"{model_name.lower().replace(' ', '_')}_feature_importance.png"
    plt.savefig(path, dpi=120)
    plt.close(fig)
    return str(path)


# ══════════════════════════════════════════════════════════════════════════════
#  Individual trainers
# ══════════════════════════════════════════════════════════════════════════════


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: list[str] | None = None,
    cfg: dict | None = None,
    model_dir: str = "models",
    mlflow_enabled: bool = True,
    experiment_name: str = "fraud-detection",
) -> RandomForestClassifier:
    """
    Train a Random Forest classifier with MLflow experiment tracking.

    Uses ``class_weight="balanced"`` to handle imbalance without resampling.

    Parameters
    ----------
    X_train, y_train : SMOTE-resampled training data.
    X_val, y_val     : Held-out validation data.
    feature_names    : List of feature column names (for importance plot).
    cfg              : ``models.random_forest`` section from config.
    model_dir        : Directory to save model artifacts.
    mlflow_enabled   : Set False to skip MLflow logging.
    experiment_name  : MLflow experiment name.
    """
    cfg = cfg or {}
    params = {
        "n_estimators": int(cfg.get("n_estimators", 300)),
        "max_depth": cfg.get("max_depth", None),
        "min_samples_leaf": int(cfg.get("min_samples_leaf", 2)),
        "class_weight": cfg.get("class_weight", "balanced"),
        "n_jobs": int(cfg.get("n_jobs", -1)),
        "random_state": int(cfg.get("random_state", 42)),
    }
    feature_names = feature_names or [f"feature_{i}" for i in range(X_train.shape[1])]

    with _MlflowRun(
        experiment_name=experiment_name,
        run_name="random_forest",
        enabled=mlflow_enabled,
    ) as run:
        run.log_tags(
            {
                "model_type": "RandomForestClassifier",
                "framework": "sklearn",
                "training_rows": len(X_train),
                "validation_rows": len(X_val),
                "n_features": X_train.shape[1],
                "python_version": sys.version.split()[0],
            }
        )
        run.log_params(params)

        logger.info("Training RandomForest — %d trees ...", params["n_estimators"])
        t0 = time.perf_counter()
        clf = RandomForestClassifier(**params)
        clf.fit(X_train, y_train)
        elapsed = time.perf_counter() - t0

        metrics = _evaluate(clf, X_val, y_val)
        metrics["train_time_seconds"] = round(elapsed, 2)
        metrics["train_size"] = len(X_train)
        run.log_metrics(metrics)

        logger.info(
            "RandomForest | PR-AUC=%.4f  ROC-AUC=%.4f  F1=%.4f  time=%.1fs",
            metrics["val_pr_auc"],
            metrics["val_roc_auc"],
            metrics["val_f1"],
            elapsed,
        )

        # Save + log artifacts
        _save_model(clf, "random_forest_model", model_dir, params, metrics)
        Path(model_dir).mkdir(parents=True, exist_ok=True)

        cm_path = _plot_confusion_matrix(clf, X_val, y_val, "Random Forest", model_dir)
        run.log_artifact(cm_path)

        fi_path = _plot_feature_importance(clf, feature_names, "Random Forest", model_dir)
        if fi_path:
            run.log_artifact(fi_path)

        report = classification_report(
            y_val,
            (clf.predict_proba(X_val)[:, 1] >= 0.5).astype(int),
            output_dict=True,
        )
        run.log_dict_as_artifact(report, "classification_report_rf")
        run.log_model(clf, "random_forest")

    return clf


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: list[str] | None = None,
    cfg: dict | None = None,
    model_dir: str = "models",
    mlflow_enabled: bool = True,
    experiment_name: str = "fraud-detection",
) -> XGBClassifier:
    """
    Train an XGBoost classifier with full MLflow experiment tracking.

    Key design choices
    ------------------
    * ``scale_pos_weight`` computed from training label distribution.
    * ``eval_metric="aucpr"`` focuses boosting on the minority class.
    * All hyperparameters, metrics, plots, and the model itself are logged.

    Parameters
    ----------
    X_train, y_train : Resampled training data.
    X_val, y_val     : Validation data (not resampled).
    feature_names    : Column names for importance plot.
    cfg              : ``models.xgboost`` section from config.
    model_dir        : Directory to save artifacts.
    mlflow_enabled   : Set False to skip tracking.
    experiment_name  : MLflow experiment name.
    """
    if not XGB_OK:
        raise ImportError("XGBoost is required.  Run: pip install xgboost")

    cfg = cfg or {}
    feature_names = feature_names or [f"feature_{i}" for i in range(X_train.shape[1])]

    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    scale_pos = round(neg / max(pos, 1), 4)

    params: dict[str, Any] = {
        "n_estimators": int(cfg.get("n_estimators", 400)),
        "learning_rate": float(cfg.get("learning_rate", 0.05)),
        "max_depth": int(cfg.get("max_depth", 6)),
        "subsample": float(cfg.get("subsample", 0.8)),
        "colsample_bytree": float(cfg.get("colsample_bytree", 0.8)),
        "min_child_weight": int(cfg.get("min_child_weight", 3)),
        "scale_pos_weight": scale_pos,
        "eval_metric": cfg.get("eval_metric", "aucpr"),
        "n_jobs": int(cfg.get("n_jobs", -1)),
        "verbosity": int(cfg.get("verbosity", 0)),
        "random_state": int(cfg.get("random_state", 42)),
    }

    with _MlflowRun(
        experiment_name=experiment_name,
        run_name="xgboost",
        enabled=mlflow_enabled,
    ) as run:
        run.log_tags(
            {
                "model_type": "XGBClassifier",
                "framework": "xgboost",
                "training_rows": len(X_train),
                "validation_rows": len(X_val),
                "n_features": X_train.shape[1],
                "imbalance_ratio": f"{scale_pos:.1f}",
                "python_version": sys.version.split()[0],
            }
        )
        run.log_params(params)

        logger.info(
            "Training XGBoost — %d rounds  lr=%.3f  scale_pos_weight=%.1f ...",
            params["n_estimators"],
            params["learning_rate"],
            params["scale_pos_weight"],
        )
        t0 = time.perf_counter()
        clf = XGBClassifier(**params)
        clf.fit(X_train, y_train)
        elapsed = time.perf_counter() - t0

        metrics = _evaluate(clf, X_val, y_val)
        metrics["train_time_seconds"] = round(elapsed, 2)
        metrics["train_size"] = len(X_train)
        metrics["scale_pos_weight"] = scale_pos
        run.log_metrics(metrics)

        logger.info(
            "XGBoost | PR-AUC=%.4f  ROC-AUC=%.4f  F1=%.4f  time=%.1fs",
            metrics["val_pr_auc"],
            metrics["val_roc_auc"],
            metrics["val_f1"],
            elapsed,
        )

        # Artifacts
        _save_model(clf, "xgboost_model", model_dir, params, metrics)
        Path(model_dir).mkdir(parents=True, exist_ok=True)

        cm_path = _plot_confusion_matrix(clf, X_val, y_val, "XGBoost", model_dir)
        run.log_artifact(cm_path)

        fi_path = _plot_feature_importance(clf, feature_names, "XGBoost", model_dir)
        if fi_path:
            run.log_artifact(fi_path)

        report = classification_report(
            y_val,
            (clf.predict_proba(X_val)[:, 1] >= 0.5).astype(int),
            output_dict=True,
        )
        run.log_dict_as_artifact(report, "classification_report_xgb")
        run.log_model(clf, "xgboost")

    return clf


def train_decision_tree_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray | None = None,
    y_val: np.ndarray | None = None,
    cfg: dict | None = None,
    model_dir: str = "models",
    mlflow_enabled: bool = True,
    experiment_name: str = "fraud-detection",
) -> DecisionTreeClassifier:
    """
    Train a shallow Decision Tree as a fast interpretable baseline.

    Included for benchmarking — not intended for production.
    Logged to MLflow as the 'baseline' run so comparisons are easy.
    """
    cfg = cfg or {}
    params = {
        "max_depth": int(cfg.get("max_depth", 8)),
        "class_weight": "balanced",
        "random_state": int(cfg.get("random_state", 42)),
    }

    with _MlflowRun(
        experiment_name=experiment_name,
        run_name="decision_tree_baseline",
        enabled=mlflow_enabled,
    ) as run:
        run.log_tags({"model_type": "DecisionTreeClassifier", "role": "baseline"})
        run.log_params(params)

        logger.info("Training Decision Tree baseline — max_depth=%d", params["max_depth"])
        t0 = time.perf_counter()
        clf = DecisionTreeClassifier(**params)
        clf.fit(X_train, y_train)
        elapsed = time.perf_counter() - t0

        metrics: dict = {"train_time_seconds": round(elapsed, 2)}
        if X_val is not None and y_val is not None:
            metrics.update(_evaluate(clf, X_val, y_val))
            logger.info(
                "DecisionTree | PR-AUC=%.4f  F1=%.4f  time=%.1fs",
                metrics.get("val_pr_auc", 0.0),
                metrics.get("val_f1", 0.0),
                elapsed,
            )
        run.log_metrics(metrics)
        _save_model(clf, "decision_tree_model", model_dir, params, metrics)

    return clf


# ══════════════════════════════════════════════════════════════════════════════
#  Train-all orchestrator
# ══════════════════════════════════════════════════════════════════════════════


def train_all_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: list[str] | None = None,
    cfg: dict | None = None,
    model_dir: str = "models",
    mlflow_enabled: bool = True,
    experiment_name: str = "fraud-detection",
) -> dict[str, Any]:
    """
    Train all models, log experiments, and return a name → estimator dict.

    Parameters
    ----------
    X_train, y_train   : Resampled training data.
    X_val, y_val       : Validation data.
    feature_names      : Column names (optional; used for importance plots).
    cfg                : ``models`` section from config.yaml.
    model_dir          : Output directory for saved models.
    mlflow_enabled     : Master switch for MLflow tracking.
    experiment_name    : MLflow experiment name.

    Returns
    -------
    dict[str, estimator]
        Keys: "Random Forest", "XGBoost", "Decision Tree Baseline".
    """
    cfg = cfg or {}
    enabled = mlflow_enabled and MLFLOW_OK

    if enabled:
        logger.info(
            "MLflow tracking enabled | experiment='%s' | uri=%s",
            experiment_name,
            os.getenv("MLFLOW_TRACKING_URI", "mlruns"),
        )
        logger.info("View experiments: mlflow ui --port 5000")
    else:
        logger.info("MLflow tracking disabled (set mlflow_enabled=True to enable).")

    models: dict[str, Any] = {}

    rf = train_random_forest(
        X_train,
        y_train,
        X_val,
        y_val,
        feature_names=feature_names,
        cfg=cfg.get("random_forest", {}),
        model_dir=model_dir,
        mlflow_enabled=enabled,
        experiment_name=experiment_name,
    )
    models["Random Forest"] = rf

    if XGB_OK:
        xgb = train_xgboost(
            X_train,
            y_train,
            X_val,
            y_val,
            feature_names=feature_names,
            cfg=cfg.get("xgboost", {}),
            model_dir=model_dir,
            mlflow_enabled=enabled,
            experiment_name=experiment_name,
        )
        models["XGBoost"] = xgb

    dt = train_decision_tree_baseline(
        X_train,
        y_train,
        X_val,
        y_val,
        cfg=cfg.get("decision_tree", {}),
        model_dir=model_dir,
        mlflow_enabled=enabled,
        experiment_name=experiment_name,
    )
    models["Decision Tree Baseline"] = dt

    logger.info("Trained %d models: %s", len(models), list(models.keys()))
    return models


# ══════════════════════════════════════════════════════════════════════════════
#  Persistence helpers (shared by all trainers)
# ══════════════════════════════════════════════════════════════════════════════


def _save_model(
    model: Any,
    name: str,
    model_dir: str,
    params: dict,
    metrics: dict,
) -> None:
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    pkl_path = Path(model_dir) / f"{name}.pkl"
    meta_path = Path(model_dir) / f"{name}_metadata.json"

    joblib.dump(model, pkl_path)

    meta = {
        "name": name,
        "class": type(model).__name__,
        "params": {k: str(v) for k, v in params.items()},
        **{k: v for k, v in metrics.items() if v is not None},
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    logger.info("Saved %s → %s", name, pkl_path)


def load_model(name: str, model_dir: str = "models") -> Any:
    """Load a previously saved model by name (stem of pkl file)."""
    path = Path(model_dir) / f"{name}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Model not found at {path}.  Run main.py to train first.")
    model = joblib.load(path)
    logger.info("Loaded model ← %s", path)
    return model

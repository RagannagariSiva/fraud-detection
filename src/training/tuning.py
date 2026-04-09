"""
src/training/tuning.py
=======================
Optuna-based hyperparameter optimisation for XGBoost.

Why Optuna over GridSearchCV
-----------------------------
GridSearchCV exhaustively tries every combination — with 5 params each having
5 values that is 5^5 = 3125 trials.  Optuna uses Tree-structured Parzen
Estimators (TPE) to intelligently sample the space and converge in 50-100
trials, which is 10-30x fewer evaluations for equivalent results.

Objective
----------
Maximise ``average_precision`` (PR-AUC) with stratified 5-fold CV.
PR-AUC is the correct objective for an imbalanced fraud dataset — it directly
measures how well the model finds fraud, not how well it identifies the
overwhelming majority of legitimate transactions.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

try:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_OK = True
except ImportError:
    OPTUNA_OK = False
    logger.warning("optuna not installed.  Run: pip install optuna")

try:
    from xgboost import XGBClassifier

    XGB_OK = True
except ImportError:
    XGB_OK = False


def tune_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_trials: int = 60,
    cv_folds: int = 5,
    scoring: str = "average_precision",
    random_state: int = 42,
) -> dict:
    """
    Find the best XGBoost hyperparameters using Optuna TPE search.

    Parameters
    ----------
    X_train, y_train:
        Resampled training data.
    n_trials:
        Number of Optuna trials (50-100 is usually sufficient).
    cv_folds:
        Number of stratified CV folds for each trial.
    scoring:
        Sklearn scoring string.  Use ``"average_precision"`` for PR-AUC.
    random_state:
        Seed for reproducibility.

    Returns
    -------
    dict
        Best hyperparameter dict ready to pass to XGBClassifier(**best_params).
    """
    if not OPTUNA_OK:
        raise ImportError("Install optuna: pip install optuna")
    if not XGB_OK:
        raise ImportError("Install xgboost: pip install xgboost")

    from sklearn.model_selection import StratifiedKFold, cross_val_score

    scale_pos = float((y_train == 0).sum()) / max(float((y_train == 1).sum()), 1)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    def objective(trial: "optuna.Trial") -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 10.0, log=False),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
            "scale_pos_weight": scale_pos,
            "eval_metric": "aucpr",
            "verbosity": 0,
            "random_state": random_state,
            "n_jobs": -1,
        }
        clf = XGBClassifier(**params)
        scores = cross_val_score(clf, X_train, y_train, scoring=scoring, cv=cv, n_jobs=1)
        return float(scores.mean())

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=random_state),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
    )
    logger.info("Starting Optuna search — %d trials", n_trials)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_params
    best["scale_pos_weight"] = scale_pos
    best["eval_metric"] = "aucpr"
    best["verbosity"] = 0
    best["random_state"] = random_state
    best["n_jobs"] = -1

    logger.info(
        "Best PR-AUC=%.4f  params=%s",
        study.best_value, best,
    )
    return best

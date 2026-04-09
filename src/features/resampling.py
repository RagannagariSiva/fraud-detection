"""
src/features/resampling.py
===========================
Class-imbalance handling strategies.

The Kaggle creditcard dataset has ~578 legitimate transactions for every
fraud case.  Without correction, classifiers will optimise for the majority
class and miss most fraud.

Available strategies
--------------------
smote       — Synthetic Minority Over-sampling Technique (default)
adasyn      — Adaptive Synthetic Sampling (focuses on hard cases)
undersample — Random majority-class undersampling (fast but lossy)
none        — Pass through without resampling

Rule: SMOTE must ONLY be applied to the training set.
      Applying it before the train/test split leaks synthetic samples into
      the test set, inflating all evaluation metrics.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    from imblearn.over_sampling import ADASYN, SMOTE
    from imblearn.under_sampling import RandomUnderSampler

    IMBLEARN_OK = True
except ImportError:
    IMBLEARN_OK = False
    logger.warning(
        "imbalanced-learn not installed — SMOTE/ADASYN unavailable. "
        "Run: pip install imbalanced-learn"
    )


def resample(
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
    strategy: str = "smote",
    cfg: dict | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply the requested imbalance-correction strategy to the training set.

    Parameters
    ----------
    X_train:
        Feature matrix (training only — never include test data).
    y_train:
        Binary label vector.
    strategy:
        One of ``"smote"``, ``"adasyn"``, ``"undersample"``, ``"none"``.
    cfg:
        Optional config dict with keys ``smote_k_neighbors`` and ``random_state``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Resampled (X, y) as numpy arrays.
    """
    cfg = cfg or {}
    rs = int(cfg.get("random_state", 42))

    dispatch = {
        "smote": _smote,
        "adasyn": _adasyn,
        "undersample": _undersample,
        "none": _passthrough,
    }
    fn = dispatch.get(strategy.lower(), _passthrough)
    return fn(X_train, y_train, rs, cfg)


# ── Strategy implementations ───────────────────────────────────────────────────


def _smote(X, y, rs, cfg):
    if not IMBLEARN_OK:
        logger.warning("Skipping SMOTE — imbalanced-learn missing")
        return _to_numpy(X, y)
    k = int(cfg.get("smote_k_neighbors", 5))
    sm = SMOTE(k_neighbors=k, random_state=rs)
    X_r, y_r = sm.fit_resample(X, y)
    _log_resample("SMOTE", y, y_r)
    return X_r, y_r


def _adasyn(X, y, rs, cfg):
    if not IMBLEARN_OK:
        logger.warning("Skipping ADASYN — imbalanced-learn missing")
        return _to_numpy(X, y)
    ad = ADASYN(random_state=rs)
    X_r, y_r = ad.fit_resample(X, y)
    _log_resample("ADASYN", y, y_r)
    return X_r, y_r


def _undersample(X, y, rs, cfg):
    if not IMBLEARN_OK:
        logger.warning("Skipping under-sampling — imbalanced-learn missing")
        return _to_numpy(X, y)
    rus = RandomUnderSampler(random_state=rs)
    X_r, y_r = rus.fit_resample(X, y)
    _log_resample("Under-sample", y, y_r)
    return X_r, y_r


def _passthrough(X, y, rs, cfg):
    logger.info("No resampling applied")
    return _to_numpy(X, y)


# ── Utilities ──────────────────────────────────────────────────────────────────


def _to_numpy(X, y):
    X_arr = X.values if hasattr(X, "values") else np.asarray(X)
    y_arr = y.values if hasattr(y, "values") else np.asarray(y)
    return X_arr, y_arr


def _log_resample(name: str, y_before, y_after):
    n_before = len(y_before)
    n_after = len(y_after)
    fraud_after = int(y_after.sum()) if hasattr(y_after, "sum") else int(np.sum(y_after))
    logger.info(
        "%s: %s → %s rows  |  fraud after: %d (%.1f%%)",
        name,
        f"{n_before:,}",
        f"{n_after:,}",
        fraud_after,
        fraud_after / n_after * 100,
    )

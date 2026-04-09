"""
src/data/preprocessing.py
==========================
Data cleaning, scaling, and train/val/test splitting.

Design decisions
----------------
The scaler is fitted ONLY on the training set and persisted to disk so
inference can apply exactly the same transformation. This prevents the
most common production bug in ML systems: training-serving skew, where the
model sees different feature distributions at inference than it was trained on.

The val split is kept separate from the test split throughout the pipeline.
Hyperparameter tuning and threshold calibration use the val set; the test
set is touched exactly once for final reporting.

Artifacts written
-----------------
models/scaler.pkl           Fitted RobustScaler
models/feature_names.pkl    Column-ordered feature list
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler

logger = logging.getLogger(__name__)

# Columns that receive scaling treatment
_SCALE_COLS: tuple[str, ...] = ("Amount", "Time")


# ── Public helpers (used directly in tests and notebooks) ──────────────────────


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply defensive cleaning to the raw dataframe.

    - Drop exact duplicate rows (the Kaggle dataset has none, but a production
      ingestion pipeline might create them from retry logic).
    - Fill any remaining nulls with column medians rather than dropping rows
      — losing fraud rows is more costly than a small imputation error.
    - Ensure the Class column is an integer label, not a float.
    """
    original_len = len(df)
    df = df.drop_duplicates()
    if (dropped := original_len - len(df)):
        logger.info("Dropped %d duplicate rows", dropped)

    null_counts = df.isnull().sum()
    if null_counts.any():
        for col in null_counts[null_counts > 0].index:
            fill_val = df[col].median()
            df = df.copy()  # ensure we own the frame before mutation
            df.loc[:, col] = df[col].fillna(fill_val)
            logger.info("Imputed %d nulls in '%s' with median %.4f",
                        null_counts[col], col, fill_val)

    if "Class" in df.columns:
        df["Class"] = df["Class"].astype(int)
    return df


def fit_scale_save(
    X: pd.DataFrame,
    scale_cols: list[str],
    scaler_type: str,
    scaler_path: str,
) -> tuple[pd.DataFrame, Any]:
    """
    Fit a scaler on the specified columns, transform in place, and persist.

    Call this on the TRAINING set only. Then use the returned scaler (or
    transform_single()) to apply the same transform at val/test/inference time.

    We use RobustScaler by default because Amount contains extreme outliers
    (fraudulent transactions can be very large or very small). RobustScaler
    uses the IQR instead of the standard deviation, which is more resistant
    to those extremes than StandardScaler.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (training split only — do not include test or val).
    scale_cols : list[str]
        Column names to scale.
    scaler_type : str
        "robust" or "standard".
    scaler_path : str
        Where to save the fitted scaler pkl.

    Returns
    -------
    tuple[pd.DataFrame, scaler]
        Scaled copy of X and the fitted scaler object.
    """
    available = [c for c in scale_cols if c in X.columns]
    if not available:
        logger.warning("No scale columns found in dataframe. Skipping scaling.")
        return X.copy(), None

    scaler_cls = {"robust": RobustScaler, "standard": StandardScaler}.get(
        scaler_type.lower(), RobustScaler
    )
    scaler = scaler_cls()

    X_out = X.copy()
    X_out[available] = scaler.fit_transform(X_out[available])

    Path(scaler_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, scaler_path)
    logger.info(
        "Fitted %s on %s — saved to %s",
        scaler.__class__.__name__,
        available,
        scaler_path,
    )
    return X_out, scaler


def transform_single(
    transaction: dict[str, float],
    scaler_path: str = "models/scaler.pkl",
    scale_cols: tuple[str, ...] = _SCALE_COLS,
) -> dict[str, float]:
    """
    Apply the saved scaler to a single transaction dict (for inference).

    Loads the scaler fitted during training and applies it to the Amount/Time
    values in the transaction. Other features pass through unchanged.

    Parameters
    ----------
    transaction : dict
        Feature name -> raw value mapping.
    scaler_path : str
        Path to the saved scaler pkl.
    scale_cols : tuple
        Columns the scaler was fitted on (must match training order).

    Returns
    -------
    dict with Amount/Time replaced by their scaled values.
    """
    scaler = joblib.load(scaler_path)
    t = dict(transaction)

    # Build a full-width array preserving training column order.
    # Missing cols get 0.0 (won't matter — we only write back present cols).
    full_raw = np.zeros((1, len(scale_cols)))
    for i, col in enumerate(scale_cols):
        if col in t:
            full_raw[0, i] = t[col]

    scaled_full = scaler.transform(full_raw)[0]
    for i, col in enumerate(scale_cols):
        if col in t:
            t[col] = float(scaled_full[i])

    return t


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Split a labelled dataframe into features X and target y.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a 'Class' column (0=legit, 1=fraud).

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        X without the Class column, y as a binary Series.
    """
    X = df.drop(columns=["Class"])
    y = df["Class"]
    return X, y


# ── Full pipeline ──────────────────────────────────────────────────────────────


def preprocess_pipeline(
    raw_path: str,
    cfg: dict[str, Any],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    pd.Series,
    list[str],
]:
    """
    Run the full offline preprocessing pipeline.

    Steps
    -----
    1. Load raw data via the data loader (with synthetic fallback).
    2. Clean: remove duplicates, impute nulls, enforce dtypes.
    3. Three-way stratified split: train / val / test (no leakage).
    4. Fit RobustScaler on training Amount/Time ONLY, save to disk.
    5. Apply the fitted scaler to val and test (transform, not fit).
    6. Save feature name order to disk for deterministic inference.

    Parameters
    ----------
    raw_path : str
        Path to creditcard.csv.
    cfg : dict
        Merged ``data`` and ``preprocessing`` sections from config.yaml.

    Returns
    -------
    X_train, X_val, X_test, y_train, y_val, y_test, feature_names
    """
    from src.data.loader import load_data

    df = load_data(raw_path)
    df = clean(df)

    X, y = split_xy(df)

    test_size = float(cfg.get("test_size", 0.20))
    val_size  = float(cfg.get("val_size", 0.10))
    seed      = int(cfg.get("random_state", 42))

    # First cut: hold out the test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
    # Second cut: carve val from the remaining data.
    # Adjust val fraction so it represents val_size of the original dataset.
    adjusted_val = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=adjusted_val,
        stratify=y_temp,
        random_state=seed,
    )

    # Fit scaler on training set only, then apply to all three splits.
    # This is the critical step that prevents training-serving skew.
    scale_cols  = list(cfg.get("scale_cols", _SCALE_COLS))
    scaler_type = cfg.get("scaler", "robust")
    scaler_path = cfg.get("scaler_path", "models/scaler.pkl")

    X_train, scaler = fit_scale_save(X_train, scale_cols, scaler_type, scaler_path)

    if scaler is not None:
        # Transform val and test with the training scaler — never refit
        available = [c for c in scale_cols if c in X_val.columns]
        if available:
            X_val  = X_val.copy()
            X_test = X_test.copy()
            X_val[available]  = scaler.transform(X_val[available])
            X_test[available] = scaler.transform(X_test[available])

    feature_names = list(X_train.columns)
    feat_path = cfg.get("feature_names_path", "models/feature_names.pkl")
    Path(feat_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(feature_names, feat_path)

    logger.info(
        "Split complete | train=%s  val=%s  test=%s  features=%d",
        f"{len(X_train):,}",
        f"{len(X_val):,}",
        f"{len(X_test):,}",
        len(feature_names),
    )
    logger.info(
        "Fraud rate | train=%.4f%%  val=%.4f%%  test=%.4f%%",
        y_train.mean() * 100,
        y_val.mean() * 100,
        y_test.mean() * 100,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test, feature_names

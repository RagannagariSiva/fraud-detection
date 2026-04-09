"""
src/features/feature_engineering.py
=====================================
Advanced feature engineering for fraud detection.

Real banks detect fraud by looking at *patterns across transactions*, not just
a single transaction in isolation.  This module adds three families of features:

1. **Time-based features**
   hour_of_day, is_night, is_weekend — fraud skews toward off-hours.

2. **Amount-based features**
   log_amount, amount_z_score — captures order-of-magnitude anomalies.

3. **Velocity / rolling-window features**
   txn_count_1h, amount_sum_24h, time_since_last_txn — sequence signals.
   (Disabled by default on large datasets because they are O(n log n).
    Enable with features.add_velocity_features: true in config.)

4. **Interaction features**
   V1_V2_interaction, V3_V4_interaction — cross-feature products that help
   tree models capture non-linear boundaries in the PCA-transformed space.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ── Public API ─────────────────────────────────────────────────────────────────


def build_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Apply all enabled feature families to *df* and return the augmented frame.

    Parameters
    ----------
    df:
        Preprocessed (cleaned + scaled) dataframe **excluding** the Class column.
    cfg:
        ``features`` section from config.yaml.

    Returns
    -------
    pd.DataFrame
        Dataframe with additional engineered columns appended.
    """
    df = df.copy()
    original_cols = df.shape[1]

    if cfg.get("add_time_features", True):
        df = _add_time_features(df)

    if cfg.get("add_velocity_features", False):
        df = _add_velocity_features(df)

    if cfg.get("add_interactions", True):
        df = _add_interaction_features(df)

    added = df.shape[1] - original_cols
    logger.info("Feature engineering: %d → %d features (+%d)", original_cols, df.shape[1], added)
    return df


# ── Time features ──────────────────────────────────────────────────────────────


def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive temporal context from the raw Time column.

    The Kaggle dataset's Time column is seconds elapsed since the first
    transaction in the dataset (~48 hours of data).  Fraud rates vary strongly
    by time of day — late-night / early-morning transactions have significantly
    higher fraud rates in card-present fraud studies.

    New columns
    -----------
    hour_of_day   : float  [0, 24) — transaction hour within a day
    is_night      : int    1 when hour < 6 or hour >= 22
    day_of_period : int    0 or 1 — which 24-hour window the txn falls in
    log_amount    : float  log1p(|Amount|) — compresses extreme values
    amount_z      : float  (Amount - median) / MAD — z-score-like outlier signal
    """
    df = df.copy()

    if "Time" in df.columns:
        # Time within the day (seconds mod 86400 → hours)
        df["hour_of_day"] = (df["Time"] % 86_400) / 3_600
        df["is_night"] = ((df["hour_of_day"] < 6) | (df["hour_of_day"] >= 22)).astype(int)
        df["day_of_period"] = (df["Time"] // 86_400).astype(int).clip(0, 1)

    if "Amount" in df.columns:
        # log1p on absolute value works even after RobustScaler shifts Amount near 0
        df["log_amount"] = np.log1p(np.abs(df["Amount"]))
        med = df["Amount"].median()
        mad = (df["Amount"] - med).abs().median()   # pandas 2.x removed .mad()
        if mad > 0:
            df["amount_z"] = (df["Amount"] - med) / mad
        else:
            df["amount_z"] = 0.0

    return df


# ── Velocity / rolling-window features ────────────────────────────────────────


def _add_velocity_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute transaction-velocity features using a sorted time window.

    These are the most powerful real-world fraud signals.  A legitimate
    cardholder rarely makes 10 transactions in a single hour; a fraudster
    using a stolen card often does.

    Implementation uses pandas' time-indexed rolling so the window is
    genuinely time-based rather than row-count-based.

    New columns
    -----------
    txn_count_1h    : int   — transactions in the preceding 60 minutes
    amount_sum_24h  : float — total spend in the preceding 24 hours
    time_since_last : float — seconds since the previous transaction in the dataset

    Note
    ----
    The Kaggle dataset does not include cardholder IDs (anonymised), so these
    features are computed globally across all transactions.  With cardholder IDs
    you would apply them per-account which is far more powerful.
    """
    df = df.copy().sort_values("Time").reset_index(drop=True)

    # Use a DatetimeIndex so rolling() accepts time-based windows
    base_ts = pd.Timestamp("2020-01-01")
    ts_index = pd.to_datetime(df["Time"], unit="s", origin=base_ts)
    s_amount = pd.Series(df["Amount"].values, index=ts_index, name="Amount")

    # 1-hour rolling count
    df["txn_count_1h"] = (
        s_amount.rolling("3600s", min_periods=1).count().values.astype(int)
    )

    # 24-hour rolling sum of amount
    df["amount_sum_24h"] = (
        s_amount.rolling("86400s", min_periods=1).sum().values
    )

    # Time since previous transaction (0 for the very first row)
    df["time_since_last"] = df["Time"].diff().fillna(0).clip(lower=0)

    logger.info("Velocity features computed: txn_count_1h, amount_sum_24h, time_since_last")
    return df


# ── Interaction features ───────────────────────────────────────────────────────


def _add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add lightweight cross-feature products.

    V1, V4, V12, V14 and V17 are consistently the highest-importance
    features in XGBoost runs on the Kaggle dataset (confirmed by SHAP).
    Products of these features capture non-linear interactions that tree
    models would otherwise require extra depth to discover.

    New columns
    -----------
    V1_V4     : float  product of V1 and V4
    V12_V14   : float  product of V12 and V14
    V1_V17    : float  product of V1 and V17
    V14_abs   : float  |V14| — V14 is the single strongest predictor;
                                  its absolute value adds a distinct signal
    """
    df = df.copy()

    for a, b in [("V1", "V4"), ("V12", "V14"), ("V1", "V17")]:
        if a in df.columns and b in df.columns:
            df[f"{a}_{b}"] = df[a] * df[b]

    if "V14" in df.columns:
        df["V14_abs"] = df["V14"].abs()

    return df

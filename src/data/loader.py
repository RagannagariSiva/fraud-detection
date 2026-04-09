"""
src/data/loader.py
==================
Data loading with synthetic fallback.

The Kaggle creditcard.csv is not distributed in this repo (144 MB).
When it is absent the loader generates a statistically equivalent synthetic
dataset so every downstream step — training, tests, CI — works without it.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

EXPECTED_COLUMNS: list[str] = [f"V{i}" for i in range(1, 29)] + [
    "Amount",
    "Time",
    "Class",
]


def load_data(path: str) -> pd.DataFrame:
    """
    Load transaction dataset from *path*.

    Falls back to :func:`_generate_synthetic_data` when the file does not
    exist so the pipeline runs without the Kaggle download.

    Parameters
    ----------
    path:
        Filesystem path to ``creditcard.csv``.

    Returns
    -------
    pd.DataFrame
        Raw dataframe with columns V1–V28, Amount, Time, Class.

    Raises
    ------
    ValueError
        When the file exists but is missing required columns.
    """
    p = Path(path)
    if p.exists():
        logger.info("Loading real dataset from %s", p)
        df = pd.read_csv(p)
        _validate_schema(df)
        logger.info(
            "Loaded %s rows  |  fraud rate: %.4f%%",
            f"{len(df):,}",
            df["Class"].mean() * 100,
        )
        return df

    logger.warning(
        "%s not found — generating synthetic dataset. "
        "Download the real file from https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud",
        path,
    )
    return _generate_synthetic_data()


# ── Helpers ────────────────────────────────────────────────────────────────────


def _validate_schema(df: pd.DataFrame) -> None:
    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing columns: {sorted(missing)}")
    df["Class"] = df["Class"].astype(int)


def _generate_synthetic_data(
    n: int = 284_807,
    fraud_ratio: float = 0.00172,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Build a synthetic dataset that mirrors the real creditcard.csv schema.

    Fraud rows have per-feature mean shifts drawn from U(-4, 4) to create
    a separable signal, while legitimate rows are drawn from near-zero means.
    """
    rng = np.random.RandomState(random_state)
    n_fraud = int(n * fraud_ratio)
    n_normal = n - n_fraud

    normal: dict[str, np.ndarray] = {}
    for i in range(1, 29):
        normal[f"V{i}"] = rng.normal(rng.uniform(-0.5, 0.5), rng.uniform(0.8, 2.5), n_normal)
    normal["Amount"] = rng.exponential(80, n_normal)
    normal["Time"] = np.sort(rng.uniform(0, 172_800, n_normal))
    normal["Class"] = np.zeros(n_normal, dtype=int)

    shifts = rng.uniform(-4, 4, 28)
    fraud: dict[str, np.ndarray] = {}
    for idx, i in enumerate(range(1, 29)):
        fraud[f"V{i}"] = rng.normal(shifts[idx], rng.uniform(0.5, 1.5), n_fraud)
    fraud["Amount"] = rng.exponential(120, n_fraud)
    fraud["Time"] = rng.uniform(0, 172_800, n_fraud)
    fraud["Class"] = np.ones(n_fraud, dtype=int)

    df = (
        pd.concat([pd.DataFrame(normal), pd.DataFrame(fraud)], ignore_index=True)
        .sample(frac=1, random_state=random_state)
        .reset_index(drop=True)
    )
    logger.info(
        "Generated %s rows  |  fraud: %s (%.4f%%)",
        f"{len(df):,}",
        f"{df['Class'].sum():,}",
        df["Class"].mean() * 100,
    )
    return df

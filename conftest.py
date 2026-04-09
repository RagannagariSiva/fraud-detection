"""
conftest.py
============
Pytest root configuration and shared fixtures.

Running tests
-------------
    pytest tests/ -v                       # all tests
    pytest tests/test_api.py -v            # API only
    pytest tests/ --cov=src --cov=api      # with coverage

All test modules can import from ``src``, ``api``, ``simulation``, and
``monitoring`` without needing to manipulate ``sys.path`` manually, because
pytest adds the project root to ``sys.path`` when it finds this file.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ── Shared synthetic data fixtures ────────────────────────────────────────────


@pytest.fixture(scope="session")
def synthetic_transactions() -> pd.DataFrame:
    """
    A small synthetic transaction DataFrame shaped like the Kaggle creditcard
    dataset. Session-scoped so it's built once and reused across all tests.
    """
    rng = np.random.default_rng(seed=0)
    n = 300
    fraud_idx = rng.choice(n, size=5, replace=False)
    labels = np.zeros(n, dtype=int)
    labels[fraud_idx] = 1

    data = {f"V{i}": rng.normal(0, 1.5, n) for i in range(1, 29)}
    data["Amount"] = rng.exponential(80, n).clip(min=0)
    data["Time"] = np.sort(rng.uniform(0, 172_800, n))
    data["Class"] = labels

    return pd.DataFrame(data)


@pytest.fixture(scope="session")
def feature_matrix(synthetic_transactions) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) split from synthetic_transactions."""
    df = synthetic_transactions
    X = df.drop("Class", axis=1)
    y = df["Class"]
    return X, y


@pytest.fixture(scope="session")
def v_feature_names() -> list[str]:
    """The 30 feature names expected by the model (V1–V28 + Amount + Time)."""
    return [f"V{i}" for i in range(1, 29)] + ["Amount", "Time"]


@pytest.fixture
def valid_transaction_dict() -> dict[str, float]:
    """A single legitimate transaction as a flat feature dict."""
    return {
        "V1": -1.3598, "V2": -0.0728, "V3":  2.5364, "V4":  1.3782,
        "V5": -0.3383, "V6":  0.4624, "V7":  0.2396, "V8":  0.0987,
        "V9":  0.3638, "V10":-0.0902, "V11":-0.5516, "V12":-0.6178,
        "V13":-0.9914, "V14":-0.3114, "V15":  1.4682, "V16":-0.4704,
        "V17": 0.2079, "V18": 0.0258, "V19":  0.4039, "V20":  0.2514,
        "V21":-0.0183, "V22": 0.2778, "V23": -0.1105, "V24":  0.0669,
        "V25": 0.1285, "V26":-0.1891, "V27":  0.1336, "V28": -0.0211,
        "Amount": 149.62,
        "Time": 406.0,
    }

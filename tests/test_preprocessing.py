"""
tests/test_preprocessing.py
============================
Tests for data loading, cleaning, and scaling.

Run: pytest tests/test_preprocessing.py -v
"""

import numpy as np
import pandas as pd
import pytest
import joblib
from pathlib import Path


# ── Fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Minimal dataframe with the expected schema."""
    n = 200
    rng = np.random.RandomState(0)
    data = {f"V{i}": rng.randn(n) for i in range(1, 29)}
    data["Amount"] = rng.exponential(80, n)
    data["Time"] = np.sort(rng.uniform(0, 172_800, n))
    data["Class"] = np.zeros(n, dtype=int)
    data["Class"][:5] = 1  # inject 5 fraud cases
    return pd.DataFrame(data)


# ── Loader tests ───────────────────────────────────────────────────────────────


def test_synthetic_data_schema():
    """Synthetic fallback produces the correct column set."""
    from src.data.loader import _generate_synthetic_data

    df = _generate_synthetic_data(n=1_000)
    expected = set([f"V{i}" for i in range(1, 29)] + ["Amount", "Time", "Class"])
    assert set(df.columns) == expected


def test_synthetic_data_fraud_ratio():
    """Synthetic data approximates the specified fraud ratio."""
    from src.data.loader import _generate_synthetic_data

    df = _generate_synthetic_data(n=50_000, fraud_ratio=0.001)
    actual_ratio = df["Class"].mean()
    assert 0.0005 < actual_ratio < 0.002, f"Fraud ratio {actual_ratio:.5f} out of expected range"


# ── Cleaning tests ─────────────────────────────────────────────────────────────


def test_clean_removes_duplicates(sample_df):
    """clean() drops duplicate rows."""
    from src.data.preprocessing import clean

    df_duped = pd.concat([sample_df, sample_df.iloc[:10]], ignore_index=True)
    cleaned = clean(df_duped)
    assert len(cleaned) == len(sample_df)


def test_clean_fills_nulls(sample_df):
    """clean() imputes NaN with column median."""
    from src.data.preprocessing import clean

    df = sample_df.copy()
    df.loc[0, "Amount"] = np.nan
    cleaned = clean(df)
    assert not cleaned.isnull().any().any()


def test_clean_class_is_int(sample_df):
    """clean() ensures Class column is integer dtype."""
    from src.data.preprocessing import clean

    df = sample_df.copy()
    df["Class"] = df["Class"].astype(float)
    cleaned = clean(df)
    assert cleaned["Class"].dtype == int


# ── Scaling tests ──────────────────────────────────────────────────────────────


def test_fit_scale_save_creates_pkl(sample_df, tmp_path):
    """fit_scale_save() writes a scaler pkl to disk."""
    from src.data.preprocessing import fit_scale_save

    scaler_path = str(tmp_path / "scaler.pkl")
    df_scaled, scaler = fit_scale_save(
        sample_df.drop("Class", axis=1),
        scale_cols=["Amount", "Time"],
        scaler_type="robust",
        scaler_path=scaler_path,
    )
    assert Path(scaler_path).exists(), "Scaler pkl was not created"
    loaded = joblib.load(scaler_path)
    assert loaded is not None


def test_fit_scale_save_transforms_correctly(sample_df, tmp_path):
    """Scaled Amount median should be near 0 after RobustScaler."""
    from src.data.preprocessing import fit_scale_save

    scaler_path = str(tmp_path / "scaler.pkl")
    df_scaled, _ = fit_scale_save(
        sample_df.drop("Class", axis=1),
        scale_cols=["Amount", "Time"],
        scaler_type="robust",
        scaler_path=scaler_path,
    )
    # RobustScaler centres on median — scaled median should be ~0
    assert abs(df_scaled["Amount"].median()) < 0.05


def test_fit_scale_save_does_not_modify_unscaled_cols(sample_df, tmp_path):
    """Columns not in scale_cols should be unchanged after scaling."""
    from src.data.preprocessing import fit_scale_save

    scaler_path = str(tmp_path / "scaler.pkl")
    X = sample_df.drop("Class", axis=1)
    df_scaled, _ = fit_scale_save(
        X,
        scale_cols=["Amount", "Time"],
        scaler_type="robust",
        scaler_path=scaler_path,
    )
    pd.testing.assert_series_equal(X["V1"].reset_index(drop=True),
                                   df_scaled["V1"].reset_index(drop=True))


def test_transform_single_uses_saved_scaler(sample_df, tmp_path):
    """transform_single() produces different values than the raw input."""
    from src.data.preprocessing import fit_scale_save, transform_single

    scaler_path = str(tmp_path / "scaler.pkl")
    fit_scale_save(
        sample_df.drop("Class", axis=1),
        scale_cols=["Amount", "Time"],
        scaler_type="robust",
        scaler_path=scaler_path,
    )
    tx = {f"V{i}": 0.0 for i in range(1, 29)}
    tx["Amount"] = 500.0
    tx["Time"] = 86_400.0

    scaled = transform_single(tx, scaler_path=scaler_path)
    # The scaler centres on the training median, so a raw value of 500
    # should be different from the raw value after scaling.
    assert scaled["Amount"] != 500.0 or scaled["Time"] != 86_400.0


# ── split_xy tests ─────────────────────────────────────────────────────────────


def test_split_xy_correct_shapes(sample_df):
    """split_xy returns X without Class column and y as a Series."""
    from src.data.preprocessing import split_xy

    X, y = split_xy(sample_df)
    assert "Class" not in X.columns
    assert len(X) == len(y)
    assert set(y.unique()).issubset({0, 1})


def test_split_xy_y_is_series(sample_df):
    """split_xy returns y as a pandas Series, not a DataFrame."""
    from src.data.preprocessing import split_xy

    _, y = split_xy(sample_df)
    assert isinstance(y, pd.Series)

"""
tests/test_api.py
==================
Integration tests for the FastAPI endpoints.

Uses FastAPI's TestClient (wraps httpx) so no server needs to be running.
The predictor is replaced with a lightweight mock so tests run without
trained model files.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# ── Fixtures ───────────────────────────────────────────────────────────────────


def _make_mock_predictor():
    mock = MagicMock()
    mock.model_name = "test_model"
    mock.threshold = 0.40
    mock.feature_names = [f"V{i}" for i in range(1, 29)] + ["Amount", "Time"]
    mock.predict.return_value = {
        "prediction": "legitimate",
        "probability": 0.05,
        "is_fraud": False,
        "risk_tier": "LOW",
        "threshold_used": 0.40,
        "message": "Transaction appears normal.",
    }
    return mock


@pytest.fixture
def client():
    """Create a test client with the predictor mocked."""
    mock_predictor = _make_mock_predictor()

    with patch("api.main._predictor", mock_predictor):
        from api.main import app

        with TestClient(app) as c:
            yield c


# ── /health ────────────────────────────────────────────────────────────────────


def test_health_returns_200(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data


# ── /predict ───────────────────────────────────────────────────────────────────


VALID_TRANSACTION = {
    "V1": -1.35,
    "V2": -0.07,
    "V3": 2.54,
    "V4": 1.38,
    "V5": -0.34,
    "V6": 0.46,
    "V7": 0.24,
    "V8": 0.10,
    "V9": 0.36,
    "V10": -0.09,
    "V11": -0.55,
    "V12": -0.62,
    "V13": -0.99,
    "V14": -0.31,
    "V15": 1.47,
    "V16": -0.47,
    "V17": 0.21,
    "V18": 0.03,
    "V19": 0.40,
    "V20": 0.25,
    "V21": -0.02,
    "V22": 0.28,
    "V23": -0.11,
    "V24": 0.07,
    "V25": 0.13,
    "V26": -0.19,
    "V27": 0.13,
    "V28": -0.02,
    "Amount": 149.62,
    "Time": 0.0,
}


def test_predict_valid_transaction(client):
    response = client.post("/predict", json=VALID_TRANSACTION)
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] in ("fraud", "legitimate")
    assert 0.0 <= data["probability"] <= 1.0
    assert data["risk_tier"] in ("LOW", "MEDIUM", "HIGH", "CRITICAL")
    assert "threshold_used" in data
    assert "message" in data


def test_predict_negative_amount_rejected(client):
    bad = {**VALID_TRANSACTION, "Amount": -100.0}
    response = client.post("/predict", json=bad)
    assert response.status_code == 422


def test_predict_missing_v_feature_rejected(client):
    bad = dict(VALID_TRANSACTION)
    del bad["V1"]
    response = client.post("/predict", json=bad)
    assert response.status_code == 422


def test_predict_returns_fraud_for_suspicious(client):
    """When predictor mock returns fraud, API should propagate it."""
    mock = _make_mock_predictor()
    mock.predict.return_value = {
        "prediction": "fraud",
        "probability": 0.95,
        "is_fraud": True,
        "risk_tier": "CRITICAL",
        "threshold_used": 0.40,
        "message": "High fraud probability. Block and alert immediately.",
    }
    with patch("api.main._predictor", mock):
        from api.main import app

        with TestClient(app) as c:
            response = c.post("/predict", json=VALID_TRANSACTION)
    assert response.status_code == 200
    assert response.json()["prediction"] == "fraud"
    assert response.json()["risk_tier"] == "CRITICAL"


# ── /predict/batch ─────────────────────────────────────────────────────────────


def test_predict_batch_valid_csv(client, tmp_path):
    import pandas as pd

    n = 10
    import numpy as np

    rng = np.random.RandomState(1)
    df = pd.DataFrame({f"V{i}": rng.randn(n) for i in range(1, 29)})
    df["Amount"] = rng.exponential(80, n)
    df["Time"] = np.sort(rng.uniform(0, 172_800, n))

    # Mock predict_batch
    mock = _make_mock_predictor()
    out_df = df.copy()
    out_df["probability"] = 0.05
    out_df["prediction"] = "legitimate"
    out_df["risk_tier"] = "LOW"
    mock.predict_batch.return_value = out_df

    csv_bytes = df.to_csv(index=False).encode()
    with patch("api.main._predictor", mock):
        from api.main import app

        with TestClient(app) as c:
            response = c.post(
                "/predict/batch",
                files={"file": ("transactions.csv", csv_bytes, "text/csv")},
            )
    assert response.status_code == 200
    data = response.json()
    assert data["total_transactions"] == n
    assert "fraud_count" in data
    assert "legitimate_count" in data


def test_predict_batch_non_csv_rejected(client):
    response = client.post(
        "/predict/batch",
        files={"file": ("data.txt", b"not a csv", "text/plain")},
    )
    assert response.status_code == 400

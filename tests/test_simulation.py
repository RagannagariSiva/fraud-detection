"""
tests/test_simulation.py
=========================
Unit tests for the real-time transaction simulator and fraud alert system.

These tests run without a live API — the HTTP client is mocked.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ══════════════════════════════════════════════════════════════════════════════
#  Simulation tests
# ══════════════════════════════════════════════════════════════════════════════

class TestSyntheticTransaction:
    """Tests for transaction generation logic."""

    def test_generate_legitimate_has_all_features(self):
        import random
        from simulation.real_time_transactions import SyntheticTransaction
        rng = random.Random(42)
        txn = SyntheticTransaction.generate(rng, sim_start_time=0.0, force_fraud=False)
        for i in range(1, 29):
            assert f"V{i}" in txn.features, f"V{i} missing from transaction"
        assert "Amount" in txn.features
        assert "Time" in txn.features

    def test_generate_fraud_sets_flag(self):
        import random
        from simulation.real_time_transactions import SyntheticTransaction
        rng = random.Random(42)
        txn = SyntheticTransaction.generate(rng, sim_start_time=0.0, force_fraud=True)
        assert txn.is_injected_fraud is True

    def test_generate_legitimate_sets_flag_false(self):
        import random
        from simulation.real_time_transactions import SyntheticTransaction
        rng = random.Random(42)
        txn = SyntheticTransaction.generate(rng, sim_start_time=0.0, force_fraud=False)
        assert txn.is_injected_fraud is False

    def test_amount_non_negative(self):
        import random
        from simulation.real_time_transactions import SyntheticTransaction
        rng = random.Random(99)
        for _ in range(50):
            txn = SyntheticTransaction.generate(rng, sim_start_time=0.0)
            assert txn.features["Amount"] >= 0.0, "Amount must be non-negative"

    def test_transaction_id_unique(self):
        import random
        import time
        from simulation.real_time_transactions import SyntheticTransaction
        rng = random.Random(7)
        ids = {SyntheticTransaction.generate(rng, sim_start_time=0.0).transaction_id
               for _ in range(20)}
        # IDs should be unique (or at least mostly unique given timestamp component)
        assert len(ids) >= 15


class TestFraudAPIClient:
    """Tests for the HTTP client wrapper."""

    def test_health_check_returns_true_on_ok(self):
        from simulation.real_time_transactions import FraudAPIClient
        client = FraudAPIClient("http://localhost:8000")

        mock_resp = MagicMock()
        mock_resp.json.return_value = {"status": "ok", "model_loaded": True}
        with patch("requests.Session.get", return_value=mock_resp):
            assert client.health_check() is True

    def test_health_check_returns_false_on_connection_error(self):
        import requests as req
        from simulation.real_time_transactions import FraudAPIClient
        client = FraudAPIClient("http://localhost:9999")
        with patch("requests.Session.get", side_effect=req.exceptions.ConnectionError):
            assert client.health_check() is False

    def test_predict_returns_result_on_200(self):
        from simulation.real_time_transactions import FraudAPIClient
        client = FraudAPIClient("http://localhost:8000")

        expected = {"prediction": "legitimate", "probability": 0.03, "risk_tier": "LOW"}
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = expected
        with patch("requests.Session.post", return_value=mock_resp):
            result = client.predict({"V1": 0.0, "Amount": 10.0, "Time": 0.0})
        assert result == expected

    def test_predict_returns_none_on_error(self):
        import requests as req
        from simulation.real_time_transactions import FraudAPIClient
        client = FraudAPIClient("http://localhost:8000")
        with patch("requests.Session.post", side_effect=req.exceptions.ConnectionError):
            result = client.predict({"V1": 0.0})
        assert result is None


# ══════════════════════════════════════════════════════════════════════════════
#  Alert system tests
# ══════════════════════════════════════════════════════════════════════════════

class TestAlertRecord:
    """Tests for the AlertRecord data class."""

    def test_fraud_alert_creates_record(self):
        from monitoring.fraud_alerts import AlertRecord
        rec = AlertRecord(
            transaction_id="TXN-001",
            prediction="fraud",
            probability=0.95,
            risk_tier="CRITICAL",
            amount=500.0,
        )
        assert rec.prediction == "fraud"
        assert rec.alert_level == "CRITICAL"
        assert "Block" in rec.action

    def test_to_dict_contains_all_fields(self):
        from monitoring.fraud_alerts import AlertRecord
        rec = AlertRecord("TXN-002", "fraud", 0.75, "HIGH", 200.0)
        d = rec.to_dict()
        for key in ("timestamp", "transaction_id", "prediction", "probability",
                     "risk_tier", "amount", "alert_level", "action"):
            assert key in d, f"Missing key: {key}"

    def test_to_json_is_valid_json(self):
        from monitoring.fraud_alerts import AlertRecord
        rec = AlertRecord("TXN-003", "fraud", 0.45, "HIGH", 99.0)
        parsed = json.loads(rec.to_json())
        assert parsed["transaction_id"] == "TXN-003"


class TestRollingStats:
    """Tests for the rolling statistics accumulator."""

    def test_records_fraud_count(self):
        from monitoring.fraud_alerts import RollingStats
        stats = RollingStats(window=100)
        for _ in range(10):
            stats.record("fraud", "CRITICAL")
        for _ in range(90):
            stats.record("legitimate", "LOW")
        snap = stats.snapshot()
        assert snap["total_seen"] == 100
        assert snap["total_fraud"] == 10

    def test_fraud_rate_calculation(self):
        from monitoring.fraud_alerts import RollingStats
        stats = RollingStats(window=100)
        for _ in range(5):
            stats.record("fraud", "HIGH")
        for _ in range(95):
            stats.record("legitimate", "LOW")
        snap = stats.snapshot()
        assert abs(snap["overall_fraud_rate"] - 0.05) < 0.001


class TestFraudAlertSystem:
    """Tests for the FraudAlertSystem dispatcher."""

    def test_process_legitimate_returns_none(self):
        from monitoring.fraud_alerts import FraudAlertSystem
        with tempfile.TemporaryDirectory() as tmpdir:
            alerter = FraudAlertSystem(log_dir=tmpdir)
            result = alerter.process(
                transaction_id="TXN-LEGIT",
                result={"prediction": "legitimate", "probability": 0.02, "risk_tier": "LOW"},
                amount=50.0,
            )
        assert result is None

    def test_process_fraud_returns_alert_record(self):
        from monitoring.fraud_alerts import FraudAlertSystem, AlertRecord
        with tempfile.TemporaryDirectory() as tmpdir:
            alerter = FraudAlertSystem(log_dir=tmpdir)
            result = alerter.process(
                transaction_id="TXN-FRAUD",
                result={"prediction": "fraud", "probability": 0.92, "risk_tier": "CRITICAL"},
                amount=500.0,
            )
        assert result is not None
        assert isinstance(result, AlertRecord)
        assert result.risk_tier == "CRITICAL"

    def test_alert_written_to_log_file(self):
        from monitoring.fraud_alerts import FraudAlertSystem
        with tempfile.TemporaryDirectory() as tmpdir:
            alerter = FraudAlertSystem(log_dir=tmpdir, log_filename="test_alerts.jsonl")
            alerter.process(
                transaction_id="TXN-LOG",
                result={"prediction": "fraud", "probability": 0.88, "risk_tier": "CRITICAL"},
                amount=1000.0,
            )
            log_path = Path(tmpdir) / "test_alerts.jsonl"
            assert log_path.exists()
            with open(log_path) as f:
                lines = f.readlines()
            assert len(lines) == 1
            parsed = json.loads(lines[0])
            assert parsed["transaction_id"] == "TXN-LOG"
            assert parsed["prediction"] == "fraud"

    def test_multiple_alerts_all_logged(self):
        from monitoring.fraud_alerts import FraudAlertSystem
        with tempfile.TemporaryDirectory() as tmpdir:
            alerter = FraudAlertSystem(log_dir=tmpdir, log_filename="multi_alerts.jsonl")
            for i in range(5):
                alerter.process(
                    transaction_id=f"TXN-{i:03d}",
                    result={"prediction": "fraud", "probability": 0.9, "risk_tier": "HIGH"},
                    amount=100.0,
                )
            log_path = Path(tmpdir) / "multi_alerts.jsonl"
            with open(log_path) as f:
                lines = f.readlines()
        assert len(lines) == 5

    def test_get_stats_returns_dict(self):
        from monitoring.fraud_alerts import FraudAlertSystem
        with tempfile.TemporaryDirectory() as tmpdir:
            alerter = FraudAlertSystem(log_dir=tmpdir)
            stats = alerter.get_stats()
        assert "total_seen" in stats
        assert "overall_fraud_rate" in stats

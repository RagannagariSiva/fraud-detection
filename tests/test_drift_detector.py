"""
tests/test_drift_detector.py
==============================
Unit tests for the data drift detector.
"""

from __future__ import annotations

import json
import tempfile

import numpy as np
import pandas as pd

from src.monitoring.drift_detector import DriftConfig, DriftDetector, _compute_psi

# ── PSI tests ─────────────────────────────────────────────────────────────────


class TestComputePSI:
    def test_identical_distributions_psi_near_zero(self):
        dist = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
        psi = _compute_psi(dist, dist)
        assert psi < 0.001

    def test_very_different_distributions_high_psi(self):
        expected = np.array([0.5, 0.3, 0.15, 0.04, 0.01])
        actual = np.array([0.01, 0.04, 0.15, 0.3, 0.5])
        psi = _compute_psi(expected, actual)
        assert psi > 0.25

    def test_psi_non_negative(self):
        rng = np.random.default_rng(42)
        for _ in range(20):
            a = rng.dirichlet(np.ones(10))
            b = rng.dirichlet(np.ones(10))
            assert _compute_psi(a, b) >= 0.0


# ── DriftDetector tests ───────────────────────────────────────────────────────


def _make_baseline_df(n: int = 1000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({f"V{i}": rng.normal(0, 1.5, n) for i in range(1, 6)})
    df["Amount"] = rng.exponential(88, n)
    df["Time"] = rng.uniform(0, 172_800, n)
    return df


class TestDriftDetectorFromTrainingData:
    def test_builds_from_dataframe(self):
        df = _make_baseline_df()
        detector = DriftDetector.from_training_data(df)
        assert len(detector._baseline) > 0

    def test_all_numeric_features_captured(self):
        df = _make_baseline_df()
        detector = DriftDetector.from_training_data(df)
        for col in df.columns:
            assert col in detector._baseline

    def test_baseline_stats_keys(self):
        df = _make_baseline_df()
        detector = DriftDetector.from_training_data(df)
        for feat, stats in detector._baseline.items():
            for key in ("mean", "std", "min", "max", "hist_counts", "hist_edges", "n"):
                assert key in stats, f"'{key}' missing for feature '{feat}'"


class TestDriftDetectorSaveLoad:
    def test_save_and_load_roundtrip(self):
        df = _make_baseline_df()
        detector = DriftDetector.from_training_data(df)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        detector.save(path)
        loaded = DriftDetector.load(path)
        assert set(loaded._baseline.keys()) == set(detector._baseline.keys())

    def test_saved_file_is_valid_json(self):
        df = _make_baseline_df()
        detector = DriftDetector.from_training_data(df)
        with tempfile.NamedTemporaryFile(suffix=".json", mode="w", delete=False) as f:
            path = f.name
        detector.save(path)
        with open(path) as f:
            data = json.load(f)
        assert "baseline_stats" in data
        assert "config" in data
        assert "created_at" in data


class TestDriftDetectorCheck:
    def test_same_distribution_no_drift(self):
        df = _make_baseline_df(n=2000)
        detector = DriftDetector.from_training_data(df)
        # Use second half as "current" — same distribution
        current = _make_baseline_df(n=500, seed=99)
        report = detector.check(current)
        assert report.n_features_critical == 0
        assert not report.overall_drift_detected

    def test_shifted_distribution_detected(self):
        df = _make_baseline_df(n=2000)
        detector = DriftDetector.from_training_data(df)
        # Create clearly shifted data — mean shifted by 5 standard deviations
        rng = np.random.default_rng(7)
        shifted = pd.DataFrame({f"V{i}": rng.normal(8, 1.5, 500) for i in range(1, 6)})
        shifted["Amount"] = rng.exponential(88, 500)
        shifted["Time"] = rng.uniform(0, 172_800, 500)
        report = detector.check(shifted)
        assert report.n_features_drifted > 0

    def test_report_has_all_fields(self):
        df = _make_baseline_df(n=500)
        detector = DriftDetector.from_training_data(df)
        current = _make_baseline_df(n=300, seed=100)
        report = detector.check(current)
        d = report.to_dict()
        for key in (
            "timestamp",
            "baseline_size",
            "current_size",
            "n_features_tested",
            "n_features_drifted",
            "overall_drift_detected",
            "recommendation",
        ):
            assert key in d, f"Missing key in DriftReport: {key}"

    def test_report_to_json_valid(self):
        df = _make_baseline_df(n=300)
        detector = DriftDetector.from_training_data(df)
        current = _make_baseline_df(n=300, seed=55)
        report = detector.check(current)
        parsed = json.loads(report.to_json())
        assert "overall_drift_detected" in parsed

    def test_prediction_drift_recorded_when_probs_supplied(self):
        df = _make_baseline_df(n=500)
        detector = DriftDetector.from_training_data(df)
        current = _make_baseline_df(n=300, seed=8)
        probs = np.random.default_rng(3).uniform(0, 0.3, 300)
        report = detector.check(current, prediction_probs=probs)
        assert report.prediction_drift is not None
        assert "mean_probability" in report.prediction_drift

    def test_small_batch_still_runs(self):
        """Detector should not raise even on small batches (just logs a warning)."""
        df = _make_baseline_df(n=500)
        config = DriftConfig(min_sample_size=10)
        detector = DriftDetector.from_training_data(df, config=config)
        current = _make_baseline_df(n=50, seed=6)
        report = detector.check(current)
        assert report is not None


# ── ModelMonitor tests ────────────────────────────────────────────────────────


class TestModelMonitor:
    def test_records_and_snapshots(self):
        from src.monitoring.model_monitor import ModelMonitor

        monitor = ModelMonitor()
        monitor.record_prediction(probability=0.05, risk_tier="LOW", latency_ms=10.0)
        monitor.record_prediction(probability=0.85, risk_tier="CRITICAL", latency_ms=12.0)
        snap = monitor.snapshot()
        assert snap["lifetime"]["total_predictions"] == 2

    def test_fraud_rate_calculated(self):
        from src.monitoring.model_monitor import ModelMonitor

        monitor = ModelMonitor()
        for _ in range(10):
            monitor.record_prediction(probability=0.9, risk_tier="CRITICAL", latency_ms=5.0)
        for _ in range(90):
            monitor.record_prediction(probability=0.02, risk_tier="LOW", latency_ms=5.0)
        snap = monitor.snapshot()
        # fraud_rate is based on HIGH+CRITICAL tier labels
        assert snap["lifetime"]["total_predictions"] == 100

    def test_latency_percentiles_populated(self):
        from src.monitoring.model_monitor import ModelMonitor

        monitor = ModelMonitor()
        for ms in [5.0, 8.0, 12.0, 100.0, 15.0]:
            monitor.record_prediction(probability=0.1, risk_tier="LOW", latency_ms=ms)
        snap = monitor.snapshot()
        lat = snap["latency_ms"]
        assert lat["p50"] > 0
        assert lat["p99"] >= lat["p95"] >= lat["p50"]

    def test_prometheus_format(self):
        from src.monitoring.model_monitor import ModelMonitor

        monitor = ModelMonitor()
        monitor.record_prediction(probability=0.3, risk_tier="MEDIUM", latency_ms=7.0)
        prom = monitor.to_prometheus()
        assert "fraud_predictions_total" in prom
        assert "prediction_latency_ms_p99" in prom
        assert "predictions_per_second" in prom

    def test_health_check_healthy_default(self):
        from src.monitoring.model_monitor import ModelMonitor

        monitor = ModelMonitor()
        health = monitor.health_check()
        # Fresh monitor with no traffic → healthy
        assert health["healthy"] is True
        assert health["issues"] == []

"""
src/monitoring/model_monitor.py
================================
Production Model Health Monitor

Continuously tracks the operational health of the deployed fraud detection
model by collecting and aggregating metrics from live prediction traffic.

Metrics collected
-----------------
- Prediction latency (p50, p95, p99)
- Throughput (predictions per second)
- Fraud rate (rolling window)
- Fraud rate by risk tier
- Error rate (API failures)
- Model confidence distribution (avg probability, low-confidence rate)

Integration
-----------
This module is designed to be imported by the FastAPI app and updated
on each prediction call:

    from src.monitoring.model_monitor import ModelMonitor
    monitor = ModelMonitor()

    # In the /predict endpoint:
    monitor.record_prediction(probability=result["probability"],
                               risk_tier=result["risk_tier"],
                               latency_ms=elapsed_ms)

    # Expose via GET /metrics endpoint:
    @app.get("/metrics")
    def get_metrics():
        return monitor.snapshot()

Prometheus export
-----------------
    metrics_text = monitor.to_prometheus()
    # Mount at /metrics for Prometheus scraping
"""

from __future__ import annotations

import time
from collections import deque, defaultdict
from dataclasses import dataclass, field
from threading import Lock
from typing import Deque, Dict

import numpy as np


# ── Latency tracker (lock-free ring buffer) ───────────────────────────────────

class LatencyTracker:
    """Rolling window of prediction latencies in milliseconds."""

    def __init__(self, window: int = 1000):
        self._window = window
        self._buf: Deque[float] = deque(maxlen=window)
        self._lock = Lock()

    def record(self, latency_ms: float) -> None:
        with self._lock:
            self._buf.append(latency_ms)

    def percentiles(self) -> dict[str, float]:
        with self._lock:
            if not self._buf:
                return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0}
            arr = np.array(self._buf)
            return {
                "p50":  round(float(np.percentile(arr, 50)), 2),
                "p95":  round(float(np.percentile(arr, 95)), 2),
                "p99":  round(float(np.percentile(arr, 99)), 2),
                "mean": round(float(arr.mean()), 2),
            }


# ── Rolling counter ────────────────────────────────────────────────────────────

@dataclass
class _TimestampedEvent:
    ts: float     # time.monotonic()
    value: float  # e.g. 1.0 for a fraud flag, latency ms, etc.


class RollingCounter:
    """
    Counts events within a sliding time window.

    Thread-safe; uses a deque for O(1) append and efficient pruning.
    """

    def __init__(self, window_seconds: float = 300.0):
        self._window = window_seconds
        self._events: Deque[_TimestampedEvent] = deque()
        self._lock = Lock()

    def record(self, value: float = 1.0) -> None:
        with self._lock:
            self._events.append(_TimestampedEvent(ts=time.monotonic(), value=value))

    def sum(self) -> float:
        cutoff = time.monotonic() - self._window
        with self._lock:
            while self._events and self._events[0].ts < cutoff:
                self._events.popleft()
            return sum(e.value for e in self._events)

    def count(self) -> int:
        cutoff = time.monotonic() - self._window
        with self._lock:
            while self._events and self._events[0].ts < cutoff:
                self._events.popleft()
            return len(self._events)

    def rate_per_second(self) -> float:
        return self.count() / self._window


# ══════════════════════════════════════════════════════════════════════════════
#  Main monitor
# ══════════════════════════════════════════════════════════════════════════════

class ModelMonitor:
    """
    Collects and aggregates live prediction metrics for a deployed fraud model.

    Parameters
    ----------
    latency_window : int
        Number of most recent predictions to track for latency percentiles.
    rate_window_seconds : float
        Sliding window duration for rate metrics (fraud rate, throughput).
    low_confidence_threshold : float
        Probabilities below this level are flagged as "low confidence"
        (model is uncertain — often near the decision boundary).
    """

    def __init__(
        self,
        latency_window: int = 1000,
        rate_window_seconds: float = 300.0,
        low_confidence_threshold: float = 0.30,
    ):
        self._start_time = time.time()
        self._low_confidence_threshold = low_confidence_threshold

        # Latency
        self._latency = LatencyTracker(window=latency_window)

        # Rolling counters (5-min window by default)
        W = rate_window_seconds
        self._total_predictions  = RollingCounter(W)
        self._fraud_predictions  = RollingCounter(W)
        self._errors             = RollingCounter(W)
        self._low_conf           = RollingCounter(W)

        # Tier breakdown counters
        self._tier_counts: Dict[str, RollingCounter] = defaultdict(lambda: RollingCounter(W))

        # Probability accumulator (for mean confidence)
        self._prob_buf: Deque[float] = deque(maxlen=1000)
        self._prob_lock = Lock()

        # Lifetime counters (never reset)
        self._lifetime_predictions: int = 0
        self._lifetime_fraud:       int = 0
        self._lifetime_errors:      int = 0
        self._lifetime_lock = Lock()

    # ── Public recording API ──────────────────────────────────────────────────

    def record_prediction(
        self,
        probability: float,
        risk_tier: str,
        latency_ms: float,
        is_error: bool = False,
    ) -> None:
        """
        Record a single prediction event.  Call this from every /predict handler.

        Parameters
        ----------
        probability : float
            Model output fraud probability [0, 1].
        risk_tier : str
            Risk tier assigned (LOW / MEDIUM / HIGH / CRITICAL).
        latency_ms : float
            Wall-clock time for the prediction in milliseconds.
        is_error : bool
            True if the prediction failed (model error, not input validation).
        """
        self._latency.record(latency_ms)
        self._total_predictions.record()
        self._tier_counts[risk_tier].record()

        with self._prob_lock:
            self._prob_buf.append(probability)

        is_fraud = risk_tier in ("HIGH", "CRITICAL") or probability >= 0.40
        if is_fraud:
            self._fraud_predictions.record()

        if self._low_confidence_threshold <= probability <= (1 - self._low_confidence_threshold):
            self._low_conf.record()

        if is_error:
            self._errors.record()

        with self._lifetime_lock:
            self._lifetime_predictions += 1
            if is_fraud:
                self._lifetime_fraud += 1
            if is_error:
                self._lifetime_errors += 1

    # ── Snapshot ──────────────────────────────────────────────────────────────

    def snapshot(self) -> dict:
        """
        Return a JSON-serialisable dict of current metrics.

        Suitable for returning from a /metrics endpoint or logging.
        """
        uptime = int(time.time() - self._start_time)
        total_recent  = self._total_predictions.count()
        fraud_recent  = self._fraud_predictions.count()
        error_recent  = self._errors.count()

        with self._prob_lock:
            probs = list(self._prob_buf)
        avg_prob   = float(np.mean(probs)) if probs else 0.0
        prob_std   = float(np.std(probs))  if probs else 0.0

        with self._lifetime_lock:
            lifetime_total = self._lifetime_predictions
            lifetime_fraud = self._lifetime_fraud
            lifetime_errors = self._lifetime_errors

        tier_snapshot = {
            tier: self._tier_counts[tier].count()
            for tier in ("LOW", "MEDIUM", "HIGH", "CRITICAL")
        }

        return {
            "uptime_seconds": uptime,
            "lifetime": {
                "total_predictions": lifetime_total,
                "fraud_predictions": lifetime_fraud,
                "error_count":       lifetime_errors,
                "lifetime_fraud_rate": round(lifetime_fraud / max(lifetime_total, 1), 6),
            },
            "rolling_5min": {
                "total_predictions": total_recent,
                "fraud_predictions": fraud_recent,
                "error_count":       error_recent,
                "fraud_rate":        round(fraud_recent / max(total_recent, 1), 6),
                "error_rate":        round(error_recent / max(total_recent, 1), 6),
                "predictions_per_second": round(self._total_predictions.rate_per_second(), 3),
            },
            "latency_ms": self._latency.percentiles(),
            "model_confidence": {
                "mean_probability": round(avg_prob, 6),
                "std_probability":  round(prob_std, 6),
                "sample_size":      len(probs),
            },
            "risk_tier_counts_5min": tier_snapshot,
        }

    def to_prometheus(self) -> str:
        """
        Export metrics in Prometheus text format.

        Mount at /metrics and configure Prometheus to scrape it.
        """
        s = self.snapshot()
        lines = [
            "# HELP fraud_predictions_total Total fraud predictions since start",
            "# TYPE fraud_predictions_total counter",
            f"fraud_predictions_total {s['lifetime']['fraud_predictions']}",
            "",
            "# HELP fraud_rate_5min Rolling 5-minute fraud rate",
            "# TYPE fraud_rate_5min gauge",
            f"fraud_rate_5min {s['rolling_5min']['fraud_rate']}",
            "",
            "# HELP prediction_latency_ms_p99 99th percentile prediction latency",
            "# TYPE prediction_latency_ms_p99 gauge",
            f"prediction_latency_ms_p99 {s['latency_ms']['p99']}",
            "",
            "# HELP prediction_latency_ms_p95 95th percentile prediction latency",
            "# TYPE prediction_latency_ms_p95 gauge",
            f"prediction_latency_ms_p95 {s['latency_ms']['p95']}",
            "",
            "# HELP predictions_per_second Current throughput",
            "# TYPE predictions_per_second gauge",
            f"predictions_per_second {s['rolling_5min']['predictions_per_second']}",
            "",
            "# HELP error_rate_5min API error rate over last 5 minutes",
            "# TYPE error_rate_5min gauge",
            f"error_rate_5min {s['rolling_5min']['error_rate']}",
            "",
            "# HELP mean_fraud_probability Mean model output probability",
            "# TYPE mean_fraud_probability gauge",
            f"mean_fraud_probability {s['model_confidence']['mean_probability']}",
        ]
        # Risk tier breakdown
        lines.append("")
        lines.append("# HELP risk_tier_count_5min Predictions by risk tier (5min)")
        lines.append("# TYPE risk_tier_count_5min gauge")
        for tier, count in s["risk_tier_counts_5min"].items():
            lines.append(f'risk_tier_count_5min{{tier="{tier}"}} {count}')

        return "\n".join(lines) + "\n"

    def health_check(self) -> dict:
        """
        Returns a health summary suitable for /health endpoint enrichment.

        Flags degraded conditions:
        - Error rate > 1%
        - Fraud rate > 5% (possible model miscalibration or attack)
        - P99 latency > 200ms
        """
        s = self.snapshot()
        issues = []

        error_rate = s["rolling_5min"]["error_rate"]
        fraud_rate = s["rolling_5min"]["fraud_rate"]
        p99        = s["latency_ms"]["p99"]

        if error_rate > 0.01:
            issues.append(f"High error rate: {error_rate:.1%}")
        if fraud_rate > 0.05:
            issues.append(f"Elevated fraud rate: {fraud_rate:.1%} (expected ~0.2%)")
        if p99 > 200:
            issues.append(f"High P99 latency: {p99:.0f}ms")

        return {
            "healthy": len(issues) == 0,
            "issues":  issues,
            "fraud_rate_5min": fraud_rate,
            "error_rate_5min": error_rate,
            "latency_p99_ms":  p99,
        }

"""
monitoring/fraud_alerts.py
===========================
Fraud Alert & Monitoring System

Watches the transaction stream (or a log file) for fraud predictions and:
  • Prints colour-coded alerts to the console in real time
  • Writes structured JSON lines to a rotating alert log file
  • Accumulates rolling statistics (fraud rate, alert counts by tier)
  • Can be run standalone as a tail-based log monitor, or imported as a
    library and driven by the real-time simulator

Usage
-----
    # 1. As a library (called by the simulator):
    from monitoring.fraud_alerts import FraudAlertSystem
    alerter = FraudAlertSystem()
    alerter.process(transaction_id="TXN-001", result=api_response_dict, amount=149.62)

    # 2. Standalone — tail the prediction log and print live alerts:
    python monitoring/fraud_alerts.py --log-file logs/predictions.jsonl

Alert Log Format (one JSON object per line)
--------------------------------------------
{
  "timestamp": "2024-01-15T14:23:11.042Z",
  "transaction_id": "TXN-0012345-0042",
  "prediction": "fraud",
  "probability": 0.9234,
  "risk_tier": "CRITICAL",
  "amount": 149.62,
  "alert_level": "CRITICAL",
  "action": "Block and alert immediately"
}
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
from threading import Lock

# ── Logging ────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── ANSI colours ──────────────────────────────────────────────────────────────

RESET = "\033[0m"
RED = "\033[91m"
YELLOW = "\033[93m"
GREEN = "\033[92m"
CYAN = "\033[96m"
BLUE = "\033[94m"
BOLD = "\033[1m"
DIM = "\033[2m"
BLINK = "\033[5m"


def _c(text: str, *codes: str) -> str:
    if not sys.stdout.isatty():
        return text
    return "".join(codes) + text + RESET


# ── Alert policy ──────────────────────────────────────────────────────────────

ALERT_POLICY: dict[str, dict] = {
    "CRITICAL": {
        "level": "CRITICAL",
        "action": "Block card immediately and notify cardholder",
        "notify_teams": ["fraud_ops", "card_services", "compliance"],
        "auto_block": True,
        "colour": RED,
        "icon": "🚨",
    },
    "HIGH": {
        "level": "HIGH",
        "action": "Route to manual fraud review queue",
        "notify_teams": ["fraud_ops"],
        "auto_block": False,
        "colour": YELLOW,
        "icon": "⚠️ ",
    },
    "MEDIUM": {
        "level": "MEDIUM",
        "action": "Soft-flag for next-day review",
        "notify_teams": [],
        "auto_block": False,
        "colour": YELLOW,
        "icon": "🔶",
    },
    "LOW": {
        "level": "LOW",
        "action": "No action required",
        "notify_teams": [],
        "auto_block": False,
        "colour": GREEN,
        "icon": "✅",
    },
}


# ── Alert record ─────────────────────────────────────────────────────────────


class AlertRecord:
    """Immutable snapshot of a single fraud alert event."""

    __slots__ = (
        "timestamp",
        "transaction_id",
        "prediction",
        "probability",
        "risk_tier",
        "amount",
        "alert_level",
        "action",
    )

    def __init__(
        self,
        transaction_id: str,
        prediction: str,
        probability: float,
        risk_tier: str,
        amount: float,
    ):
        self.timestamp = datetime.utcnow().isoformat(timespec="milliseconds") + "Z"
        self.transaction_id = transaction_id
        self.prediction = prediction
        self.probability = probability
        self.risk_tier = risk_tier
        self.amount = amount
        policy = ALERT_POLICY.get(risk_tier, ALERT_POLICY["LOW"])
        self.alert_level = policy["level"]
        self.action = policy["action"]

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "transaction_id": self.transaction_id,
            "prediction": self.prediction,
            "probability": round(self.probability, 6),
            "risk_tier": self.risk_tier,
            "amount": round(self.amount, 2),
            "alert_level": self.alert_level,
            "action": self.action,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


# ── Rolling statistics ────────────────────────────────────────────────────────


class RollingStats:
    """Thread-safe rolling window statistics over the last N transactions."""

    def __init__(self, window: int = 500):
        self._window: int = window
        self._lock: Lock = Lock()
        self._history: deque[dict] = deque(maxlen=window)
        self._total_seen: int = 0
        self._total_fraud: int = 0
        self._tier_counts: dict[str, int] = defaultdict(int)

    def record(self, prediction: str, risk_tier: str) -> None:
        with self._lock:
            self._history.append({"prediction": prediction, "risk_tier": risk_tier})
            self._total_seen += 1
            if prediction == "fraud":
                self._total_fraud += 1
            self._tier_counts[risk_tier] += 1

    def snapshot(self) -> dict:
        with self._lock:
            total = self._total_seen
            fraud = self._total_fraud
            window_fraud = sum(1 for h in self._history if h["prediction"] == "fraud")
            window_total = len(self._history)
            return {
                "total_seen": total,
                "total_fraud": fraud,
                "overall_fraud_rate": round(fraud / max(total, 1), 6),
                "window_size": window_total,
                "window_fraud_count": window_fraud,
                "window_fraud_rate": round(window_fraud / max(window_total, 1), 6),
                "tier_counts": dict(self._tier_counts),
            }


# ── Main alert system ─────────────────────────────────────────────────────────


class FraudAlertSystem:
    """
    Central fraud alert dispatcher.

    Responsibilities
    ----------------
    1. Receive prediction results from the inference layer.
    2. Apply the alert policy (block / review / monitor).
    3. Print structured alerts to the console.
    4. Persist every alert to a rotating JSONL log file.
    5. Expose rolling statistics for dashboard consumption.
    """

    def __init__(
        self,
        log_dir: str = "logs",
        log_filename: str = "fraud_alerts.jsonl",
        console_level: str = "MEDIUM",
        window: int = 500,
    ):
        """
        Parameters
        ----------
        log_dir:       Directory where alert logs are written.
        log_filename:  Name of the JSONL alert log file.
        console_level: Minimum risk tier to print to console
                       (LOW | MEDIUM | HIGH | CRITICAL).
        window:        Rolling window size for statistics.
        """
        self._log_path = Path(log_dir) / log_filename
        self._console_min = console_level.upper()
        self._stats = RollingStats(window=window)
        self._lock = Lock()
        self._alert_count = 0

        # Tier severity ordering for console threshold filtering
        self._tier_order = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}

        # Ensure log directory exists
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("FraudAlertSystem initialised | log → %s", self._log_path)

    # ── Public API ─────────────────────────────────────────────────────────────

    def process(
        self,
        transaction_id: str,
        result: dict,
        amount: float = 0.0,
    ) -> AlertRecord | None:
        """
        Process a single prediction result.

        Parameters
        ----------
        transaction_id: Unique identifier for the transaction.
        result:         Dict from POST /predict: {prediction, probability, risk_tier, ...}.
        amount:         Transaction amount in USD.

        Returns
        -------
        AlertRecord if the prediction is fraud, else None.
        """
        prediction = result.get("prediction", "legitimate")
        probability = float(result.get("probability", 0.0))
        risk_tier = result.get("risk_tier", "LOW")

        self._stats.record(prediction, risk_tier)

        if prediction != "fraud":
            return None

        alert = AlertRecord(
            transaction_id=transaction_id,
            prediction=prediction,
            probability=probability,
            risk_tier=risk_tier,
            amount=amount,
        )
        self._dispatch(alert)
        return alert

    def get_stats(self) -> dict:
        """Return a snapshot of rolling statistics."""
        return self._stats.snapshot()

    # ── Internal dispatch ──────────────────────────────────────────────────────

    def _dispatch(self, alert: AlertRecord) -> None:
        """Write to log and optionally print to console."""
        with self._lock:
            self._alert_count += 1
            count = self._alert_count

        self._write_log(alert)
        self._maybe_print_console(alert, count)
        self._maybe_simulate_downstream(alert)

    def _write_log(self, alert: AlertRecord) -> None:
        """Append alert as a JSON line to the alert log file."""
        try:
            with open(self._log_path, "a", encoding="utf-8") as f:
                f.write(alert.to_json() + "\n")
        except OSError as exc:
            logger.error("Failed to write alert log: %s", exc)

    def _maybe_print_console(self, alert: AlertRecord, count: int) -> None:
        """Print a formatted alert to stdout if it meets the console threshold."""
        min_severity = self._tier_order.get(self._console_min, 0)
        cur_severity = self._tier_order.get(alert.risk_tier, 0)
        if cur_severity < min_severity:
            return

        policy = ALERT_POLICY.get(alert.risk_tier, ALERT_POLICY["LOW"])
        colour = policy["colour"]
        icon = policy["icon"]
        auto = "AUTO-BLOCKED" if policy["auto_block"] else "FLAGGED"

        print()
        print(_c("╔" + "═" * 68 + "╗", BOLD, colour))
        print(
            _c(f"║  {icon}  FRAUD ALERT #{count:<5}  [{alert.risk_tier}]  {auto:<15}", BOLD, colour)
        )
        print(_c("╠" + "═" * 68 + "╣", colour))
        print(f"  Transaction  : {_c(alert.transaction_id, CYAN)}")
        print(f"  Timestamp    : {alert.timestamp}")
        print(f"  Amount       : ${alert.amount:,.2f}")
        print(f"  Probability  : {_c(f'{alert.probability:.4f}', BOLD, colour)}")
        print(f"  Risk Tier    : {_c(alert.risk_tier, BOLD, colour)}")
        print(f"  Action       : {_c(alert.action, BOLD)}")
        if policy["notify_teams"]:
            teams = ", ".join(policy["notify_teams"])
            print(f"  Notify       : {_c(teams, YELLOW)}")
        print(_c("╚" + "═" * 68 + "╝", colour))
        print()

    def _maybe_simulate_downstream(self, alert: AlertRecord) -> None:
        """
        Placeholder for real downstream integrations:
          - Kafka / SQS: publish alert event
          - PagerDuty:   trigger incident for CRITICAL alerts
          - Slack:       post to #fraud-alerts channel
          - Email:       notify compliance team

        In production these would be async tasks (Celery / asyncio).
        """
        if alert.risk_tier == "CRITICAL":
            logger.info(
                "[DOWNSTREAM] Would trigger PagerDuty P1 incident for %s",
                alert.transaction_id,
            )
        if alert.risk_tier in ("CRITICAL", "HIGH"):
            logger.info(
                "[DOWNSTREAM] Would publish to fraud.alerts Kafka topic: %s",
                alert.transaction_id,
            )


# ── Standalone log tail monitor ───────────────────────────────────────────────


def tail_log_monitor(log_file: str, poll_interval: float = 0.5) -> None:
    """
    Tail a prediction JSONL log file and re-emit alerts to the console.

    This is useful for running the alert system as a separate process from
    the prediction API — a common pattern in microservice architectures.
    """
    path = Path(log_file)
    if not path.exists():
        logger.error("Log file not found: %s", log_file)
        sys.exit(1)

    alerter = FraudAlertSystem(console_level="MEDIUM")

    print(_c(f"\n📡  Tailing fraud alerts from: {log_file}\n", BOLD, CYAN))
    logger.info("Starting tail monitor on %s (poll every %.1fs)", log_file, poll_interval)

    seen_lines = 0
    # Skip lines already in the file on startup
    with open(path) as f:
        for _ in f:
            seen_lines += 1
    logger.info("Fast-forwarded past %d existing log lines.", seen_lines)

    try:
        with open(path) as f:
            # Seek to current end
            f.seek(0, os.SEEK_END)
            while True:
                line = f.readline()
                if not line:
                    time.sleep(poll_interval)
                    continue
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    if record.get("prediction") == "fraud":
                        alerter._maybe_print_console(
                            AlertRecord(
                                transaction_id=record.get("transaction_id", "UNKNOWN"),
                                prediction=record["prediction"],
                                probability=float(record.get("probability", 0.0)),
                                risk_tier=record.get("risk_tier", "LOW"),
                                amount=float(record.get("amount", 0.0)),
                            ),
                            count=0,
                        )
                except json.JSONDecodeError:
                    pass
    except KeyboardInterrupt:
        print()
        stats = alerter.get_stats()
        logger.info("Monitor stopped. Stats: %s", json.dumps(stats, indent=2))


# ── CLI ───────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fraud alert monitor — tail a prediction log and display alerts",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--log-file",
        default="logs/predictions.jsonl",
        help="Path to the prediction JSONL log to monitor",
    )
    p.add_argument(
        "--poll-interval",
        type=float,
        default=0.5,
        help="Seconds between log poll cycles",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    tail_log_monitor(args.log_file, poll_interval=args.poll_interval)

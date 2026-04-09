"""
simulation/real_time_transactions.py
=====================================
Real-Time Fraud Detection Transaction Simulator

Simulates a live stream of credit card transactions, sends each one to the
fraud detection API, and prints colour-coded results to the console in real time.

Usage
-----
    # Start the API first:
    uvicorn api.main:app --port 8000

    # Then run the simulator:
    python simulation/real_time_transactions.py
    python simulation/real_time_transactions.py --tps 3 --duration 60
    python simulation/real_time_transactions.py --fraud-rate 0.15 --tps 1

Options
-------
    --api-url     Base URL of the prediction API  (default: http://localhost:8000)
    --tps         Transactions per second          (default: 2)
    --duration    How many seconds to run          (default: 120, 0 = infinite)
    --fraud-rate  Fraction of synthetic frauds     (default: 0.05)
    --seed        Random seed for reproducibility  (default: 42)
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests

# ── Logging setup ──────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Colour helpers (ANSI, disabled on non-TTY) ────────────────────────────────

RESET  = "\033[0m"
RED    = "\033[91m"
YELLOW = "\033[93m"
GREEN  = "\033[92m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
DIM    = "\033[2m"


def _c(text: str, *codes: str) -> str:
    """Wrap text in ANSI colour codes only when writing to a TTY."""
    if not sys.stdout.isatty():
        return text
    return "".join(codes) + text + RESET


# ── Transaction generator ─────────────────────────────────────────────────────

# Approximate means/stds from the real Kaggle creditcard dataset
_LEGIT_STATS: dict[str, tuple[float, float]] = {
    **{f"V{i}": (0.0, 1.5) for i in range(1, 29)},
    "Amount": (88.0, 250.0),
    "Time": (0.0, 47_906.0),
}

# Fraudulent transactions tend to have extreme V values and small amounts
_FRAUD_STATS: dict[str, tuple[float, float]] = {
    **{f"V{i}": (0.0, 3.2) for i in range(1, 29)},
    "V1":  (-4.8, 2.0),
    "V3":  (-5.5, 2.5),
    "V4":  ( 4.1, 2.0),
    "V10": (-5.2, 2.5),
    "V12": (-6.0, 2.5),
    "V14": (-9.5, 3.0),
    "V17": (-7.0, 3.0),
    "Amount": (122.0, 256.0),
    "Time": (0.0, 47_906.0),
}


@dataclass
class SyntheticTransaction:
    """A single synthetic credit card transaction with generation metadata."""

    transaction_id: str
    timestamp: str
    is_injected_fraud: bool
    features: dict[str, float] = field(default_factory=dict)

    @classmethod
    def generate(
        cls,
        rng: random.Random,
        sim_start_time: float,
        force_fraud: bool = False,
    ) -> "SyntheticTransaction":
        """
        Generate a realistic synthetic transaction.

        ``force_fraud=True`` draws feature values from the fraud distribution,
        making the model more likely (but not guaranteed) to flag it.
        """
        txn_id = f"TXN-{int(time.time()*1000) % 10_000_000:07d}-{rng.randint(0, 9999):04d}"
        ts = datetime.utcnow().isoformat(timespec="milliseconds") + "Z"
        elapsed = time.time() - sim_start_time
        stats = _FRAUD_STATS if force_fraud else _LEGIT_STATS

        features: dict[str, float] = {}
        for col, (mu, sigma) in stats.items():
            if col == "Time":
                features[col] = max(0.0, elapsed)
            elif col == "Amount":
                val = abs(rng.gauss(mu, sigma))
                features[col] = round(val, 2)
            else:
                features[col] = round(rng.gauss(mu, sigma), 6)

        return cls(
            transaction_id=txn_id,
            timestamp=ts,
            is_injected_fraud=force_fraud,
            features=features,
        )


# ── API client ─────────────────────────────────────────────────────────────────

class FraudAPIClient:
    """Thin HTTP client for the fraud detection REST API."""

    def __init__(self, base_url: str, timeout: float = 5.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session = requests.Session()
        self._session.headers.update({"Content-Type": "application/json"})

    def health_check(self) -> bool:
        """Return True if the API is reachable and healthy."""
        try:
            r = self._session.get(f"{self.base_url}/health", timeout=self.timeout)
            data = r.json()
            return data.get("status") in ("ok", "degraded")
        except Exception:
            return False

    def predict(self, features: dict[str, float]) -> Optional[dict]:
        """
        POST /predict and return the parsed response dict, or None on error.
        """
        try:
            r = self._session.post(
                f"{self.base_url}/predict",
                data=json.dumps(features),
                timeout=self.timeout,
            )
            if r.status_code == 200:
                return r.json()
            logger.warning("API returned HTTP %d: %s", r.status_code, r.text[:200])
            return None
        except requests.exceptions.ConnectionError:
            logger.error("Cannot reach API at %s — is it running?", self.base_url)
            return None
        except requests.exceptions.Timeout:
            logger.warning("API request timed out after %.1fs", self.timeout)
            return None
        except Exception as exc:
            logger.error("Unexpected API error: %s", exc)
            return None


# ── Console display ────────────────────────────────────────────────────────────

RISK_COLOUR = {
    "LOW":      GREEN,
    "MEDIUM":   YELLOW,
    "HIGH":     YELLOW,
    "CRITICAL": RED,
}


def _print_result(txn: SyntheticTransaction, result: dict, idx: int) -> None:
    """Print a single transaction result in a coloured, human-readable format."""
    prediction = result.get("prediction", "unknown")
    probability = result.get("probability", 0.0)
    risk_tier   = result.get("risk_tier", "UNKNOWN")
    colour      = RISK_COLOUR.get(risk_tier, RESET)

    injected_tag = _c(" [INJECTED FRAUD]", BOLD, RED) if txn.is_injected_fraud else ""

    if prediction == "fraud":
        status_icon = _c("🚨 FRAUD", BOLD, RED)
    else:
        status_icon = _c("✅ LEGIT", GREEN)

    print(
        f"  {_c(f'#{idx:>5}', DIM)}  "
        f"{_c(txn.transaction_id, CYAN)}  "
        f"{status_icon}  "
        f"prob={_c(f'{probability:.4f}', colour)}  "
        f"risk={_c(risk_tier, colour)}  "
        f"amt=${txn.features.get('Amount', 0):.2f}"
        f"{injected_tag}"
    )


def _print_summary(stats: dict) -> None:
    """Print a rolling summary banner every N transactions."""
    total = stats["total"]
    if total == 0:
        return
    fraud_rate = stats["fraud_detected"] / total * 100
    injected_rate = stats["injected_flagged"] / max(stats["injected_total"], 1) * 100
    avg_ms = stats["total_latency_ms"] / total

    print()
    print(_c("─" * 70, DIM))
    print(
        f"  📊 ROLLING SUMMARY  |  "
        f"Processed: {_c(str(total), BOLD)}  |  "
        f"Fraud Rate: {_c(f'{fraud_rate:.1f}%', YELLOW)}  |  "
        f"Injected Caught: {_c(f'{injected_rate:.0f}%', GREEN)}  |  "
        f"Avg Latency: {_c(f'{avg_ms:.0f}ms', CYAN)}"
    )
    print(_c("─" * 70, DIM))
    print()


# ── Main simulator loop ───────────────────────────────────────────────────────

def run_simulation(
    api_url: str = "http://localhost:8000",
    tps: float = 2.0,
    duration: float = 120.0,
    fraud_rate: float = 0.05,
    seed: int = 42,
) -> None:
    """
    Stream synthetic transactions to the fraud detection API.

    Parameters
    ----------
    api_url:    Base URL of the prediction API.
    tps:        Transactions per second to generate.
    duration:   Total simulation duration in seconds (0 = run forever).
    fraud_rate: Fraction of transactions that are injected frauds.
    seed:       Random seed for reproducibility.
    """
    rng = random.Random(seed)
    client = FraudAPIClient(api_url)
    interval = 1.0 / max(tps, 0.1)
    summary_every = max(1, int(tps * 20))  # print summary every ~20 seconds

    # ── Pre-flight check ──────────────────────────────────────────────────────
    print()
    print(_c("=" * 70, BOLD, CYAN))
    print(_c("  💳  REAL-TIME FRAUD DETECTION SIMULATOR", BOLD, CYAN))
    print(_c("=" * 70, BOLD, CYAN))
    print(f"  API     : {api_url}")
    print(f"  TPS     : {tps}")
    print(f"  Duration: {'∞' if duration == 0 else f'{duration}s'}")
    print(f"  Fraud % : {fraud_rate * 100:.1f}%  (injected into stream)")
    print(_c("=" * 70, BOLD, CYAN))
    print()

    logger.info("Checking API health at %s ...", api_url)
    if not client.health_check():
        logger.error(
            "API is not reachable. Start it with:\n"
            "  uvicorn api.main:app --host 0.0.0.0 --port 8000"
        )
        sys.exit(1)
    logger.info("API is healthy. Starting transaction stream...\n")

    stats: dict = {
        "total": 0,
        "fraud_detected": 0,
        "errors": 0,
        "injected_total": 0,
        "injected_flagged": 0,
        "total_latency_ms": 0.0,
    }

    sim_start = time.time()

    try:
        while True:
            # Duration guard
            if duration > 0 and (time.time() - sim_start) >= duration:
                break

            # Decide if this transaction is an injected fraud
            force_fraud = rng.random() < fraud_rate
            txn = SyntheticTransaction.generate(rng, sim_start, force_fraud=force_fraud)

            # Send to API and time it
            t0 = time.perf_counter()
            result = client.predict(txn.features)
            latency_ms = (time.perf_counter() - t0) * 1000

            stats["total"] += 1
            if result is None:
                stats["errors"] += 1
                logger.warning("Skipping transaction %s — no API response", txn.transaction_id)
            else:
                stats["total_latency_ms"] += latency_ms
                if result.get("prediction") == "fraud":
                    stats["fraud_detected"] += 1
                if force_fraud:
                    stats["injected_total"] += 1
                    if result.get("prediction") == "fraud":
                        stats["injected_flagged"] += 1

                _print_result(txn, result, stats["total"])

            # Print rolling summary
            if stats["total"] % summary_every == 0:
                _print_summary(stats)

            # Wait before next transaction
            time.sleep(interval)

    except KeyboardInterrupt:
        print()
        logger.info("Simulation stopped by user (Ctrl+C).")

    # ── Final report ──────────────────────────────────────────────────────────
    elapsed = time.time() - sim_start
    total = stats["total"]
    print()
    print(_c("=" * 70, BOLD))
    print(_c("  📋  FINAL SIMULATION REPORT", BOLD))
    print(_c("=" * 70, BOLD))
    print(f"  Duration           : {elapsed:.1f}s")
    print(f"  Transactions sent  : {total}")
    print(f"  Fraud detected     : {stats['fraud_detected']}  ({stats['fraud_detected']/max(total,1)*100:.1f}%)")
    print(f"  Injected frauds    : {stats['injected_total']}")
    print(f"  Injected caught    : {stats['injected_flagged']}  ({stats['injected_flagged']/max(stats['injected_total'],1)*100:.0f}% recall)")
    print(f"  API errors         : {stats['errors']}")
    if total > 0:
        print(f"  Avg latency        : {stats['total_latency_ms']/total:.1f}ms")
        print(f"  Effective TPS      : {total/elapsed:.2f}")
    print(_c("=" * 70, BOLD))
    print()


# ── CLI entry point ───────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Real-time transaction simulator for the Fraud Detection API",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--api-url",    default="http://localhost:8000", help="API base URL")
    p.add_argument("--tps",        type=float, default=2.0,  help="Transactions per second")
    p.add_argument("--duration",   type=float, default=120.0, help="Run for N seconds (0=infinite)")
    p.add_argument("--fraud-rate", type=float, default=0.05, help="Fraction of injected frauds [0,1]")
    p.add_argument("--seed",       type=int,   default=42,   help="Random seed")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_simulation(
        api_url=args.api_url,
        tps=args.tps,
        duration=args.duration,
        fraud_rate=args.fraud_rate,
        seed=args.seed,
    )

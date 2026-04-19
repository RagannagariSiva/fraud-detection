"""
src/monitoring/drift_detector.py
==================================
Production Data Drift Detector

Monitors whether the feature distributions seen in production have shifted
away from what the model was trained on. When they have, predictions become
unreliable — even if the model itself has not changed.

Two complementary tests are used for each feature:

1. **Population Stability Index (PSI)**
   Bins the training distribution and measures how much the live distribution
   deviates. A widely-used industry standard in credit risk monitoring.

   PSI < 0.10  → no significant change
   PSI < 0.25  → moderate change, worth watching
   PSI ≥ 0.25  → significant change — investigate and likely retrain

2. **Kolmogorov-Smirnov two-sample test**
   A non-parametric test that compares the actual empirical distributions
   directly, rather than relying on binning. Catches changes that PSI might
   miss because they fall within a single bin.

Typical usage
-------------
    # At training time — run once, save the baseline
    detector = DriftDetector.from_training_data(X_train_df)
    detector.save("models/drift_baseline.json")

    # In production — run periodically against a recent batch
    detector = DriftDetector.load("models/drift_baseline.json")
    report = detector.check(live_batch_df)

    if report.overall_drift_detected:
        trigger_retraining_pipeline()

CLI
---
    python src/monitoring/drift_detector.py \\
        --baseline data/processed/X_train.csv \\
        --current  data/incoming/last_24h.csv \\
        --report   reports/drift_report.json
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


# ── Configuration ──────────────────────────────────────────────────────────────


@dataclass
class DriftConfig:
    """Thresholds and settings that control how sensitive drift detection is."""

    psi_warning: float = 0.10  # Moderate drift: flag for investigation
    psi_critical: float = 0.25  # Significant drift: recommend retrain
    ks_alpha: float = 0.01  # KS test significance level (p < alpha → drift)
    min_sample_size: int = 200  # Smaller batches are statistically unreliable
    n_bins: int = 10  # Histogram bins for PSI calculation
    monitored_features: list[str] | None = None  # None = all numeric features


# ── Result types ───────────────────────────────────────────────────────────────


@dataclass
class FeatureDriftResult:
    """Drift test results for a single feature."""

    feature: str
    psi: float
    psi_level: str  # "none" | "warning" | "critical"
    ks_statistic: float
    ks_p_value: float
    ks_drift_detected: bool
    baseline_mean: float
    current_mean: float
    mean_shift_pct: float  # Percentage change in mean relative to baseline


@dataclass
class DriftReport:
    """Full drift detection report for one production batch."""

    timestamp: str
    baseline_size: int
    current_size: int
    n_features_tested: int
    n_features_drifted: int  # PSI > warning OR KS significant
    n_features_critical: int  # PSI > critical
    overall_drift_detected: bool
    recommendation: str
    feature_results: list[FeatureDriftResult] = field(default_factory=list)
    prediction_drift: dict | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def summary(self) -> str:
        """Return a formatted text summary for logging and CLI output."""
        width = 62
        lines = [
            f"╔{'═' * width}╗",
            f"║  DRIFT DETECTION REPORT{'':>{width - 24}}║",
            f"╠{'═' * width}╣",
            f"║  Timestamp        : {self.timestamp:<{width - 21}}║",
            f"║  Baseline samples : {self.baseline_size:>{10},}{'':>{width - 32}}║",
            f"║  Current samples  : {self.current_size:>{10},}{'':>{width - 32}}║",
            f"║  Features tested  : {self.n_features_tested:<{width - 21}}║",
            f"║  Features drifted : {self.n_features_drifted:<{width - 21}}║",
            f"║  Critical drift   : {self.n_features_critical:<{width - 21}}║",
            f"║  Overall drift    : {'⚠️  YES' if self.overall_drift_detected else '✅ NO':<{width - 21}}║",
            f"╠{'═' * width}╣",
            f"║  {self.recommendation[: width - 2]:<{width - 2}}║",
            f"╚{'═' * width}╝",
        ]

        drifted = [r for r in self.feature_results if r.psi > 0.10 or r.ks_drift_detected]
        if drifted:
            lines.append("\nDrifted features (sorted by PSI descending):")
            for r in sorted(drifted, key=lambda x: x.psi, reverse=True):
                lines.append(
                    f"  {r.feature:<8}  PSI={r.psi:.4f} [{r.psi_level.upper():<8}]"
                    f"  KS p={r.ks_p_value:.4f}"
                    f"  mean shift={r.mean_shift_pct:+.1f}%"
                )
        return "\n".join(lines)


# ── Core detector ──────────────────────────────────────────────────────────────


class DriftDetector:
    """
    Compares live feature distributions against a saved training baseline.

    The baseline is computed from the training DataFrame using histograms
    and summary statistics. It is serialised to JSON so it can be loaded
    in production without needing the original training data.

    Parameters
    ----------
    baseline_stats : dict[str, dict]
        Pre-computed per-feature statistics produced by
        :meth:`from_training_data`.
    config : DriftConfig, optional
        Sensitivity thresholds. Defaults to :class:`DriftConfig`.
    """

    def __init__(
        self,
        baseline_stats: dict[str, dict],
        config: DriftConfig | None = None,
    ) -> None:
        self._baseline = baseline_stats
        self._cfg = config or DriftConfig()
        logger.info("DriftDetector ready — tracking %d features", len(baseline_stats))

    # ── Constructors ───────────────────────────────────────────────────────────

    @classmethod
    def from_training_data(
        cls,
        X_train: pd.DataFrame,
        config: DriftConfig | None = None,
    ) -> DriftDetector:
        """
        Compute and store baseline statistics from the training DataFrame.

        Call this once immediately after training. The resulting detector
        should be saved with :meth:`save` and deployed alongside the model.

        Parameters
        ----------
        X_train : pd.DataFrame
            Training feature matrix (before SMOTE, using the original split).
        config : DriftConfig, optional
            Override default thresholds.
        """
        cfg = config or DriftConfig()
        numeric_cols = X_train.select_dtypes(include=np.number).columns.tolist()
        features = cfg.monitored_features or numeric_cols

        baseline_stats: dict[str, dict] = {}
        for feat in features:
            if feat not in X_train.columns:
                continue
            col = X_train[feat].dropna()
            counts, edges = np.histogram(col.values, bins=cfg.n_bins)
            baseline_stats[feat] = {
                "mean": float(col.mean()),
                "std": float(col.std()),
                "min": float(col.min()),
                "max": float(col.max()),
                "p25": float(col.quantile(0.25)),
                "p50": float(col.quantile(0.50)),
                "p75": float(col.quantile(0.75)),
                # Store the actual histogram for PSI — not a parametric approximation
                "hist_counts": counts.tolist(),
                "hist_edges": edges.tolist(),
                "n": int(len(col)),
                # Store a sample of raw values for the KS test so we compare
                # empirical distributions, not normal approximations
                "ks_sample": col.sample(n=min(2000, len(col)), random_state=42).tolist(),
            }

        logger.info(
            "Baseline statistics computed for %d features (n=%d)",
            len(baseline_stats),
            len(X_train),
        )
        return cls(baseline_stats, config=cfg)

    @classmethod
    def load(cls, path: str) -> DriftDetector:
        """Load a previously saved detector from a JSON file."""
        with open(path) as f:
            data = json.load(f)
        cfg_dict = data.get("config", {})
        # DriftConfig.monitored_features may be None — handle gracefully
        cfg = DriftConfig(
            **{k: v for k, v in cfg_dict.items() if k in DriftConfig.__dataclass_fields__}
        )
        return cls(data["baseline_stats"], config=cfg)

    def save(self, path: str) -> None:
        """Persist the detector to a JSON file for production deployment."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "baseline_stats": self._baseline,
            "config": asdict(self._cfg),
            "created_at": datetime.now(tz=timezone.utc).isoformat(),
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        logger.info("DriftDetector saved → %s", path)

    # ── Main interface ─────────────────────────────────────────────────────────

    def check(
        self,
        current: pd.DataFrame,
        prediction_probs: np.ndarray | None = None,
    ) -> DriftReport:
        """
        Run drift detection on a batch of live transactions.

        Parameters
        ----------
        current : pd.DataFrame
            A recent batch of production transactions, with the same feature
            columns that were used during training.
        prediction_probs : np.ndarray, optional
            Model output probabilities for this batch. When provided, the
            report includes prediction drift statistics — useful for catching
            model miscalibration even before labels arrive.

        Returns
        -------
        DriftReport
            Per-feature results and an overall drift verdict.
        """
        n_current = len(current)

        if n_current < self._cfg.min_sample_size:
            logger.warning(
                "Drift check: batch size %d is below recommended minimum %d. "
                "Results may be statistically unreliable.",
                n_current,
                self._cfg.min_sample_size,
            )

        features_to_check = [
            f
            for f in (self._cfg.monitored_features or self._baseline.keys())
            if f in current.columns and f in self._baseline
        ]

        feature_results: list[FeatureDriftResult] = [
            self._check_feature(feat, current[feat].dropna().values) for feat in features_to_check
        ]

        n_drifted = sum(
            1 for r in feature_results if r.psi > self._cfg.psi_warning or r.ks_drift_detected
        )
        n_critical = sum(1 for r in feature_results if r.psi > self._cfg.psi_critical)

        # Overall drift: any critical feature, or >= 15% of features drifting
        drift_threshold = max(3, int(len(feature_results) * 0.15))
        overall_drift = n_critical > 0 or n_drifted >= drift_threshold

        if n_critical > 0:
            recommendation = (
                f"⚠️  RETRAIN RECOMMENDED — {n_critical} feature(s) show critical "
                f"drift (PSI ≥ {self._cfg.psi_critical}). Model reliability is compromised."
            )
        elif n_drifted > 0:
            recommendation = (
                f"Monitor closely — {n_drifted} feature(s) show moderate drift. "
                "Schedule a retrain if this persists."
            )
        else:
            recommendation = "✅ No significant drift detected. Model is stable."

        baseline_n = max((v["n"] for v in self._baseline.values()), default=0)

        report = DriftReport(
            timestamp=datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
            baseline_size=baseline_n,
            current_size=n_current,
            n_features_tested=len(feature_results),
            n_features_drifted=n_drifted,
            n_features_critical=n_critical,
            overall_drift_detected=overall_drift,
            recommendation=recommendation,
            feature_results=feature_results,
            prediction_drift=(
                self._check_prediction_drift(prediction_probs)
                if prediction_probs is not None
                else None
            ),
        )

        if overall_drift:
            logger.warning(report.summary())
        else:
            logger.info(
                "Drift check passed — %d/%d features show drift",
                n_drifted,
                len(feature_results),
            )

        return report

    # ── Per-feature logic ──────────────────────────────────────────────────────

    def _check_feature(self, feature: str, current_vals: np.ndarray) -> FeatureDriftResult:
        baseline = self._baseline[feature]

        # PSI: re-bin current values using training histogram edges
        edges = np.array(baseline["hist_edges"])
        base_counts = np.array(baseline["hist_counts"])
        cur_counts, _ = np.histogram(current_vals, bins=edges)

        psi = _compute_psi(base_counts, cur_counts)

        if psi < self._cfg.psi_warning:
            psi_level = "none"
        elif psi < self._cfg.psi_critical:
            psi_level = "warning"
        else:
            psi_level = "critical"

        # KS test: compare against the actual saved baseline sample,
        # not a parametric approximation of it
        baseline_sample = np.array(baseline["ks_sample"])
        ks_stat, ks_p = stats.ks_2samp(baseline_sample, current_vals[: len(baseline_sample)])

        current_mean = float(np.mean(current_vals))
        baseline_mean = float(baseline["mean"])
        denom = max(abs(baseline_mean), 1e-9)
        mean_shift_pct = (current_mean - baseline_mean) / denom * 100.0

        return FeatureDriftResult(
            feature=feature,
            psi=round(psi, 6),
            psi_level=psi_level,
            ks_statistic=round(float(ks_stat), 6),
            ks_p_value=round(float(ks_p), 6),
            ks_drift_detected=bool(ks_p < self._cfg.ks_alpha),
            baseline_mean=round(baseline_mean, 6),
            current_mean=round(current_mean, 6),
            mean_shift_pct=round(mean_shift_pct, 2),
        )

    def _check_prediction_drift(self, probs: np.ndarray) -> dict:
        """Summarise the model's output distribution for this batch."""
        return {
            "mean_probability": round(float(np.mean(probs)), 6),
            "std_probability": round(float(np.std(probs)), 6),
            "fraud_rate_at_040": round(float((probs >= 0.40).mean()), 6),
            "fraud_rate_at_050": round(float((probs >= 0.50).mean()), 6),
            "p90_probability": round(float(np.percentile(probs, 90)), 6),
        }


# ── PSI helper ─────────────────────────────────────────────────────────────────


def _compute_psi(expected: np.ndarray, actual: np.ndarray, epsilon: float = 1e-7) -> float:
    """
    Population Stability Index.

    PSI = Σ (actual_i − expected_i) × ln(actual_i / expected_i)

    Clip to epsilon before taking the log to avoid division by zero.
    Both arrays are renormalised to sum to 1 so different sample sizes
    do not affect the result.
    """
    exp = np.clip(expected.astype(float), epsilon, None)
    act = np.clip(actual.astype(float), epsilon, None)
    exp /= exp.sum()
    act /= act.sum()
    return float(np.sum((act - exp) * np.log(act / exp)))


# ── CLI ────────────────────────────────────────────────────────────────────────


def _main() -> None:
    import argparse
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    parser = argparse.ArgumentParser(
        description="Detect distribution drift between training and live data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--baseline", required=True, help="Training data CSV path")
    parser.add_argument("--current", required=True, help="Live/incoming data CSV path")
    parser.add_argument(
        "--report",
        default="reports/drift_report.json",
        help="Output path for the JSON drift report",
    )
    parser.add_argument(
        "--save-detector", default=None, help="Persist the fitted detector to this JSON path"
    )
    args = parser.parse_args()

    baseline_df = pd.read_csv(args.baseline)
    current_df = pd.read_csv(args.current)

    detector = DriftDetector.from_training_data(baseline_df)
    if args.save_detector:
        detector.save(args.save_detector)

    report = detector.check(current_df)
    print(report.summary())

    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    with open(args.report, "w") as f:
        f.write(report.to_json())
    print(f"\nFull report written → {args.report}")

    sys.exit(1 if report.overall_drift_detected else 0)


if __name__ == "__main__":
    _main()

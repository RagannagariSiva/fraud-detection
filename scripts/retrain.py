"""
scripts/retrain.py
===================
Automated Model Retraining Script

Designed for use in:
  - Scheduled cron jobs (retrain weekly)
  - CI/CD pipelines triggered by data drift alerts
  - Manual retraining when new labelled data arrives

What it does
------------
1. Checks for drift between training baseline and incoming data (optional)
2. Runs the full training pipeline via main.py
3. Evaluates new model against the currently deployed model
4. Promotes new model only if it improves on the primary metric (PR-AUC)
5. Writes a retraining report to reports/retraining_report.json
6. Optionally sends a Slack/webhook notification with the result

Usage
-----
    # Simple retrain (always retrain, no comparison):
    python scripts/retrain.py

    # Retrain only if drift detected:
    python scripts/retrain.py --check-drift

    # Retrain and promote only if PR-AUC improves by ≥ 0.005:
    python scripts/retrain.py --min-improvement 0.005

    # Dry run (run pipeline but don't replace model files):
    python scripts/retrain.py --dry-run

    # With Slack notification:
    python scripts/retrain.py --slack-webhook https://hooks.slack.com/...
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
CONFIG_PATH = "config/config.yaml"
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")
RETRAIN_REPORT = REPORT_DIR / "retraining_report.json"
BACKUP_DIR = MODEL_DIR / "backup"


# ── Helpers ───────────────────────────────────────────────────────────────────


def _load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _load_current_metrics(model_name: str = "xgboost_model") -> dict:
    """Load metrics from the currently deployed model's metadata file."""
    path = MODEL_DIR / f"{model_name}_metadata.json"
    if not path.exists():
        logger.warning("No existing model metadata found at %s", path)
        return {}
    with open(path) as f:
        return json.load(f)


def _backup_current_models() -> None:
    """Back up current model files before retraining."""
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    backup = BACKUP_DIR / ts
    backup.mkdir()
    for pkl in MODEL_DIR.glob("*.pkl"):
        shutil.copy2(pkl, backup / pkl.name)
    for meta in MODEL_DIR.glob("*_metadata.json"):
        shutil.copy2(meta, backup / meta.name)
    logger.info("Current models backed up → %s", backup)


def _restore_backup(backup_ts: str) -> None:
    """Restore models from a backup timestamp directory."""
    backup = BACKUP_DIR / backup_ts
    if not backup.exists():
        logger.error("Backup not found: %s", backup)
        return
    for f in backup.glob("*"):
        shutil.copy2(f, MODEL_DIR / f.name)
    logger.info("Models restored from backup: %s", backup)


def _check_drift(cfg: dict) -> bool:
    """
    Run drift detection between training data and the most recent data batch.

    Returns True if significant drift is detected (recommend retrain).
    """
    try:
        import pandas as pd

        from src.monitoring.drift_detector import DriftDetector

        baseline_path = cfg["data"]["raw_path"]
        # In production this would be a separate rolling incoming data file
        # For this demo, compare first half vs second half of training data
        df = pd.read_csv(baseline_path)
        midpoint = len(df) // 2
        baseline_df = df.iloc[:midpoint]
        current_df = df.iloc[midpoint:]

        detector = DriftDetector.from_training_data(baseline_df)
        report = detector.check(current_df)
        logger.info(report.summary())
        return report.overall_drift_detected
    except Exception as exc:
        logger.warning("Drift check failed: %s — proceeding with retrain.", exc)
        return False


def _send_notification(webhook_url: str, payload: dict) -> None:
    """Send a Slack-compatible webhook notification."""
    try:
        import requests

        status_emoji = "✅" if payload.get("promoted") else "⚠️"
        text = (
            f"{status_emoji} *FraudGuard ML — Retraining Complete*\n"
            f"• New PR-AUC: `{payload.get('new_pr_auc', 'N/A')}`\n"
            f"• Old PR-AUC: `{payload.get('old_pr_auc', 'N/A')}`\n"
            f"• Model promoted: `{payload.get('promoted', False)}`\n"
            f"• Duration: `{payload.get('duration_seconds', 0):.0f}s`\n"
            f"• Reason: {payload.get('reason', '')}"
        )
        requests.post(webhook_url, json={"text": text}, timeout=5)
        logger.info("Slack notification sent.")
    except Exception as exc:
        logger.warning("Failed to send notification: %s", exc)


# ── Main retraining logic ─────────────────────────────────────────────────────


def retrain(
    check_drift: bool = False,
    min_improvement: float = 0.0,
    dry_run: bool = False,
    slack_webhook: str | None = None,
) -> dict:
    """
    Run the full retraining pipeline.

    Parameters
    ----------
    check_drift     : Skip retraining if no drift detected.
    min_improvement : Minimum PR-AUC gain required to promote new model.
    dry_run         : Run pipeline but do not replace production model files.
    slack_webhook   : Optional Slack webhook URL for result notification.

    Returns
    -------
    dict
        Summary of the retraining run with metrics and decision.
    """
    t_start = time.time()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = _load_config()

    report: dict = {
        "started_at": datetime.utcnow().isoformat() + "Z",
        "config_path": CONFIG_PATH,
        "check_drift": check_drift,
        "min_improvement": min_improvement,
        "dry_run": dry_run,
        "drift_detected": None,
        "retrain_triggered": False,
        "old_pr_auc": None,
        "new_pr_auc": None,
        "promoted": False,
        "reason": "",
        "duration_seconds": 0,
    }

    # ── Step 1: Check drift ────────────────────────────────────────────────────
    if check_drift:
        logger.info("Step 1/4: Checking for data drift...")
        drift_detected = _check_drift(cfg)
        report["drift_detected"] = drift_detected
        if not drift_detected:
            report["reason"] = "No significant drift detected — retraining skipped."
            logger.info(report["reason"])
            _write_report(report, t_start)
            return report
        logger.info("Drift detected — triggering retraining.")
    else:
        logger.info("Step 1/4: Drift check skipped (--check-drift not set).")

    report["retrain_triggered"] = True

    # ── Step 2: Capture current model metrics ─────────────────────────────────
    logger.info("Step 2/4: Recording current model metrics...")
    primary_model = cfg["training"].get("primary_model", "xgboost")
    current_meta = _load_current_metrics(f"{primary_model}_model")
    old_pr_auc = float(current_meta.get("val_pr_auc", 0.0))
    report["old_pr_auc"] = old_pr_auc
    logger.info("Current model PR-AUC: %.4f", old_pr_auc)

    # ── Step 3: Back up and retrain ───────────────────────────────────────────
    if not dry_run:
        logger.info("Step 3/4: Backing up current models...")
        backup_ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        _backup_current_models()
    else:
        logger.info("Step 3/4: DRY RUN — skipping backup.")
        backup_ts = None

    logger.info("Step 3/4: Running training pipeline...")
    try:
        from src.training.pipeline import load_config as _lc
        from src.training.pipeline import run_pipeline, setup_logging

        pipeline_cfg = _lc(CONFIG_PATH)
        setup_logging(pipeline_cfg["training"]["log_path"])
        run_pipeline(pipeline_cfg)
        logger.info("Pipeline complete.")
    except Exception as exc:
        logger.exception("Training pipeline failed: %s", exc)
        if not dry_run and backup_ts:
            logger.info("Restoring backup due to training failure...")
            _restore_backup(backup_ts)
        report["reason"] = f"Training failed: {exc}"
        _write_report(report, t_start)
        if slack_webhook:
            _send_notification(slack_webhook, report)
        return report

    # ── Step 4: Compare and promote ───────────────────────────────────────────
    logger.info("Step 4/4: Evaluating new model...")
    new_meta = _load_current_metrics(f"{primary_model}_model")
    new_pr_auc = float(new_meta.get("val_pr_auc", 0.0))
    report["new_pr_auc"] = new_pr_auc

    improvement = new_pr_auc - old_pr_auc
    logger.info(
        "New PR-AUC: %.4f | Old PR-AUC: %.4f | Δ = %+.4f",
        new_pr_auc,
        old_pr_auc,
        improvement,
    )

    if dry_run:
        report["promoted"] = False
        report["reason"] = (
            f"Dry run — new model NOT promoted (would gain {improvement:+.4f} PR-AUC)."
        )
    elif improvement >= min_improvement:
        report["promoted"] = True
        report["reason"] = (
            f"New model promoted ✅ (PR-AUC: {old_pr_auc:.4f} → {new_pr_auc:.4f}, "
            f"Δ={improvement:+.4f} ≥ min={min_improvement:.4f})"
        )
        logger.info(report["reason"])
    else:
        # Restore old model — new model didn't improve enough
        if backup_ts:
            logger.info(
                "New model NOT promoted (Δ=%.4f < min=%.4f). Restoring previous model.",
                improvement,
                min_improvement,
            )
            _restore_backup(backup_ts)
        report["promoted"] = False
        report["reason"] = (
            f"Previous model retained (Δ={improvement:+.4f} < min={min_improvement:.4f})"
        )

    _write_report(report, t_start)

    if slack_webhook:
        _send_notification(slack_webhook, report)

    logger.info("Retraining complete. Report → %s", RETRAIN_REPORT)
    return report


def _write_report(report: dict, t_start: float) -> None:
    report["duration_seconds"] = round(time.time() - t_start, 1)
    report["completed_at"] = datetime.utcnow().isoformat() + "Z"
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with open(RETRAIN_REPORT, "w") as f:
        json.dump(report, f, indent=2)


# ── CLI ───────────────────────────────────────────────────────────────────────


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Automated model retraining for the FraudGuard ML platform",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--check-drift", action="store_true", help="Only retrain if data drift is detected"
    )
    p.add_argument(
        "--min-improvement",
        type=float,
        default=0.0,
        help="Minimum PR-AUC improvement required to promote new model",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Run pipeline but don't replace production model files",
    )
    p.add_argument(
        "--slack-webhook", default=None, help="Slack incoming webhook URL for result notification"
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    result = retrain(
        check_drift=args.check_drift,
        min_improvement=args.min_improvement,
        dry_run=args.dry_run,
        slack_webhook=args.slack_webhook,
    )
    print(json.dumps(result, indent=2))
    sys.exit(0 if result.get("promoted") or result.get("dry_run") else 1)

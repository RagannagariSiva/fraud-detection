"""
main.py
=======
Top-level entry point for the Credit Card Fraud Detection training pipeline.

Usage
-----
    python main.py                              # run full pipeline
    python main.py --config config/config.yaml # explicit config path

After running
-------------
    models/          — trained model pkl files + scaler + feature_names
    reports/figures/ — all evaluation plots (ROC, PR, confusion matrices, SHAP)
    reports/model_results.csv — metric comparison table
    logs/training.log — full log file

Then start the API:
    uvicorn api.main:app --host 0.0.0.0 --port 8000

Or the dashboard:
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import argparse
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the fraud detection ML pipeline")
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to config YAML (default: config/config.yaml)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Import here so logging is configured before any module-level log calls
    from src.training.pipeline import load_config, run_pipeline, setup_logging

    cfg = load_config(args.config)
    setup_logging(cfg["training"]["log_path"])

    import logging
    logger = logging.getLogger(__name__)
    logger.info("Starting Credit Card Fraud Detection Pipeline")
    logger.info("Config: %s", args.config)

    df_results = run_pipeline(cfg)

    logger.info("\nFinal Results:")
    logger.info("\n%s", df_results.to_string(index=False))
    logger.info("\nPipeline complete. Next steps:")
    logger.info("  API:       uvicorn api.main:app --port 8000")
    logger.info("  Dashboard: streamlit run dashboard/app.py")


if __name__ == "__main__":
    main()

"""
src/training
============
Model training orchestration, MLflow experiment logging, and Optuna tuning.

Public API
----------
run_pipeline    — execute the full end-to-end training workflow
load_config     — load and validate config.yaml
setup_logging   — configure root logger (call once from main.py)
"""

from src.training.pipeline import load_config, run_pipeline, setup_logging

__all__ = ["run_pipeline", "load_config", "setup_logging"]

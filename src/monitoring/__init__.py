"""
src/monitoring
==============
Production monitoring utilities for the FraudGuard ML platform.

Public API
----------
DriftDetector   — detects feature distribution shift between training and live data
DriftConfig     — configuration dataclass for drift thresholds
ModelMonitor    — thread-safe rolling metrics collector (latency, fraud rate, errors)
"""

from src.monitoring.drift_detector import DriftConfig, DriftDetector
from src.monitoring.model_monitor import ModelMonitor

__all__ = ["DriftDetector", "DriftConfig", "ModelMonitor"]

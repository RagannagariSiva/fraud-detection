"""
monitoring/
===========
Top-level monitoring package — fraud alert dispatch and real-time log watching.

This package contains runtime observability tools that run *alongside* the
inference API:

  fraud_alerts.py     — FraudAlertSystem: JSONL alert log + console output

The feature-level monitoring internals (drift detection, model health metrics)
live in ``src/monitoring/`` because they are tightly coupled to training
artifacts and are imported by the FastAPI app. This package is for the
operational layer that wraps the API output.
"""

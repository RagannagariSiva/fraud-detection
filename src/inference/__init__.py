"""
src/inference
=============
Runtime inference utilities: loading trained models, applying preprocessing,
and scoring transactions.

Public API
----------
FraudPredictor     — load a trained model and score transactions
TransactionRequest — Pydantic schema for single-transaction input validation
PredictionResponse — Pydantic schema for the API response contract
"""

from src.inference.predictor import FraudPredictor

# Pydantic schemas are only needed when running the FastAPI service.
# Importing them here would make FraudPredictor unavailable in environments
# without pydantic installed (e.g. lightweight batch scoring scripts).
try:
    from src.inference.schema import PredictionResponse, TransactionRequest

    __all__ = ["FraudPredictor", "TransactionRequest", "PredictionResponse"]
except ImportError:
    __all__ = ["FraudPredictor"]

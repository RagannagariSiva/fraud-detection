"""
src/inference/schema.py
========================
Pydantic v2 request/response contracts for the fraud detection API.

These schemas serve three purposes:
  1. Input validation — FastAPI returns a structured 422 with a clear message
     if any field is missing, the wrong type, or out of the allowed range.
  2. Output contract — callers can rely on the response shape being stable
     across API versions.
  3. Documentation — every field is described and an example is provided,
     so the Swagger UI at /docs is immediately usable without reading code.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class TransactionRequest(BaseModel):
    """
    A single credit card transaction to be scored for fraud.

    V1–V28 are PCA-transformed features from the original Kaggle dataset
    (anonymised for cardholder privacy). The values are typically in [-5, 5].

    Amount and Time are raw values. The API applies the RobustScaler fitted
    during training before passing features to the model — callers should
    send raw, unscaled values.
    """

    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float

    Amount: float = Field(
        ...,
        ge=0.0,
        description="Transaction amount in USD. Must be ≥ 0.",
        examples=[149.62],
    )
    Time: float = Field(
        ...,
        ge=0.0,
        description="Seconds elapsed since the first transaction in the dataset.",
        examples=[406.0],
    )

    @field_validator("Amount")
    @classmethod
    def amount_must_be_non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError("Amount cannot be negative")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "V1": -1.3598,
                "V2": -0.0728,
                "V3": 2.5364,
                "V4": 1.3782,
                "V5": -0.3383,
                "V6": 0.4624,
                "V7": 0.2396,
                "V8": 0.0987,
                "V9": 0.3638,
                "V10": -0.0902,
                "V11": -0.5516,
                "V12": -0.6178,
                "V13": -0.9914,
                "V14": -0.3114,
                "V15": 1.4682,
                "V16": -0.4704,
                "V17": 0.2079,
                "V18": 0.0258,
                "V19": 0.4039,
                "V20": 0.2514,
                "V21": -0.0183,
                "V22": 0.2778,
                "V23": -0.1105,
                "V24": 0.0669,
                "V25": 0.1285,
                "V26": -0.1891,
                "V27": 0.1336,
                "V28": -0.0211,
                "Amount": 149.62,
                "Time": 406.0,
            }
        }
    }


class ShapContribution(BaseModel):
    """A single feature's contribution to a fraud prediction."""

    feature: str = Field(..., description="Feature name")
    shap_value: float = Field(..., description="SHAP contribution to fraud probability")
    feature_value: float = Field(..., description="Raw feature value submitted")


class PredictionExplanation(BaseModel):
    """
    SHAP-based explanation for a single fraud prediction.

    Tells the fraud investigator *why* the model flagged this transaction,
    not just *that* it did. Required for GDPR Article 22 compliance in
    automated decision systems.
    """

    base_value: float = Field(
        ...,
        description="Model's expected output on the training set (prior fraud probability)",
    )
    top_features: list[ShapContribution] = Field(
        ...,
        description="Top 10 features ranked by |SHAP value| — their individual contributions",
    )
    explanation_text: str = Field(
        ...,
        description="Human-readable summary of the main fraud drivers",
    )


class PredictionResponse(BaseModel):
    """
    Response from POST /predict.

    The ``explanation`` field is populated when the model has been loaded with
    SHAP support and ``explain=true`` is passed as a query parameter.
    """

    prediction: str = Field(..., description="'fraud' or 'legitimate'")
    probability: float = Field(..., ge=0.0, le=1.0, description="Fraud probability [0, 1]")
    risk_tier: str = Field(..., description="LOW | MEDIUM | HIGH | CRITICAL")
    threshold_used: float = Field(..., description="Decision threshold applied")
    message: str = Field(..., description="Human-readable risk assessment")
    explanation: PredictionExplanation | None = Field(
        None,
        description="SHAP feature contributions (present when ?explain=true)",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "prediction": "fraud",
                "probability": 0.9234,
                "risk_tier": "CRITICAL",
                "threshold_used": 0.40,
                "message": "High fraud probability. Block and alert immediately.",
                "explanation": None,
            }
        }
    }


class BatchPredictionResponse(BaseModel):
    """Summary response for POST /predict/batch."""

    total_transactions: int = Field(..., description="Number of rows scored")
    fraud_count: int = Field(..., description="Rows classified as fraud")
    legitimate_count: int = Field(..., description="Rows classified as legitimate")
    fraud_rate: float = Field(..., description="fraud_count / total_transactions")
    predictions: list[dict] = Field(
        ...,
        description="Per-row results: probability, prediction, risk_tier",
    )

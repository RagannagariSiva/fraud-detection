"""
api/main.py
===========
Fraud Detection ML — FastAPI Inference Service

Endpoints
---------
GET  /health                  Liveness + readiness with rolling monitor metrics
GET  /info                    Model metadata, version, feature count
GET  /metrics                 Operational KPIs (JSON or Prometheus text)
POST /predict                 Score one transaction; optional SHAP explanation
POST /predict/batch           Score a CSV upload (≤ 30 000 rows)


-----------
- ?explain=true on /predict returns SHAP feature contributions per prediction
- /metrics now includes cost-impact estimate (avg flagged amount)
- Prometheus export covers all rolling counters

Run
---
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

Docs
----
    http://localhost:8000/docs   (Swagger UI)
    http://localhost:8000/redoc  (ReDoc)
"""

from __future__ import annotations

import io
import json
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse

from src.inference.predictor import FraudPredictor
from src.inference.schema import (
    BatchPredictionResponse,
    PredictionExplanation,
    PredictionResponse,
    ShapContribution,
    TransactionRequest,
)
from src.monitoring.model_monitor import ModelMonitor
from monitoring.fraud_alerts import FraudAlertSystem

logger = logging.getLogger(__name__)

# ── Module-level singletons (loaded once at startup) ──────────────────────────

_predictor: FraudPredictor | None = None
_config:    dict[str, Any]        = {}
_monitor:   ModelMonitor          = ModelMonitor()
_alerter:   FraudAlertSystem      = FraudAlertSystem()
_start_time: float                = time.time()

# SHAP explainer — loaded lazily on first ?explain=true request so startup
# time is unaffected for callers that don't need explanations.
_explainer: Any = None
_explainer_loaded: bool = False


def _load_config(path: str = "config/config.yaml") -> dict[str, Any]:
    if Path(path).exists():
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


def _get_explainer(model: Any) -> Any | None:
    """Lazily initialise the SHAP TreeExplainer. Returns None if shap not installed."""
    global _explainer, _explainer_loaded
    if _explainer_loaded:
        return _explainer
    try:
        import shap
        _explainer = shap.TreeExplainer(model)
        logger.info("SHAP TreeExplainer loaded successfully")
    except Exception as exc:
        logger.warning("SHAP unavailable: %s — /predict?explain=true will be skipped", exc)
        _explainer = None
    _explainer_loaded = True
    return _explainer


# ── Application lifespan ──────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model artifacts at startup; log clean shutdown."""
    global _predictor, _config
    _config    = _load_config()
    infer_cfg  = _config.get("inference", {})

    try:
        _predictor = FraudPredictor(
            model_name=infer_cfg.get("model_name", "xgboost_model"),
            model_dir =infer_cfg.get("model_dir",  "models"),
            threshold =float(infer_cfg.get("threshold", 0.40)),
        )
        logger.info(
            "Model loaded: %s | features=%d | threshold=%.4f",
            _predictor.model_name,
            len(_predictor.feature_names),
            _predictor.threshold,
        )
    except FileNotFoundError as exc:
        logger.error("Model load failed: %s", exc)
        logger.error("Run `python main.py` to train the model first.")

    yield  # application runs here

    logger.info("Credit Card Fraud Detection API shutting down cleanly.")


# ── FastAPI application ────────────────────────────────────────────────────────

limiter = Limiter(key_func=get_remote_address)
app = FastAPI(
    title="Credit Card Fraud Detection API",
    description=(
        "Real-time fraud scoring using XGBoost trained on the ULB creditcard dataset.\n\n"
        "- Returns a fraud probability, risk tier, and actionable message.\n"
        "- Add `?explain=true` to `/predict` to receive SHAP feature contributions.\n"
        "- Train the model first: `python main.py`\n"
    ),
    version="2.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:8501"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ── Request timing middleware ─────────────────────────────────────────────────

@app.middleware("http")
async def record_latency(request: Request, call_next):
    t0       = time.perf_counter()
    response = await call_next(request)
    ms       = (time.perf_counter() - t0) * 1_000
    response.headers["X-Process-Time-Ms"] = f"{ms:.2f}"
    return response


# ── Internal helpers ──────────────────────────────────────────────────────────

def _require_model() -> FraudPredictor:
    if _predictor is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Model not loaded. "
                "Run `python main.py` to train it, then restart the API."
            ),
        )
    return _predictor


def _build_explanation(
    predictor: FraudPredictor,
    feature_dict: dict[str, float],
) -> PredictionExplanation | None:
    """
    Compute SHAP values for a single transaction and return a structured
    explanation object.  Returns None gracefully if SHAP is unavailable.
    """
    explainer = _get_explainer(predictor.model)
    if explainer is None:
        return None

    try:
        import numpy as np
        import pandas as pd

        # Build the same array the predictor uses — scaled, correct column order
        arr = predictor._build_feature_array(feature_dict)
        row_df = pd.DataFrame(arr, columns=predictor.feature_names)

        shap_vals = explainer.shap_values(row_df)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]  # fraud class for RF binary classifiers
        contributions = shap_vals[0]

        base = (
            float(explainer.expected_value[1])
            if hasattr(explainer.expected_value, "__len__")
            else float(explainer.expected_value)
        )

        # Rank by absolute SHAP value
        ranked = sorted(
            zip(predictor.feature_names, contributions, arr[0]),
            key=lambda t: abs(t[1]),
            reverse=True,
        )[:10]

        top = [
            ShapContribution(
                feature=feat,
                shap_value=round(float(sv), 6),
                feature_value=round(float(fv), 6),
            )
            for feat, sv, fv in ranked
        ]

        fraud_drivers = [f"{c.feature}({c.shap_value:+.3f})" for c in top if c.shap_value > 0][:3]
        safe_drivers  = [f"{c.feature}({c.shap_value:+.3f})" for c in top if c.shap_value < 0][:3]
        lines = []
        if fraud_drivers:
            lines.append(f"Fraud drivers: {', '.join(fraud_drivers)}")
        if safe_drivers:
            lines.append(f"Legitimate signals: {', '.join(safe_drivers)}")

        return PredictionExplanation(
            base_value=round(base, 6),
            top_features=top,
            explanation_text="\n".join(lines) or "No dominant features identified.",
        )

    except Exception as exc:
        logger.warning("SHAP explanation failed (non-fatal): %s", exc)
        return None


# ══════════════════════════════════════════════════════════════════════════════
#  System endpoints
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/health", summary="Liveness and readiness check", tags=["System"])
async def health() -> dict[str, Any]:
    """
    Returns HTTP 200 when the service can accept predictions.

    The `monitor` block reports rolling 5-minute operational metrics so
    Kubernetes probes and alerting rules can detect degraded performance
    without a separate monitoring call.
    """
    uptime    = int(time.time() - _start_time)
    mon_check = _monitor.health_check()

    return {
        "status":         "ok" if _predictor is not None else "degraded",
        "model_loaded":   _predictor is not None,
        "model_name":     _predictor.model_name if _predictor else None,
        "version":        app.version,
        "uptime_seconds": uptime,
        "monitor": {
            "healthy":         mon_check["healthy"],
            "issues":          mon_check["issues"],
            "fraud_rate_5min": mon_check["fraud_rate_5min"],
            "error_rate_5min": mon_check["error_rate_5min"],
            "latency_p99_ms":  mon_check["latency_p99_ms"],
        },
    }


@app.get("/info", summary="Model metadata and configuration", tags=["System"])
async def info() -> dict[str, Any]:
    """Return the active model's name, threshold, feature count, and training metrics."""
    predictor = _require_model()
    meta_path = Path("models") / f"{predictor.model_name}_metadata.json"
    metadata: dict = {}
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            metadata = json.load(f)
    return {
        "model_name":    predictor.model_name,
        "feature_count": len(predictor.feature_names),
        "threshold":     predictor.threshold,
        "metadata":      metadata,
        "version":       app.version,
        "shap_available": _explainer is not None or not _explainer_loaded,
    }


@app.get("/metrics", summary="Operational metrics for monitoring", tags=["System"])
async def metrics(
    format: str = Query("json", description="Response format: 'json' or 'prometheus'"),
) -> Any:
    """
    Operational metrics for dashboards and alerting systems.

    - `format=json` — structured JSON with all rolling counters
    - `format=prometheus` — Prometheus text exposition for scraping

    Metrics cover: fraud rate, error rate, throughput, latency percentiles,
    and per-tier prediction counts — all over a rolling 5-minute window.
    """
    if format == "prometheus":
        return PlainTextResponse(
            content=_monitor.to_prometheus(),
            media_type="text/plain; version=0.0.4",
        )
    return JSONResponse(content=_monitor.snapshot())


# ══════════════════════════════════════════════════════════════════════════════
#  Prediction endpoints
# ══════════════════════════════════════════════════════════════════════════════

@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Score a single transaction",
    tags=["Prediction"],
)
async def predict(
    request: TransactionRequest,
    explain: bool = Query(
        False,
        description="Set to true to include SHAP feature contributions in the response",
    ),
) -> PredictionResponse:
    """
    Score one credit card transaction and return a fraud probability.

    **Input:** JSON body with V1–V28, Amount (USD, ≥ 0), and Time (seconds).

    **Output:** prediction label, probability, risk tier, and optional SHAP explanation.

    **Risk tiers:**

    | Tier | Probability | Recommended action |
    |------|-------------|-------------------|
    | LOW | < 0.15 | Allow |
    | MEDIUM | 0.15–0.40 | Soft review |
    | HIGH | 0.40–0.70 | Manual review |
    | CRITICAL | ≥ 0.70 | Auto-block |

    Add `?explain=true` to receive SHAP feature contributions showing *why*
    the model assigned this probability.
    """
    predictor = _require_model()
    t0        = time.perf_counter()

    try:
        feature_dict = request.model_dump()
        for k, v in feature_dict.items():
            if abs(float(v)) > 1e6:
                raise ValueError(f"Feature {k} value {v} out of range")
        result       = predictor.predict(feature_dict)
        latency_ms   = (time.perf_counter() - t0) * 1_000

        _monitor.record_prediction(
            probability=result["probability"],
            risk_tier  =result["risk_tier"],
            latency_ms =latency_ms,
        )

        import uuid
        _alerter.process(
            transaction_id=f"TXN-{uuid.uuid4().hex[:12].upper()}",
            result=result,
            amount=float(feature_dict.get("Amount", 0.0)),
        )

        explanation: PredictionExplanation | None = None
        if explain:
            explanation = _build_explanation(predictor, feature_dict)

        return PredictionResponse(
            prediction    =result["prediction"],
            probability   =result["probability"],
            risk_tier     =result["risk_tier"],
            threshold_used=result["threshold_used"],
            message       =result["message"],
            explanation   =explanation,
        )

    except ValueError as exc:
        _monitor.record_prediction(probability=0.0, risk_tier="LOW",
                                    latency_ms=0.0, is_error=True)
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        _monitor.record_prediction(probability=0.0, risk_tier="LOW",
                                    latency_ms=0.0, is_error=True)
        logger.exception("Unhandled error during prediction")
        raise HTTPException(status_code=500, detail="Internal prediction error") from exc


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    summary="Score a CSV file of transactions",
    tags=["Prediction"],
)
async def predict_batch(
    file: UploadFile = File(..., description="CSV file with V1–V28, Amount, Time columns"),
) -> BatchPredictionResponse:
    """
    Upload a CSV and score every row against the live model.

    The CSV must have columns V1–V28, Amount, and Time. Any additional
    columns are preserved in the output but not passed to the model.

    Returns a summary with per-row probabilities, predictions, and risk tiers.

    Maximum 30 000 rows per request.
    """
    predictor = _require_model()

    if not (file.filename or "").endswith(".csv"):
        raise HTTPException(status_code=400, detail="File must have a .csv extension")

    raw = await file.read()
    try:
        df = pd.read_csv(io.StringIO(raw.decode("utf-8")))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {exc}") from exc

    if len(df) > 30_000:
        raise HTTPException(
            status_code=400,
            detail=f"File has {len(df):,} rows. Maximum is 30 000 per request.",
        )

    t0 = time.perf_counter()
    try:
        scored = predictor.predict_batch(df)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Batch prediction error")
        raise HTTPException(status_code=500, detail="Internal batch prediction error") from exc

    latency_ms  = (time.perf_counter() - t0) * 1_000
    fraud_count = int((scored["prediction"] == "fraud").sum())
    total       = len(scored)

    # Record every row in the monitor (amortised latency)
    per_row_ms = latency_ms / max(total, 1)
    for prob, tier in zip(scored["probability"], scored["risk_tier"]):
        _monitor.record_prediction(
            probability=float(prob),
            risk_tier  =str(tier),
            latency_ms =per_row_ms,
        )

    return BatchPredictionResponse(
        total_transactions=total,
        fraud_count       =fraud_count,
        legitimate_count  =total - fraud_count,
        fraud_rate        =round(fraud_count / total, 6) if total > 0 else 0.0,
        predictions       =scored[["probability", "prediction", "risk_tier"]].to_dict(orient="records"),
    )
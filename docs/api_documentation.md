# API Documentation

## Overview

The FraudGuard ML REST API provides real-time and batch fraud scoring for credit
card transactions. It is built with **FastAPI** and served by **Uvicorn**.

- **Base URL (local)**: `http://localhost:8000`
- **Interactive docs (Swagger UI)**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`
- **OpenAPI spec**: `http://localhost:8000/openapi.json`

---

## Quick Start

```bash
# 1. Train the model (required before starting the API)
python main.py

# 2. Start the API server
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# 3. Send a test prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "V1": -1.3598, "V2": -0.0728, "V3": 2.5364,
    "V4": 1.3782, "V5": -0.3383, "V6": 0.4624,
    "V7": 0.2396, "V8": 0.0987, "V9": 0.3638,
    "V10": -0.0902, "V11": -0.5516, "V12": -0.6178,
    "V13": -0.9914, "V14": -0.3114, "V15": 1.4682,
    "V16": -0.4704, "V17": 0.2079, "V18": 0.0258,
    "V19": 0.4039, "V20": 0.2514, "V21": -0.0183,
    "V22": 0.2778, "V23": -0.1105, "V24": 0.0669,
    "V25": 0.1285, "V26": -0.1891, "V27": 0.1336,
    "V28": -0.0211, "Amount": 149.62, "Time": 0.0
  }'
```

---

## Endpoints

### `GET /health`

Liveness and readiness check. Use this for Kubernetes probes.

**Response 200**:
```json
{
  "status": "ok",
  "model_loaded": true,
  "model_name": "xgboost_model",
  "uptime_seconds": 3820
}
```

| Field | Type | Description |
|-------|------|-------------|
| `status` | string | `"ok"` or `"degraded"` (model not loaded) |
| `model_loaded` | bool | Whether the model is ready to score |
| `model_name` | string \| null | Name of the loaded model |
| `uptime_seconds` | int | Seconds since API startup |

---

### `GET /info`

Returns model metadata including feature count, threshold, and training metrics.

**Response 200**:
```json
{
  "model_name": "xgboost_model",
  "feature_count": 30,
  "threshold": 0.40,
  "metadata": {
    "name": "xgboost_model",
    "class": "XGBClassifier",
    "val_pr_auc": 0.882341,
    "val_roc_auc": 0.978652,
    "train_time_seconds": 42.5,
    "params": {
      "n_estimators": "400",
      "learning_rate": "0.05",
      "max_depth": "6"
    }
  }
}
```

---

### `POST /predict`

Score a single transaction for fraud.

**Request body** (`application/json`):

| Field | Type | Required | Constraints | Description |
|-------|------|----------|-------------|-------------|
| `V1` – `V28` | float | ✅ | Any | PCA-transformed anonymised features |
| `Amount` | float | ✅ | `≥ 0` | Transaction amount in USD |
| `Time` | float | ✅ | `≥ 0` | Seconds since dataset start |

**Response 200**:
```json
{
  "prediction": "fraud",
  "probability": 0.9234,
  "risk_tier": "CRITICAL",
  "threshold_used": 0.40,
  "message": "High fraud probability. Block and alert immediately."
}
```

| Field | Type | Description |
|-------|------|-------------|
| `prediction` | string | `"fraud"` or `"legitimate"` |
| `probability` | float [0,1] | Model's estimated fraud probability |
| `risk_tier` | string | `LOW` / `MEDIUM` / `HIGH` / `CRITICAL` |
| `threshold_used` | float | Decision threshold applied |
| `message` | string | Human-readable recommendation |

**Risk Tier Thresholds**:

| Tier | Probability Range | Recommended Action |
|------|------------------|-------------------|
| `LOW` | < 0.15 | Allow — no action |
| `MEDIUM` | 0.15 – 0.40 | Soft flag for review |
| `HIGH` | 0.40 – 0.70 | Route to manual review queue |
| `CRITICAL` | ≥ 0.70 | Block card immediately, notify cardholder |

**Error responses**:

| Code | Meaning |
|------|---------|
| 422 | Invalid input (missing field, wrong type, negative Amount) |
| 503 | Model not loaded — run `python main.py` first |
| 500 | Internal prediction error (check server logs) |

---

### `POST /predict/batch`

Score up to 10,000 transactions from a CSV file upload.

**Request**: `multipart/form-data` with a single `file` field containing a `.csv`

The CSV must have columns matching the training feature set (V1–V28, Amount, Time).
Additional columns are ignored.

**Response 200**:
```json
{
  "total_transactions": 5000,
  "fraud_count": 12,
  "legitimate_count": 4988,
  "fraud_rate": 0.0024,
  "predictions": [
    {"probability": 0.0041, "prediction": "legitimate", "risk_tier": "LOW"},
    {"probability": 0.9712, "prediction": "fraud",      "risk_tier": "CRITICAL"},
    ...
  ]
}
```

**Limits**:
- Maximum 10,000 rows per request
- File must have `.csv` extension
- Must be UTF-8 encoded

---

## Response Headers

| Header | Example | Description |
|--------|---------|-------------|
| `X-Process-Time-Ms` | `3.42` | Server-side processing time in milliseconds |

---

## Python Client Example

```python
import requests

API_URL = "http://localhost:8000"

def predict_transaction(features: dict) -> dict:
    """Score a single transaction."""
    response = requests.post(f"{API_URL}/predict", json=features, timeout=5)
    response.raise_for_status()
    return response.json()

# Example usage
result = predict_transaction({
    "V1": -1.36, "V2": -0.07, "V3": 2.54,
    # ... V4–V28 ...
    "V4": 1.38, "V5": -0.34, "V6": 0.46,
    "V7": 0.24, "V8": 0.10,  "V9": 0.36,
    "V10": -0.09, "V11": -0.55, "V12": -0.62,
    "V13": -0.99, "V14": -0.31, "V15": 1.47,
    "V16": -0.47, "V17": 0.21,  "V18": 0.03,
    "V19": 0.40,  "V20": 0.25,  "V21": -0.02,
    "V22": 0.28,  "V23": -0.11, "V24": 0.07,
    "V25": 0.13,  "V26": -0.19, "V27": 0.13,
    "V28": -0.02, "Amount": 149.62, "Time": 0.0,
})

print(result["prediction"])    # "fraud" or "legitimate"
print(result["probability"])   # e.g. 0.0341
print(result["risk_tier"])     # e.g. "LOW"
```

---

## Configuration

API behaviour is controlled by `config/config.yaml`:

```yaml
inference:
  model_name: xgboost_model   # stem of the .pkl file in models/
  model_dir:  models
  threshold:  0.40             # decision boundary (lower = higher recall)
```

Override the threshold at startup via environment variable:
```bash
FRAUD_THRESHOLD=0.35 uvicorn api.main:app --port 8000
```

---

## Running in Docker

```bash
docker build -t fraudguard-api .
docker run -p 8000:8000 -v $(pwd)/models:/app/models fraudguard-api
```

Or with Docker Compose:
```bash
docker-compose up api
```

# System Architecture

## Overview

FraudGuard ML is a production-style fraud detection platform built around a
single core idea: the boundary between training and serving must be explicit and
tested. Every preprocessing artifact that is created during training (the
scaler, the feature name list, the decision threshold, the drift baseline) is
persisted to disk and loaded verbatim at inference time. This eliminates the
most common cause of silent production failures in ML systems: training-serving
skew.

---

## Component Map

```
┌────────────────────────────────────────────────────────────────────────┐
│                        FraudGuard ML Platform                          │
│                                                                        │
│  OFFLINE TRAINING                     ONLINE SERVING                  │
│  ─────────────────────────────────    ──────────────────────────────── │
│                                                                        │
│  creditcard.csv                        HTTP client / payment network   │
│        │                                         │                    │
│  src/data/loader.py                    POST /predict                  │
│  src/data/preprocessing.py                       │                    │
│        │                               api/main.py (FastAPI)          │
│  src/features/feature_engineering.py             │                    │
│  src/features/resampling.py (SMOTE)   src/inference/predictor.py      │
│        │                                         │                    │
│  src/training/train_model.py           models/xgboost_model.pkl       │
│  src/training/tuning.py (Optuna)       models/scaler.pkl              │
│        │                               models/feature_names.pkl       │
│  MLflow (experiment log)                         │                    │
│  models/*.pkl  ◄──────────────────────  shared artifact store         │
│  models/drift_baseline.json                      │                    │
│        │                               src/monitoring/model_monitor   │
│  src/monitoring/drift_detector         GET /metrics                   │
│  reports/figures/*.png                           │                    │
│                                        monitoring/fraud_alerts.py     │
│                                        logs/fraud_alerts.jsonl        │
│                                                  │                    │
│                                        dashboard/app.py (Streamlit)   │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow

### Training path (offline)

1. **Load** — `src/data/loader.py` reads `creditcard.csv` or generates a
   statistically equivalent synthetic dataset if the file is absent (allows CI
   to pass without the 144 MB Kaggle download).

2. **Clean** — `src/data/preprocessing.py` removes duplicates, imputes any
   nulls with column medians (the real dataset has none, but this makes the
   pipeline robust to extended datasets).

3. **Split** — stratified 70/10/20 train/val/test split. Stratification
   preserves the ~0.17% fraud rate in every partition.

4. **Scale** — `RobustScaler` is fitted on `Amount` and `Time` columns of the
   **training set only** and saved to `models/scaler.pkl`. The val and test
   sets are then transformed using the training scaler — never refitted.
   This is the critical step that prevents training-serving skew.

5. **Feature engineering** — time-of-day features, log-amount, amount z-score,
   and interaction products (V1×V4, V12×V14, V1×V17, |V14|) are added to all
   three splits identically. The final feature name list is saved to
   `models/feature_names.pkl`.

6. **Resampling** — SMOTE is applied **only to the training set** to balance
   the classes before fitting. Applying SMOTE before the split would leak
   synthetic samples into validation/test, inflating all evaluation metrics.

7. **Train** — XGBoost, Random Forest, and a Decision Tree baseline are trained
   and logged to MLflow with full parameter/metric/artifact tracking.

8. **Evaluate** — ROC curves, PR curves, confusion matrices, and feature
   importance plots are generated for every model on the held-out test set.

9. **Threshold tuning** — The primary model's decision threshold is updated to
   the Youden's J optimum on the validation set and saved in the model metadata.

10. **Drift baseline** — `DriftDetector` computes per-feature histogram
    statistics on the training set and saves them to
    `models/drift_baseline.json`. This file is the reference point for all
    future production drift checks.

### Serving path (online)

1. FastAPI loads `FraudPredictor` on startup via the lifespan event. The
   predictor loads the model, scaler, and feature name list — it will refuse to
   start if any of these files is missing, because a partial load would produce
   silently wrong results.

2. `POST /predict` receives a `TransactionRequest` (Pydantic v2), which
   validates that all 30 features are present and that Amount ≥ 0.

3. `FraudPredictor._to_array()` applies the saved `RobustScaler` to Amount and
   Time, then builds a numpy array in the exact column order the model was
   trained on.

4. The model returns a fraud probability. The probability is mapped to a risk
   tier (LOW/MEDIUM/HIGH/CRITICAL) and a human-readable message.

5. Every prediction is recorded in `ModelMonitor`, which maintains rolling
   5-minute windows for fraud rate, error rate, throughput, and latency
   percentiles. These are exposed at `GET /metrics`.

6. High-risk predictions are simultaneously forwarded to `FraudAlertSystem`,
   which appends a JSON line to `logs/fraud_alerts.jsonl` and prints a
   colour-coded console alert.

---

## Directory Reference

| Path | Purpose |
|------|---------|
| `src/data/` | Data loading, cleaning, scaling, splitting |
| `src/features/` | Feature engineering and SMOTE resampling |
| `src/training/` | Model training, MLflow logging, Optuna tuning |
| `src/inference/` | `FraudPredictor` and Pydantic schemas |
| `src/monitoring/` | Drift detection and real-time metrics |
| `src/models/` | Model evaluation, plots, comparison reports |
| `api/` | FastAPI application — prediction endpoints |
| `dashboard/` | Streamlit analytics dashboard |
| `simulation/` | Synthetic transaction stream for load testing |
| `monitoring/` | Standalone fraud alert system |
| `scripts/` | Automated retraining pipeline |
| `models/` | Saved model artifacts (generated at runtime) |
| `reports/` | Evaluation plots and metrics CSVs |
| `logs/` | Alert log (JSONL), training log |
| `mlruns/` | MLflow experiment store (generated at runtime) |
| `config/` | YAML configuration consumed by every component |
| `tests/` | Unit and integration tests |
| `docs/` | Architecture, pipeline, API, and design documentation |
| `notebooks/` | EDA and model comparison scripts |

---

## Deployment Topology

### Local development

```
Terminal 1:  uvicorn api.main:app --port 8000 --reload
Terminal 2:  streamlit run dashboard/app.py
Terminal 3:  python simulation/real_time_transactions.py --tps 2
Terminal 4:  mlflow ui --port 5000
```

### Docker Compose (all services)

```
docker compose run --rm train        # Step 1: train
docker compose up api dashboard mlflow   # Step 2: launch
```

Services: `api` (:8000), `dashboard` (:8501), `mlflow` (:5000), `simulator`
(run-once).

### Production Kubernetes sketch

```
Deployment: fraud-api
  - image: fraudguard-ml:v2.0
  - replicas: 3
  - resources: requests cpu=500m memory=512Mi
  - livenessProbe:  GET /health
  - readinessProbe: GET /health
  - env: MLFLOW_TRACKING_URI=http://mlflow-service:5000

CronJob: fraud-retrain
  - schedule: "0 2 * * 1"   # weekly Monday 02:00
  - command: python scripts/retrain.py --check-drift --min-improvement 0.005

Service: fraud-api-svc (ClusterIP + Ingress)
```

---

## Key Design Decisions

### Why XGBoost over neural networks?

The Kaggle creditcard dataset has 30 features and 284 K rows. A gradient
boosted tree is interpretable via SHAP, trains in under two minutes on CPU, and
delivers near-optimal performance on tabular data at this scale. A neural
network would require 10× more infrastructure for negligible gain.

### Why RobustScaler instead of StandardScaler?

`Amount` has extreme right-skew with outliers exceeding $25 000. RobustScaler
uses the IQR rather than variance, making it insensitive to those outliers and
producing more stable scaling at inference time.

### Why PR-AUC as the primary metric instead of accuracy?

With 0.17% fraud, a model that predicts "legitimate" for every transaction
achieves 99.83% accuracy while catching zero fraud. PR-AUC measures the
tradeoff between precision and recall on the minority class and is not inflated
by the massive true-negative count.

### Why threshold 0.40 instead of 0.50?

Fraud asymmetry: missing a fraud (false negative) costs significantly more than
a false block (false positive, which triggers a customer review). Lowering the
threshold from 0.50 to 0.40 recovers ~8% additional fraud at the cost of ~15%
more false positives — an acceptable tradeoff in a real deployment. The
threshold is tuned on the validation set using Youden's J statistic.

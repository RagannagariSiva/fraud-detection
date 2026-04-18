# Fraud Detection ML

A production-grade credit card fraud detection system built end-to-end — from raw transaction data to a live inference API, a real-time analytics dashboard, experiment tracking, and automated drift monitoring.

Click the image below to open the live dashboard:

[![Live Dashboard](imgs/DashBord.png)](https://fraud-detection-988itbtnyczqkfo3fqqk8e.streamlit.app/)

---

## What this project is

Most fraud detection projects stop at training a model and printing accuracy. This one goes all the way to production. It handles every engineering concern that matters when you actually deploy a machine learning model in a real system:

The scaler is fitted only on the training set and saved to disk, then loaded at inference time. This prevents training-serving skew, which is the most common silent failure in deployed ML systems. If the scaler sees test data during fitting, the model learns a subtly wrong representation and fails in production without any error message.

The feature column order is saved alongside the model. Scikit-learn models are sensitive to the order of input columns, and Python dictionary iteration order is not guaranteed to be consistent across environments. Saving the column list explicitly eliminates this class of bug entirely.

The decision threshold is calibrated on the validation set using Youden's J statistic instead of the default 0.5. A missed fraud costs roughly $80. A false positive costs roughly $5 in analyst review time. Given that cost asymmetry, a lower threshold catches more fraud at the cost of more false alerts, which is the correct tradeoff for this domain.

SMOTE oversampling is applied only to the training split, never before the split. Applying SMOTE before splitting leaks synthetic minority samples into the test set, which inflates every reported metric and produces a model that appears to perform better than it actually does.

Drift detection compares live feature distributions against the training baseline using Population Stability Index and a Kolmogorov-Smirnov test on every feature. PSI is the industry standard in credit risk monitoring. KS catches distribution shifts that PSI misses when changes fall within a single bin boundary.

---

## Dashboard pages

![Dashboard Overview](imgs/DashBord.png)

![Live Prediction](imgs/Local%20host.png)

![Model Analysis](imgs/Machine%20Learning.png)

![Screen 1](imgs/Screen%20short-1.png)

![Screen 2](imgs/Screen%20short-2.png)

![Screen 3](imgs/Screen%20short-3.png)

![Screen 4](imgs/Screen%20short-4.png)

![Screen 5](imgs/Screen%20short-5.png)

![Screen 6](imgs/Screen%20short-6.png)

![Screen 7](imgs/Screen%20short-7.png)

The dashboard has five pages. The Overview page shows total alerts, critical and high-risk counts, average fraud probability, and financial exposure in real time. The Live Prediction page lets you score any transaction manually against the model, with a probability gauge and tier classification. The Model Analysis page shows PR curves, ROC curves, confusion matrices, and SHAP feature importance plots. The Alert Feed page streams live fraud alerts as the simulator runs. The Batch Scoring page accepts a CSV upload and returns predictions for up to 10,000 rows.

---

## Model performance

Evaluated on the held-out test set, which is 20% of the data and was never seen during training or threshold calibration. All numbers come from the real Kaggle dataset, not synthetic data.

| Model | PR-AUC | ROC-AUC | Recall | Precision | F1 |
|---|---|---|---|---|---|
| XGBoost | 0.870 | 0.978 | 0.854 | 0.882 | 0.868 |
| Random Forest | 0.841 | 0.971 | 0.826 | 0.863 | 0.844 |
| Decision Tree | 0.631 | 0.918 | 0.784 | 0.607 | 0.684 |

PR-AUC is the primary metric rather than accuracy because a model that classifies every transaction as legitimate achieves 99.83% accuracy while catching zero fraud. PR-AUC measures the tradeoff between precision and recall on the minority class, which is the correct metric for heavily imbalanced datasets.

XGBoost is the active model. It is loaded at API startup and used for all predictions.

---

## Business impact

Based on test set results at threshold 0.40, using $80 average fraud loss and $5 per false positive review:

| Metric | Value |
|---|---|
| Fraud caught | ~$39,200 |
| Fraud missed | ~$6,800 |
| Review cost | ~$1,100 |
| Net benefit | ~$38,100 |
| ROI | ~3,400% |

---

## System architecture

```
Offline training                          Online serving
─────────────────────────────────         ─────────────────────────────────

creditcard.csv                            HTTP client / payment network
      |                                             |
  src/data/loader.py                          POST /predict
  src/data/preprocessing.py                       |
      |                                    api/main.py  (FastAPI)
  src/features/feature_engineering.py             |
  src/features/resampling.py (SMOTE)       src/inference/predictor.py
      |                                             |
  src/training/train_model.py              models/xgboost_model.pkl
  src/training/tuning.py  (Optuna)         models/scaler.pkl
      |                                    models/feature_names.pkl
  MLflow experiment log                            |
  models/*.pkl  <─────────────────         src/monitoring/model_monitor.py
  models/drift_baseline.json                       |
      |                                    dashboard/app.py  (Streamlit)
  reports/figures/*.png
```

The training pipeline runs in 11 named phases. Each phase logs its start and end so a failed run immediately shows which phase broke. Phase 4 applies SMOTE to the training split only. Phase 9 calibrates the decision threshold on the validation set. Phase 10 generates SHAP explainability plots. Phase 11 calculates the business impact estimate.

The serving path loads the predictor at API startup via FastAPI's lifespan event. The predictor loads the model, scaler, and feature name list and refuses to start if any file is missing, because a partial load produces silently wrong predictions. Every prediction is recorded in the model monitor, which maintains rolling 5-minute windows for fraud rate, error rate, throughput, and latency percentiles.

---

## Project layout

```
fraud-detection/
├── src/
│   ├── data/               Data loading, cleaning, scaling, stratified splits
│   ├── features/           Feature engineering and SMOTE resampling
│   ├── training/           Model training with MLflow logging, Optuna tuning, pipeline
│   ├── inference/          FraudPredictor class and Pydantic request/response schemas
│   ├── models/             Evaluation: metrics, ROC/PR curves, confusion matrix
│   └── monitoring/         Drift detection (PSI + KS) and rolling model health metrics
├── api/                    FastAPI application — prediction endpoints
├── dashboard/              Streamlit analytics interface (5 pages)
├── monitoring/             Fraud alert dispatcher and JSONL event log
├── simulation/             Synthetic transaction stream for load testing the API
├── scripts/
│   ├── retrain.py          Drift-gated automated retraining with model promotion
│   └── evaluate.py         Standalone evaluation on any model and any CSV
├── tests/                  Unit and integration tests (pytest, 39 tests)
├── notebooks/              EDA and model comparison (Jupytext percent-format)
├── docs/                   Architecture, pipeline, API reference, system design
├── config/config.yaml      Single configuration file for all components
├── Makefile                All common operations as make targets
├── Dockerfile              Multi-stage production image
└── docker-compose.yml      Full stack: API, Dashboard, MLflow, Simulator
```

---

## Setup

Requirements: Python 3.10 or higher.

```bash
git clone https://github.com/RagannagariSiva/fraud-detection.git
cd fraud-detection

python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

Dataset: Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it at `data/raw/creditcard.csv`. If the file is absent, the pipeline generates a statistically equivalent synthetic dataset automatically so all tests and CI pass without the download.

---

## Running the project

Train the model, which takes around 2 minutes on CPU:

```bash
python main.py
```

This runs all 11 pipeline phases and writes trained models to `models/`, evaluation plots to `reports/figures/`, and a metric comparison table to `reports/model_results.csv`.

Start the inference API:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

The Swagger UI is at http://localhost:8000/docs

Start the analytics dashboard:

```bash
streamlit run dashboard/app.py
```

The dashboard is at http://localhost:8501

Run the transaction simulator in a new terminal after the API is running:

```bash
python simulation/real_time_transactions.py --tps 2 --duration 0 --fraud-rate 0.05
```

View MLflow experiment history:

```bash
mlflow ui --port 5001
```

Run all services at once with Docker:

```bash
docker compose run --rm train
docker compose up api dashboard mlflow
```

---

## API reference

| Method | Endpoint | Description |
|---|---|---|
| GET | /health | Liveness check with live fraud rate and P99 latency |
| GET | /info | Model name, threshold, feature count, training metadata |
| GET | /metrics | Operational metrics in JSON or Prometheus text format |
| POST | /predict | Score a single transaction |
| POST | /predict/batch | Score a CSV upload up to 10,000 rows |

Risk tiers returned by the API:

| Tier | Probability | Action |
|---|---|---|
| Low | Below 15% | Allow |
| Medium | 15% to 40% | Soft review |
| High | 40% to 70% | Manual review |
| Critical | 70% and above | Auto-block |

Example request:

```bash
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "V1": -1.3598, "V2": -0.0728, "V3": 2.5364, "V4": 1.3782,
    "V5": -0.3383, "V6":  0.4624, "V7": 0.2396, "V8": 0.0987,
    "V9":  0.3638, "V10":-0.0902, "V11":-0.5516, "V12":-0.6178,
    "V13":-0.9914, "V14":-0.3114, "V15": 1.4682, "V16":-0.4704,
    "V17": 0.2079, "V18": 0.0258, "V19": 0.4039, "V20": 0.2514,
    "V21":-0.0183, "V22": 0.2778, "V23":-0.1105, "V24": 0.0669,
    "V25": 0.1285, "V26":-0.1891, "V27": 0.1336, "V28":-0.0211,
    "Amount": 149.62,
    "Time": 406.0
  }'
```

Example response:

```json
{
  "prediction": "legitimate",
  "probability": 0.032,
  "risk_tier": "LOW",
  "threshold_used": 0.40,
  "message": "Transaction appears normal. No action required."
}
```

Add `?explain=true` to the request to receive SHAP feature contributions alongside the prediction.

---

## Feature engineering

The V1 through V28 columns in the Kaggle dataset are PCA-transformed components from the original transaction data, anonymised for cardholder privacy. The pipeline adds several derived features:

- `hour_of_day` and `is_night` — fraud skews heavily toward off-hours in card-present fraud studies
- `log_amount` — compresses the extreme right tail of transaction amounts
- `amount_z` — median absolute deviation z-score for amount outlier detection
- `V1_V4`, `V12_V14`, `V1_V17` — interaction products between the highest-importance features
- `V14_abs` — absolute value of V14, which is the single strongest fraud predictor in published analyses of this dataset

These features are applied identically at training time and at inference time. The predictor applies feature engineering internally so the API caller only needs to send the 30 raw fields.

---

## Monitoring and drift detection

Drift is checked by comparing live feature distributions against the training baseline:

| PSI value | Status |
|---|---|
| Below 0.10 | No significant change |
| 0.10 to 0.25 | Moderate change, monitor closely |
| Above 0.25 | Significant change, retraining recommended |

Run a manual drift check:

```bash
python scripts/retrain.py --check-drift --dry-run
```

Run drift-gated automated retraining:

```bash
python scripts/retrain.py --check-drift --min-improvement 0.005
```

This skips retraining if no drift is detected and restores the previous model if the new one does not improve PR-AUC by at least 0.005.

Live metrics are available at GET /metrics while the API is running, covering fraud rate, error rate, throughput, and P50, P95, P99 latency over a rolling 5-minute window.

---

## Tests

```bash
pytest tests/ -v
```

39 tests covering preprocessing, feature engineering, the predictor, drift detection, monitoring, API endpoints, and a full end-to-end pipeline integration test. The integration test trains a small model from scratch, verifies all artifacts are created, loads the predictor, and scores a transaction without any external dependencies.

---

## Common commands

```bash
make train        # train the full pipeline
make api          # start the inference API
make dashboard    # start the Streamlit dashboard
make simulate     # run the transaction simulator
make test         # run the test suite with coverage
make retrain      # drift-gated automated retraining
make docker-up    # start all services in Docker
make help         # full command reference
```

---

## Tech stack

Python 3.10+ · scikit-learn · XGBoost · imbalanced-learn · FastAPI · Pydantic v2 · Streamlit · MLflow · Optuna · SHAP · scipy · pytest · ruff · Docker

---

## Dataset

[Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) by the ULB Machine Learning Group, released under the Open Database License.

284,807 transactions collected over two days in September 2013 by European cardholders. 492 are fraudulent, which is 0.172% of all transactions. Features V1 through V28 are the principal components of the original transaction data. Amount and Time are the raw values. The dataset cannot be redistributed — download it directly from Kaggle and place it at `data/raw/creditcard.csv`.
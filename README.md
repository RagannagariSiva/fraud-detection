# Fraud Detection ML

A production-grade credit card fraud detection system built end-to-end — from raw transaction data to a live inference API, an analytics dashboard, experiment tracking, and automated monitoring.

The dataset used is the [ULB Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) from Kaggle: 284,807 transactions over 48 hours, with 492 confirmed fraud cases (0.172% of all transactions).

---

## What this project covers

Most fraud detection examples stop at training a model and printing metrics. This one goes further — it handles every engineering concern that comes up when you actually deploy a model:

- The scaler is fitted only on the training set and saved to disk, then loaded at inference time. This prevents training-serving skew, which is the most common silent failure in deployed ML.
- The feature column order is saved alongside the model. Scikit-learn models are sensitive to column ordering, and dictionary iteration order is not reliable.
- The decision threshold is calibrated on the validation set using Youden's J statistic, not left at the default 0.5.
- SMOTE resampling is applied only to the training split, not before the split. Applying it before the split leaks synthetic samples into the test set and inflates every metric.
- Drift detection runs PSI and a Kolmogorov-Smirnov test on every feature. PSI is the industry standard in credit risk; KS catches shifts that PSI misses when they fall within a single bin.

---

## Model performance

Evaluated on the held-out test set (20% of data, never seen during training or threshold calibration). Numbers are from the real Kaggle dataset, not synthetic data.

| Model | PR-AUC | ROC-AUC | Recall | Precision | F1 |
|---|---|---|---|---|---|
| XGBoost | 0.870 | 0.978 | 0.854 | 0.882 | 0.868 |
| Random Forest | 0.841 | 0.971 | 0.826 | 0.863 | 0.844 |
| Decision Tree | 0.631 | 0.918 | 0.784 | 0.607 | 0.684 |

**Why PR-AUC and not accuracy?** A model that classifies every transaction as legitimate achieves 99.83% accuracy and catches zero fraud. PR-AUC measures the tradeoff between precision and recall on the minority class and is the correct primary metric for heavily imbalanced datasets.

**Why threshold 0.40 and not 0.50?** A missed fraud costs roughly $80 in lost funds. A false positive costs roughly $5 in customer review time. Given that asymmetry, it is worth catching more fraud at the cost of more false alerts. Youden's J statistic on the validation set converges to around 0.40 for this dataset.

---

## Business impact estimate

Based on the test set results (threshold = 0.40, $80 average fraud loss, $5 per false positive review):

| Metric | Value |
|---|---|
| Fraud caught | ~$39,200 |
| Fraud missed | ~$6,800 |
| Review cost | ~$1,100 |
| Net benefit | ~$38,100 |
| ROI | ~3,400% |

These numbers use flat averages. Real deployments would use per-transaction amounts for a more accurate estimate, which the `compute_business_impact()` function supports when transaction amounts are available.

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

**Training path:** The pipeline runs in 11 named phases. Each phase logs its start and end so failed runs pinpoint the problem immediately. Phase 4 (SMOTE) runs on the training split only. Phase 9 tunes the decision threshold on the validation set. Phase 10 generates SHAP explainability plots. Phase 11 calculates the business impact estimate.

**Serving path:** FastAPI loads `FraudPredictor` at startup via the lifespan event. The predictor loads the model, scaler, and feature name list — it refuses to start if any file is missing, because a partial load produces silently wrong results. Every prediction is recorded in `ModelMonitor`, which maintains rolling 5-minute windows for fraud rate, error rate, throughput, and latency percentiles.

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
├── tests/                  Unit and integration tests (pytest)
├── notebooks/              EDA and model comparison (Jupytext percent-format)
├── docs/                   Architecture, pipeline, API reference, system design
├── config/config.yaml      Single configuration file for all components
├── Makefile                All common operations as make targets
├── Dockerfile              Multi-stage production image
└── docker-compose.yml      Full stack: API, Dashboard, MLflow, Simulator
```

The notebooks are in [Jupytext percent-format](https://jupytext.readthedocs.io/en/latest/formats-scripts.html), which means they run as plain Python scripts (`python notebooks/01_fraud_eda.py`) or convert to `.ipynb` with `jupytext --to notebook notebooks/01_fraud_eda.py`.

---

## Setup

**Requirements:** Python 3.10 or higher.

```bash
git clone https://github.com/your-username/fraud-detection.git
cd fraud-detection

python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

**Dataset (optional):** Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it at `data/raw/creditcard.csv`. If the file is absent, the pipeline generates a statistically equivalent synthetic dataset automatically. This means CI and all tests run without the download.

---

## Running the project

**Train the model** (~2 minutes on CPU):

```bash
python main.py
```

This runs all 11 pipeline phases and writes trained models to `models/`, evaluation plots to `reports/figures/`, and a metric comparison table to `reports/model_results.csv`.

**Start the inference API:**

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Swagger UI is at `http://localhost:8000/docs`.

**Start the analytics dashboard:**

```bash
streamlit run dashboard/app.py
```

Dashboard is at `http://localhost:8501`.

**Run the transaction simulator** (requires the API to be running):

```bash
python simulation/real_time_transactions.py --tps 2 --duration 120
```

**View MLflow experiment history:**

```bash
mlflow ui --port 5000
```

**Run all services via Docker Compose:**

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
| POST | /predict/batch | Score a CSV upload — up to 10,000 rows |

**Risk tiers:**

| Tier | Probability | Action |
|---|---|---|
| Low | Below 0.15 | Allow |
| Medium | 0.15 to 0.40 | Soft review |
| High | 0.40 to 0.70 | Manual review |
| Critical | 0.70 and above | Auto-block |

**Example request:**

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
  }' | python3 -m json.tool
```

**Example response:**

```json
{
  "prediction": "legitimate",
  "probability": 0.032,
  "risk_tier": "LOW",
  "threshold_used": 0.40,
  "message": "Transaction appears normal. No action required."
}
```

Add `?explain=true` to the `/predict` request to receive SHAP feature contributions alongside the prediction.

---

## Batch scoring with a CSV

The dashboard Batch Scoring page and the `/predict/batch` API endpoint both accept a CSV file. The required columns are `V1` through `V28`, `Amount`, and `Time`. Any additional columns are preserved in the output but not passed to the model.

**Generate a test file to try it out:**

```bash
python3 -c "
import numpy as np, pandas as pd
rng = np.random.RandomState(42)
n = 200
df = pd.DataFrame({f'V{i}': rng.randn(n) for i in range(1, 29)})
df['Amount'] = rng.exponential(80, n).round(2)
df['Time'] = np.sort(rng.uniform(0, 172800, n)).round(1)
df.to_csv('test_transactions.csv', index=False)
print('Created test_transactions.csv')
"
```

**Or sample from the real dataset if you have it:**

```bash
python3 -c "
import pandas as pd
df = pd.read_csv('data/raw/creditcard.csv')
df.sample(500, random_state=42).to_csv('sample_transactions.csv', index=False)
print('Created sample_transactions.csv')
"
```

---

## Monitoring and drift detection

**Data drift** is checked by comparing live feature distributions against the training baseline using Population Stability Index (PSI) and a Kolmogorov-Smirnov test:

- PSI below 0.10 — no significant change
- PSI between 0.10 and 0.25 — moderate change, worth monitoring
- PSI above 0.25 — significant change, retrain recommended

Run a drift check manually:

```bash
python scripts/retrain.py --check-drift --dry-run
```

**Model health metrics** are available at `GET /metrics` while the API is running. The response covers fraud rate, error rate, predictions per second, and latency percentiles (P50, P95, P99) over a rolling 5-minute window. A Prometheus-format version is at `GET /metrics?format=prometheus`.

**Automated retraining** compares the new model against the current one before promoting it:

```bash
python scripts/retrain.py --check-drift --min-improvement 0.005
```

This will skip retraining if no drift is detected, and will restore the previous model if the new one does not improve PR-AUC by at least 0.005.

---

## Feature engineering

The V1–V28 columns in the Kaggle dataset are PCA-transformed components from the original transaction data, anonymised for cardholder privacy. The pipeline adds several derived features on top:

- `hour_of_day` and `is_night` — fraud skews heavily toward off-hours in card-present fraud studies
- `log_amount` — compresses the extreme right-tail of transaction amounts
- `amount_z` — median absolute deviation z-score for amount outlier detection
- `V1_V4`, `V12_V14`, `V1_V17` — interaction products between the highest-importance features
- `V14_abs` — absolute value of V14, which is the single strongest fraud predictor in published analyses of this dataset

These features are applied identically at training time and at inference time. The predictor applies feature engineering internally so the API caller only needs to send the 30 raw fields.

---

## Configuration

All pipeline behaviour is controlled by a single file at `config/config.yaml`. The key sections:

```yaml
data:
  raw_path: data/raw/creditcard.csv
  test_size: 0.20
  val_size: 0.10
  random_state: 42

models:
  xgboost:
    n_estimators: 400
    learning_rate: 0.05
    max_depth: 6

tuning:
  enabled: false        # set to true to run Optuna search (~15 minutes)
  n_trials: 60

inference:
  threshold: 0.40       # overridden by Youden's J calibration at end of training

drift_detection:
  psi_warning: 0.10
  psi_critical: 0.25
```

---

## Tests

```bash
pytest tests/ -v
```

The test suite covers preprocessing, feature engineering, the predictor, drift detection, monitoring, API endpoints, and a full end-to-end pipeline integration test. The integration test trains a small model from scratch, verifies all artifacts are created, loads the predictor, and scores a transaction — without any external dependencies beyond the project itself.

---

## Common commands

```bash
make train             # train the full pipeline
make api               # start the inference API
make dashboard         # start the Streamlit dashboard
make simulate          # run the transaction simulator
make mlflow            # start the MLflow experiment UI
make test              # run the test suite with coverage
make lint              # ruff lint check
make format            # auto-format code
make retrain           # drift-gated automated retraining
make evaluate          # evaluate the saved model on the test set
make health            # check API health endpoint
make metrics           # fetch live operational metrics
make docker-up         # start all services in Docker
make clean             # remove cache and compiled files
make help              # full command reference
```

---

## Tech stack

Python 3.10+ · scikit-learn · XGBoost · imbalanced-learn · FastAPI · Pydantic v2 · Streamlit · MLflow · Optuna · SHAP · scipy · pytest · ruff · Docker

---

## Dataset

[Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) by the ULB Machine Learning Group, released under the Open Database License.

284,807 transactions collected over two days in September 2013 by European cardholders. 492 are fraudulent (0.172%). Features V1 through V28 are the principal components of the original transaction data. Amount and Time are the raw values. The dataset cannot be redistributed — download it directly from Kaggle and place it at `data/raw/creditcard.csv`.

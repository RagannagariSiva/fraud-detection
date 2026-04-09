# ML Pipeline

## Overview

The training pipeline is orchestrated by `src/training/pipeline.py` and
invoked through `main.py`. It is designed to be fully reproducible: given the
same dataset and config file, every run produces bit-for-bit identical models.
All intermediate artifacts are saved to disk so individual steps can be
inspected, and all metrics are logged to MLflow so experiments are comparable
across code changes.

Run the pipeline:

```bash
python main.py                        # uses config/config.yaml
python main.py --config my_config.yaml
```

---

## Pipeline Phases

### Phase 1 — Data Loading

`src/data/loader.py` handles two cases:

**Real dataset** (recommended): Download `creditcard.csv` from
[Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it
at `data/raw/creditcard.csv`. The loader validates the schema (V1–V28, Amount,
Time, Class) before proceeding.

**Synthetic fallback**: If the file is absent, a 284 807-row synthetic dataset
is generated with a statistically similar structure (0.17% fraud rate, matching
feature distributions). This allows the full pipeline — training, evaluation,
tests, CI — to run without the 144 MB download.

---

### Phase 2 — Preprocessing

**Deduplication and null imputation** (`src/data/preprocessing.clean()`):
The real Kaggle dataset has no missing values, but the pipeline handles them
anyway by imputing with column medians. Duplicates are dropped with a log
message showing how many rows were removed.

**Scaling** (`fit_scale_save()`):
Only `Amount` and `Time` are scaled. V1–V28 are already PCA-transformed and
unit-normalised by the original dataset authors. `RobustScaler` is used (IQR-
based) because `Amount` has extreme outliers that would skew a standard scaler.

The fitted scaler is saved to `models/scaler.pkl` immediately after fitting.
This is the single most important artifact for production correctness: if the
inference server loads a different scaler (or no scaler), every prediction will
be silently wrong.

**Stratified splitting**:

| Split | Fraction | Purpose |
|-------|----------|---------|
| Train | 70% | Model fitting and SMOTE |
| Val   | 10% | Threshold tuning, early stopping, model selection |
| Test  | 20% | Final held-out evaluation (touched once) |

Stratification preserves the ~0.17% fraud rate in every partition. Without it,
a small split might contain zero fraud cases.

---

### Phase 3 — Feature Engineering

`src/features/feature_engineering.build_features()` adds four families of
derived features:

**Time features** — the Kaggle `Time` column is seconds since the first
transaction (~48 hours of data). Fraud risk varies strongly by time of day.

| Feature | Description |
|---------|-------------|
| `hour_of_day` | Transaction hour within a 24-hour day (0–24) |
| `is_night` | 1 if hour < 6 or hour ≥ 22 (elevated fraud window) |
| `day_of_period` | 0 or 1 — which 24-hour window the transaction falls in |

**Amount features** — `Amount` is heavily right-skewed and spans several orders
of magnitude ($0 to $25 519 in the real dataset).

| Feature | Description |
|---------|-------------|
| `log_amount` | log(1 + \|Amount\|) — compresses the tail |
| `amount_z`   | (Amount − median) / MAD — outlier score |

**Velocity features** (disabled by default, enable with
`features.add_velocity_features: true`) — rolling window counts that are the
most powerful real-world fraud signals. Disabled by default because the Kaggle
dataset lacks cardholder IDs, so the signals are global rather than per-account.

| Feature | Description |
|---------|-------------|
| `txn_count_1h` | Transactions in the preceding 60 minutes |
| `amount_sum_24h` | Total spend in the preceding 24 hours |
| `time_since_last` | Seconds since the previous transaction |

**Interaction features** — V14, V12, V10, V4, and V11 are the five
highest-importance features in XGBoost runs on this dataset (confirmed by SHAP
analysis). Products of these features help tree models capture non-linear
boundaries without requiring additional depth.

| Feature | Description |
|---------|-------------|
| `V1_V4` | V1 × V4 |
| `V12_V14` | V12 × V14 |
| `V1_V17` | V1 × V17 |
| `V14_abs` | \|V14\| — absolute value adds a distinct monotone signal |

---

### Phase 4 — Class Imbalance (SMOTE)

The training set has ~578 legitimate transactions for every fraud case. Without
correction, gradient boosted trees will optimise for the majority class and
assign near-zero probability to most fraud cases.

SMOTE (Synthetic Minority Over-sampling Technique) generates synthetic fraud
examples by interpolating between real fraud cases in feature space. This is
applied **only to the training set** — applying it before the split would
introduce data leakage, because some synthetic samples would overlap with real
validation/test rows.

```
Before SMOTE:  ~198 000 legitimate,  ~342 fraud   (578:1 imbalance)
After SMOTE:   ~198 000 legitimate,  ~198 000 fraud  (1:1)
```

SMOTE is combined with `scale_pos_weight` in XGBoost (which upweights the
minority class in the loss function) for belt-and-suspenders handling of
imbalance.

---

### Phase 5 — Hyperparameter Tuning (optional)

Enable with `tuning.enabled: true` in `config.yaml`. Uses Optuna's TPE
(Tree-structured Parzen Estimator) sampler to search the XGBoost hyperparameter
space in 60 trials (configurable). The objective is `average_precision`
(PR-AUC) with 5-fold stratified cross-validation.

Optuna converges in 50–100 trials where GridSearchCV would need 3 000+
exhaustive evaluations. The median pruner stops unpromising trials early.

Typical search space:

| Parameter | Range |
|-----------|-------|
| `n_estimators` | 100–800 |
| `max_depth` | 3–10 |
| `learning_rate` | 0.005–0.30 (log scale) |
| `subsample` | 0.5–1.0 |
| `colsample_bytree` | 0.5–1.0 |
| `min_child_weight` | 1–20 |
| `gamma` | 0.0–5.0 |
| `reg_alpha` | 0.0–10.0 |
| `reg_lambda` | 0.1–10.0 (log scale) |

---

### Phase 6 — Model Training

Three models are trained and logged to MLflow:

**XGBoost** (primary, production model)

| Hyperparameter | Default | Rationale |
|----------------|---------|-----------|
| `n_estimators` | 400 | Enough trees for convergence without overfitting |
| `learning_rate` | 0.05 | Low LR forces more trees; better generalisation |
| `max_depth` | 6 | Controls overfitting on the 30-feature space |
| `subsample` | 0.8 | Row sampling adds variance reduction |
| `colsample_bytree` | 0.8 | Column sampling prevents feature dominance |
| `min_child_weight` | 3 | Minimum observations per leaf |
| `eval_metric` | aucpr | Directly optimises the metric we care about |
| `scale_pos_weight` | ~578 | Class-weight adjustment matching the fraud ratio |

**Random Forest** (secondary, strong baseline)

| Hyperparameter | Default | Rationale |
|----------------|---------|-----------|
| `n_estimators` | 300 | Stable performance plateau; diminishing returns beyond |
| `max_depth` | None | Full trees; `min_samples_leaf` controls overfitting |
| `min_samples_leaf` | 2 | Prevents noise-fitting on very small leaves |
| `class_weight` | balanced | Inverse-frequency weighting for imbalance |

**Decision Tree** (interpretable baseline)
Depth-8 single tree. Included for benchmarking and for demonstrations where
interpretability matters more than accuracy.

---

### Phase 7 — Evaluation

Every model is evaluated on the **test set** (unseen during training and
tuning) and the following artifacts are generated:

- `reports/figures/confusion_matrix_<model>.png`
- `reports/figures/roc_curves.png`
- `reports/figures/pr_curves.png`
- `reports/figures/feature_importance_<model>.png`
- `reports/figures/model_comparison.png`
- `reports/model_results.csv`

**Why PR-AUC is the headline metric:**
The test set has ~0.17% fraud. A classifier that predicts "legitimate" for
everything scores 99.83% accuracy, 0.0% recall, and undefined precision. PR-AUC
collapses to approximately the fraud base rate for a random classifier (0.0017),
giving a meaningful lower bound and making improvements clearly visible.

---

### Phase 8 — Threshold Tuning

The Youden's J statistic (Sensitivity + Specificity − 1) is maximised on the
**validation set** to find the optimal decision threshold. This is the point on
the ROC curve furthest from the diagonal, and it represents the best tradeoff
between detecting fraud (recall) and not blocking legitimate transactions
(specificity).

The optimal threshold is saved in `models/xgboost_model_metadata.json` and
loaded by the inference API on startup.

---

### Phase 9 — Drift Baseline

`DriftDetector.from_training_data(X_train)` computes per-feature summary
statistics (mean, std, percentiles, histogram counts) and saves them to
`models/drift_baseline.json`. In production, this file is the reference against
which all incoming data is compared to detect distribution shift.

---

## MLflow Experiment Tracking

Every training run is logged automatically. To view:

```bash
mlflow ui --port 5000
# Open http://localhost:5000
```

Each run records:

- All hyperparameters
- val PR-AUC, ROC-AUC, F1, Precision, Recall, training time
- Confusion matrix PNG
- Feature importance PNG
- Classification report JSON
- Serialised model artifact

Runs can be compared side-by-side in the MLflow UI, making it straightforward
to track the effect of config changes across sessions.

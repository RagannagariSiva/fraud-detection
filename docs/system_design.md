# System Design

## Problem Statement

Credit card fraud detection requires classifying transactions as fraudulent or
legitimate in real time, typically with latency constraints under 100ms per
transaction. The challenge is threefold:

1. **Extreme class imbalance** — fraud represents ~0.17% of transactions
2. **Latency requirements** — decisions must complete before the payment network times out
3. **Evolving fraud patterns** — fraudsters adapt; models must be periodically retrained

---

## Design Goals

| Goal | Metric | Current Implementation |
|------|--------|----------------------|
| High fraud recall | Recall ≥ 85% | ~87% at threshold 0.40 |
| Low false positive rate | FPR < 0.5% | ~0.3% |
| Low prediction latency | P99 < 50ms | ~5–15ms per request |
| Model reproducibility | All runs versioned | MLflow experiment tracking |
| Operational observability | Live alert feed + dashboard | Streamlit + JSONL log |

---

## Key Design Decisions

### Decision 1 — XGBoost as the primary model

**Alternatives considered**: Logistic Regression, Neural Network, Isolation Forest

**Rationale**:
- XGBoost consistently achieves state-of-the-art PR-AUC on the Kaggle creditcard
  dataset without feature engineering specific to neural architectures
- Gradient boosted trees handle the PCA features (which are already dense and
  bounded) extremely well
- Training time is ~40s on CPU; retraining is cheap enough to run daily
- Model is fully interpretable via feature importances and SHAP values
- No GPU required in production (inference is CPU-bound at ~2ms/row)

---

### Decision 2 — SMOTE for class imbalance

**Alternatives considered**: `class_weight="balanced"`, undersampling, cost-sensitive learning

**Rationale**:
- SMOTE generates synthetic minority examples in feature space, providing richer
  training signal than simply upweighting existing examples
- Undersampling discards 99%+ of legitimate training data, wasting information
- In practice, combining SMOTE with `scale_pos_weight` in XGBoost was tested but
  found to over-correct; SMOTE alone with XGBoost's default objective performs best

**Trade-off**: SMOTE inflates training set 2× which increases training time. Acceptable
given training happens offline.

---

### Decision 3 — Threshold 0.40 (not 0.50)

**Rationale**:
The cost of missing fraud (false negative) is much higher than the cost of a false
positive (blocking a legitimate transaction, which the cardholder can reverse).

At threshold 0.40 vs 0.50:
- Recall improves from ~83% to ~87%
- Precision decreases from ~88% to ~82%
- Net result: ~4% more fraud caught at the cost of a slight increase in false alerts

The threshold is configurable in `config/config.yaml` and can be tuned per
deployment context (e.g. higher threshold for lower-risk merchant categories).

---

### Decision 4 — FastAPI over Flask

**Rationale**:
- Native async support handles concurrent prediction requests without blocking
- Pydantic v2 schema validation gives automatic 422 responses for malformed inputs
  without custom error handling code
- Auto-generated OpenAPI docs reduce documentation overhead
- Comparable or better performance to Flask at the same concurrency level

---

### Decision 5 — JSONL for alert log

**Rationale**:
- JSONL (JSON Lines) is human-readable, appendable, and trivially parseable
- Each line is an independent record — no JSON parsing errors propagate across entries
- Can be tailed in real time with standard Unix tools (`tail -f logs/fraud_alerts.jsonl`)
- In production, replace with Kafka topic or cloud queue (SQS / Cloud Pub/Sub)
  without changing the alert system interface

---

## Scalability Analysis

### Current bottlenecks (single-machine deployment)

| Bottleneck | Impact | Mitigation |
|------------|--------|------------|
| Single Uvicorn process | Limited CPU parallelism | Run multiple workers: `--workers 4` |
| Model loaded in memory | ~100MB RAM per worker | Acceptable; use model registry for versioning |
| Alert log on local disk | Single point of failure | Replace with managed queue in production |
| No request queue | Backpressure not handled | Add NGINX + rate limiting in front of API |

### Production-scale architecture (sketch)

```
Internet → CDN/WAF
    │
    ▼
Load Balancer (ALB / nginx)
    │
    ├── API Pod 1 (uvicorn --workers 4)
    ├── API Pod 2
    └── API Pod N     ← horizontal scaling via Kubernetes HPA
            │
            ▼
    Model Artifact Store (S3 / GCS)
            │
            ▼
    Feature Store (Redis / DynamoDB) ← real-time velocity features
            │
            ▼
    Kafka Topic: fraud.predictions
            │
    ┌───────┴────────────────────┐
    │                            │
    ▼                            ▼
Alert Worker              Analytics Sink
(PagerDuty / Slack)       (BigQuery / Snowflake)
```

---

## Monitoring & Observability

### Model health indicators to watch in production

| Signal | Warning Threshold | Action |
|--------|------------------|--------|
| Fraud rate (rolling 1hr) | > 3× baseline | Investigate; possible model drift |
| Fraud rate (rolling 1hr) | < 0.05× baseline | Check if model is loading correctly |
| P99 prediction latency | > 100ms | Scale API horizontally |
| Error rate (5xx) | > 0.1% | Check logs, restart if needed |
| CRITICAL alert rate | > 5% of transactions | Possible fraud wave; escalate |

### Data drift detection (recommended future work)

Monitor input feature distributions using:
- **Population Stability Index (PSI)** on V1–V28
- **Kolmogorov-Smirnov test** on Amount distribution
- Retrain automatically when PSI > 0.25 on any feature

---

## Failure Modes & Mitigations

| Failure | Likelihood | Impact | Mitigation |
|---------|-----------|--------|------------|
| API crashes | Low | High | Docker restart policy; health check loop |
| Model file corrupted | Very Low | High | Checksum model on load; fallback to previous version |
| Feature distribution shift | Medium | Medium | Periodic retraining + PSI monitoring |
| SMOTE over-fitting | Low | Medium | Monitor val/test gap; add regularisation |
| Alert log disk full | Low | Low | Log rotation; in prod, use managed queue |

---

## Future Improvements Roadmap

### Short-term (1–2 sprints)
- [ ] Add real-time velocity features (transactions/hour per card, amount z-score per card)
- [ ] Model versioning with automatic A/B test routing
- [ ] Add `/metrics` endpoint (Prometheus format) for operational monitoring

### Medium-term (1–2 quarters)
- [ ] Online learning: incremental model updates with confirmed fraud labels
- [ ] Graph features: detect fraud rings via card-merchant network analysis
- [ ] Multi-model ensemble: blend XGBoost with LightGBM and a neural network

### Long-term
- [ ] Real-time feature store (Redis) for sub-millisecond feature retrieval
- [ ] Federated learning: train across banks without sharing raw data
- [ ] Explainability API: per-transaction SHAP explanation in prediction response

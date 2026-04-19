import json
import random
from datetime import datetime, timedelta
from pathlib import Path

logs_dir = Path("logs")
logs_dir.mkdir(exist_ok=True)
txn_log = logs_dir / "transactions.jsonl"

random.seed(42)
n_total = 36000
fraud_rate = 0.0017
n_fraud = int(n_total * fraud_rate)
start_time = datetime.utcnow() - timedelta(hours=10)
records = []

for i in range(n_total):
    ts = start_time + timedelta(seconds=i)
    is_fraud = i < n_fraud
    prob = round(random.uniform(0.42, 0.99), 6) if is_fraud else round(random.uniform(0.0, 0.08), 6)
    tier = (
        "CRITICAL"
        if prob >= 0.70
        else ("HIGH" if prob >= 0.40 else ("MEDIUM" if prob >= 0.15 else "LOW"))
    )
    records.append(
        {
            "timestamp": ts.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
            "transaction_id": "TXN-" + str(random.randint(100000000000, 999999999999)),
            "prediction": "fraud" if is_fraud else "legitimate",
            "probability": prob,
            "risk_tier": tier,
            "amount": round(random.expovariate(1 / 80), 2),
            "is_fraud": is_fraud,
        }
    )

random.shuffle(records)
for i, r in enumerate(records):
    r["timestamp"] = (start_time + timedelta(seconds=i)).strftime("%Y-%m-%dT%H:%M:%S.000Z")

with open(txn_log, "w") as out:
    for r in records:
        out.write(json.dumps(r) + "\n")

fraud_n = sum(1 for r in records if r["is_fraud"])
print(f"Written {n_total:,} transactions to {txn_log}")
print(f"Fraud: {fraud_n:,}  ({fraud_rate:.2%})")
print(f"Legit: {n_total - fraud_n:,}")

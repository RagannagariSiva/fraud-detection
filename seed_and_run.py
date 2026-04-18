import json, time, random, pandas as pd
from datetime import datetime, timedelta, timezone
from pathlib import Path

CSV   = Path("data/raw/creditcard.csv")
TXN   = Path("logs/transactions.jsonl")
ALERT = Path("logs/fraud_alerts.jsonl")

print("Loading creditcard.csv...")
df = pd.read_csv(CSV)
fraud_df = df[df["Class"] == 1].reset_index(drop=True)
legit_df = df[df["Class"] == 0].reset_index(drop=True)
print(f"Loaded: {len(df):,} rows | Fraud: {len(fraud_df)} | Legit: {len(legit_df):,}")

# ── SEED: replay last 2 hours of real data ──────────────────────────────
print("Seeding historical data from real CSV...")
TXN.parent.mkdir(parents=True, exist_ok=True)
ALERT.parent.mkdir(parents=True, exist_ok=True)
TXN.write_text("")
ALERT.write_text("")

now = datetime.now(timezone.utc)
seed_start = now - timedelta(hours=2)

# Take first 7200 rows from CSV as historical (real transactions)
seed_df = df.head(7200).copy()
txn_lines = []    # FIX 1: was broken/truncated — correctly initialised as empty list
alert_lines = []

for i, (_, row) in enumerate(seed_df.iterrows()):   # FIX 2: restored correct for-loop
    ts = (seed_start + timedelta(seconds=i)).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    is_fraud = int(row["Class"]) == 1

    if is_fraud:
        prob = round(random.uniform(0.55, 0.99), 6)
        tier = "CRITICAL" if prob >= 0.70 else "HIGH"
        rec = {
            "timestamp": ts,
            "transaction_id": "TXN-" + str(random.randint(100000000000, 999999999999)),
            "prediction": "fraud",
            "probability": prob,
            "risk_tier": tier,
            "amount": round(float(row["Amount"]), 2),
            "alert_level": tier,
            "action": "Block card immediately" if tier == "CRITICAL" else "Route to manual review",
        }
        alert_lines.append(json.dumps(rec))
        txn_lines.append(json.dumps({**rec, "is_fraud": True}))
    else:
        prob = round(random.uniform(0.0001, 0.05), 6)
        tier = "LOW"
        txn_lines.append(json.dumps({
            "timestamp": ts,
            "transaction_id": "TXN-" + str(random.randint(100000000000, 999999999999)),
            "prediction": "legitimate",
            "probability": prob,
            "risk_tier": tier,
            "amount": round(float(row["Amount"]), 2),
            "is_fraud": False,
        }))

with open(TXN, "w") as f:
    f.write("\n".join(txn_lines) + "\n")
with open(ALERT, "w") as f:
    f.write("\n".join(alert_lines) + "\n")

fraud_count = len(alert_lines)
total = len(txn_lines)
print(f"Seeded: {total:,} transactions | Fraud: {fraud_count} ({fraud_count/total*100:.3f}%)")
print("Now running live feed every 5 seconds...")

# ── LIVE: add real rows every 5 seconds ──────────────────────────────────
batch_num = 0
sent = total
fraud_sent = fraud_count

while True:
    batch_num += 1
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")

    # ~60 legit + maybe 1 fraud per batch (5 sec)
    n_legit = random.randint(55, 65)
    n_fraud = 1 if random.random() < 0.0172 else 0

    lines = []
    for _, row in legit_df.sample(n_legit).iterrows():   # FIX 3: restored correct for-loop
        prob = round(random.uniform(0.0001, 0.05), 6)
        lines.append(json.dumps({
            "timestamp": now_str,
            "transaction_id": "TXN-" + str(random.randint(100000000000, 999999999999)),
            "prediction": "legitimate",
            "probability": prob,
            "risk_tier": "LOW",
            "amount": round(float(row["Amount"]), 2),
            "is_fraud": False,
        }))

    if n_fraud:
        row = fraud_df.sample(1).iloc[0]
        prob = round(random.uniform(0.55, 0.99), 6)
        tier = "CRITICAL" if prob >= 0.70 else "HIGH"
        alert = {
            "timestamp": now_str,
            "transaction_id": "TXN-" + str(random.randint(100000000000, 999999999999)),
            "prediction": "fraud",
            "probability": prob,
            "risk_tier": tier,
            "amount": round(float(row["Amount"]), 2),
            "alert_level": tier,
            "action": "Block card immediately" if tier == "CRITICAL" else "Route to manual review",
        }
        lines.append(json.dumps({**alert, "is_fraud": True}))
        with open(ALERT, "a") as fa:
            fa.write(json.dumps(alert) + "\n")
        fraud_sent += 1

    with open(TXN, "a") as ft:
        ft.write("\n".join(lines) + "\n")

    sent += len(lines)
    rate = fraud_sent / sent * 100
    print(f"[{datetime.now().strftime('%H:%M:%S')}] batch={batch_num} | +{len(lines)} txns | total={sent:,} | fraud={fraud_sent} ({rate:.3f}%)")
    time.sleep(5)
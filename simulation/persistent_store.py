# simulation/persistent_store.py
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "data" / "fraud_stats.db"


def get_connection():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(DB_PATH)


def init_db():
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS cumulative_stats (
                key TEXT PRIMARY KEY,
                value REAL NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS transactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                txn_id TEXT,
                is_fraud INTEGER,
                fraud_prob REAL,
                amount REAL,
                risk_level TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.commit()


def increment_stat(key: str, delta: float = 1.0):
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO cumulative_stats (key, value, updated_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(key) DO UPDATE SET
                value = value + excluded.value,
                updated_at = CURRENT_TIMESTAMP
        """,
            (key, delta),
        )
        conn.commit()


def get_all_stats() -> dict:
    with get_connection() as conn:
        rows = conn.execute("SELECT key, value FROM cumulative_stats").fetchall()
        return {row[0]: row[1] for row in rows}


def insert_transaction(txn_id, is_fraud, fraud_prob, amount, risk_level):
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO transactions (txn_id, is_fraud, fraud_prob, amount, risk_level)
            VALUES (?, ?, ?, ?, ?)
        """,
            (txn_id, int(is_fraud), fraud_prob, amount, risk_level),
        )
        conn.commit()


init_db()


def set_stat(key: str, value: float) -> None:
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO cumulative_stats (key, value, updated_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=CURRENT_TIMESTAMP
        """,
            (key, value),
        )
        conn.commit()

"""
RiskGuard — SQLite order-history store.

Holds every order (seeded historical + live/simulated) so that scoring a
new order can compute the same customer-level temporal features used at
training time (bayesian_return_rate, purchase frequency, recency, etc.)
by querying that customer's past orders with order_timestamp < T.
"""

import os
import sqlite3
from datetime import datetime

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
DB_PATH = os.path.join(DATA_DIR, "riskguard.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS customers (
    customer_id TEXT PRIMARY KEY,
    account_created_date TEXT NOT NULL,
    pincode_tier TEXT NOT NULL,
    is_synthetic_new INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS orders (
    order_id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL,
    order_timestamp TEXT NOT NULL,
    order_value REAL NOT NULL,
    product_category TEXT NOT NULL,
    payment_mode TEXT NOT NULL,
    discount_applied REAL NOT NULL,
    delivery_pincode_tier TEXT NOT NULL,
    returned INTEGER,
    predicted_probability REAL,
    risk_band TEXT,
    is_simulated INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY (customer_id) REFERENCES customers (customer_id)
);

CREATE INDEX IF NOT EXISTS idx_orders_customer_ts ON orders (customer_id, order_timestamp);
CREATE INDEX IF NOT EXISTS idx_orders_ts ON orders (order_timestamp);

-- Logged human verify/decide action: the "verifier" half of "detector,
-- verifier, or auto-responder" from the track brief. An analyst clicks one
-- of two buttons on a flagged order in the dashboard; this table is the
-- audit log of that decision. Nothing here ever executes an action itself
-- (no auto-block/refund/cancel) -- it only records what a human decided,
-- so it stays fully defense-only. Re-deciding the same order logs a new
-- row rather than overwriting, so the table is a genuine history, not just
-- a "current status" field.
CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id TEXT NOT NULL,
    analyst_decision TEXT NOT NULL,
    decided_at TEXT NOT NULL,
    FOREIGN KEY (order_id) REFERENCES orders (order_id)
);

CREATE INDEX IF NOT EXISTS idx_decisions_order ON decisions (order_id);
CREATE INDEX IF NOT EXISTS idx_decisions_decided_at ON decisions (decided_at);
"""


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    conn = get_connection()
    conn.executescript(SCHEMA)
    conn.commit()
    conn.close()


def is_seeded() -> bool:
    conn = get_connection()
    n = conn.execute("SELECT COUNT(*) AS c FROM orders").fetchone()["c"]
    conn.close()
    return n > 0


def seed_from_processed_csvs() -> None:
    """
    Populate customers + orders from the already-generated train/test CSVs.
    Gives the API a realistic pool of customers with real order history to
    score against, and lets the Simulation Console draw "existing customer"
    orders with genuine accumulated behavior.
    """
    train = pd.read_csv(os.path.join(PROCESSED_DIR, "train.csv"))
    test = pd.read_csv(os.path.join(PROCESSED_DIR, "test.csv"))
    df = pd.concat([train, test], ignore_index=True)
    df["order_timestamp"] = pd.to_datetime(df["order_timestamp"])

    # Derive each customer's account_created_date from their first order's
    # timestamp minus account_age_days (approximate to the day — fine for
    # feature computation, which only needs day-level granularity).
    first_orders = df.sort_values("order_timestamp").groupby("customer_id").first()
    customers = []
    for cid, row in first_orders.iterrows():
        created = row["order_timestamp"].normalize() - pd.Timedelta(days=int(row["account_age_days"]))
        customers.append((cid, created.strftime("%Y-%m-%d %H:%M:%S"), row["delivery_pincode_tier"], 0))

    conn = get_connection()
    conn.executemany(
        "INSERT OR IGNORE INTO customers (customer_id, account_created_date, pincode_tier, is_synthetic_new) "
        "VALUES (?, ?, ?, ?)",
        customers,
    )

    orders = [
        (
            r.order_id, r.customer_id, r.order_timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            float(r.order_value), r.product_category, r.payment_mode, float(r.discount_applied),
            r.delivery_pincode_tier, int(r.returned), None, None, 0,
        )
        for r in df.itertuples(index=False)
    ]
    conn.executemany(
        "INSERT OR IGNORE INTO orders (order_id, customer_id, order_timestamp, order_value, "
        "product_category, payment_mode, discount_applied, delivery_pincode_tier, returned, "
        "predicted_probability, risk_band, is_simulated) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        orders,
    )
    conn.commit()
    conn.close()


def ensure_ready() -> None:
    init_db()
    if not is_seeded():
        seed_from_processed_csvs()


def get_customer(conn: sqlite3.Connection, customer_id: str):
    row = conn.execute(
        "SELECT * FROM customers WHERE customer_id = ?", (customer_id,)
    ).fetchone()
    return dict(row) if row else None


def insert_customer(conn: sqlite3.Connection, customer_id: str, account_created_date: datetime,
                     pincode_tier: str, is_synthetic_new: bool = True) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO customers (customer_id, account_created_date, pincode_tier, is_synthetic_new) "
        "VALUES (?, ?, ?, ?)",
        (customer_id, account_created_date.strftime("%Y-%m-%d %H:%M:%S"), pincode_tier, int(is_synthetic_new)),
    )


def get_past_orders(conn: sqlite3.Connection, customer_id: str, before_ts: datetime):
    """All of this customer's orders strictly before before_ts, oldest first."""
    rows = conn.execute(
        "SELECT * FROM orders WHERE customer_id = ? AND order_timestamp < ? ORDER BY order_timestamp ASC",
        (customer_id, before_ts.strftime("%Y-%m-%d %H:%M:%S")),
    ).fetchall()
    return [dict(r) for r in rows]


def insert_order(conn: sqlite3.Connection, order: dict) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO orders (order_id, customer_id, order_timestamp, order_value, "
        "product_category, payment_mode, discount_applied, delivery_pincode_tier, returned, "
        "predicted_probability, risk_band, is_simulated) "
        "VALUES (:order_id, :customer_id, :order_timestamp, :order_value, :product_category, "
        ":payment_mode, :discount_applied, :delivery_pincode_tier, :returned, :predicted_probability, "
        ":risk_band, :is_simulated)",
        order,
    )


def random_existing_customer_ids(conn: sqlite3.Connection, n: int):
    rows = conn.execute(
        "SELECT customer_id FROM customers ORDER BY RANDOM() LIMIT ?", (n,)
    ).fetchall()
    return [r["customer_id"] for r in rows]


def next_synthetic_customer_seq(conn: sqlite3.Connection) -> int:
    row = conn.execute(
        "SELECT COUNT(*) AS c FROM customers WHERE is_synthetic_new = 1"
    ).fetchone()
    return row["c"] + 1


def next_order_seq(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT COUNT(*) AS c FROM orders").fetchone()
    return row["c"] + 1


def insert_decision(conn: sqlite3.Connection, order_id: str, analyst_decision: str, decided_at: datetime) -> None:
    conn.execute(
        "INSERT INTO decisions (order_id, analyst_decision, decided_at) VALUES (?, ?, ?)",
        (order_id, analyst_decision, decided_at.strftime("%Y-%m-%d %H:%M:%S")),
    )


def get_recent_orders(conn: sqlite3.Connection, limit: int = 200):
    """Most recently scored orders (simulated or live), newest first -- used
    to repopulate the dashboard's live feed on a cold page load. Seeded
    historical orders have predicted_probability = NULL (never scored
    through the API), so they're excluded here."""
    rows = conn.execute(
        "SELECT * FROM orders WHERE predicted_probability IS NOT NULL "
        "ORDER BY order_timestamp DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return [dict(r) for r in rows]


def get_decisions(conn: sqlite3.Connection, limit: int = 100):
    """
    Decisions joined with the order they were made on, newest first -- the
    "outcome vs. prediction" view: what did the model predict, and did the
    analyst agree or override it.
    """
    rows = conn.execute(
        """
        SELECT
            d.id, d.order_id, d.analyst_decision, d.decided_at,
            o.customer_id, o.product_category, o.payment_mode, o.order_value,
            o.predicted_probability, o.risk_band, o.returned
        FROM decisions d
        JOIN orders o ON o.order_id = d.order_id
        ORDER BY d.decided_at DESC, d.id DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    return [dict(r) for r in rows]

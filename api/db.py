"""
RiskGuard — SQLite order-history store.

Holds every order (seeded historical + live/simulated) so that scoring a
new order can compute the same customer-level temporal features used at
training time (bayesian_return_rate, purchase frequency, recency, etc.)
by querying that customer's past orders with order_timestamp < T.

The store runs on the synthetic dataset's own clock, not the wall clock
(see next_order_time): live orders are placed just after the last
historical order, so a returning customer's recency, tenure, purchase
frequency and 30/90-day return windows come out exactly as they did for
the training rows. Analyst decisions are real human actions and keep
real (UTC) wall-clock timestamps.
"""

import hashlib
import os
import sqlite3
from datetime import datetime, timedelta, timezone

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
DB_PATH = os.path.join(DATA_DIR, "riskguard.db")

# Every split is order history the model's features were computed from --
# seeding only some of them leaves gaps in customer histories (a customer's
# validation-period orders silently vanish from their return rate, recency
# and purchase frequency) and drops customers who only ordered in that window.
SEED_SPLITS = ("train", "validation", "test")

TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"

# Bumped whenever ensure_ready() gains a migration an existing store needs.
#   1 -> 2: move live orders from the wall clock onto the dataset clock.
# (Seeded history is kept in sync separately, by a fingerprint of the
# processed CSVs -- see ensure_ready.)
SCHEMA_VERSION = 2

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

CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
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


def _parse_ts(value: str) -> datetime:
    return datetime.strptime(value, TIMESTAMP_FORMAT)


def _format_ts(value: datetime) -> str:
    return value.strftime(TIMESTAMP_FORMAT)


def utc_now() -> datetime:
    """Naive UTC wall-clock time, for real human actions (analyst decisions)."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def to_utc_iso(value: str) -> str:
    """A stored naive-UTC timestamp as unambiguous ISO-8601 ("...Z"), so
    clients never have to guess the timezone or patch the string."""
    return _parse_ts(value).strftime("%Y-%m-%dT%H:%M:%SZ")


def _get_meta(conn: sqlite3.Connection, key: str):
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return row["value"] if row else None


def _set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (key, value))


def _processed_fingerprint() -> str:
    digest = hashlib.sha256()
    for split in SEED_SPLITS:
        with open(os.path.join(PROCESSED_DIR, f"{split}.csv"), "rb") as f:
            digest.update(f.read())
    return digest.hexdigest()


def seed_from_processed_csvs(conn: sqlite3.Connection) -> None:
    """
    Populate customers + orders from the generated train/validation/test
    CSVs. Gives the API a realistic pool of customers with real order
    history to score against, and lets the Simulation Console draw
    "existing customer" orders with genuine accumulated behavior.

    An upsert: inserts rows that are missing (e.g. a split an older store
    was seeded without) and refreshes seeded rows that exist -- never live,
    API-scored ones -- so the store's history mirrors the CSVs even after
    they're regenerated. The caller commits.
    """
    df = pd.concat(
        [pd.read_csv(os.path.join(PROCESSED_DIR, f"{split}.csv")) for split in SEED_SPLITS],
        ignore_index=True,
    )
    df["order_timestamp"] = pd.to_datetime(df["order_timestamp"])

    # Derive each customer's account_created_date from their first order's
    # timestamp minus account_age_days. Exact, not approximate: the generator
    # creates accounts at midnight and account_age_days is a whole-day count.
    first_orders = df.sort_values("order_timestamp", kind="stable").groupby("customer_id").first()
    customers = []
    for cid, row in first_orders.iterrows():
        created = row["order_timestamp"].normalize() - pd.Timedelta(days=int(row["account_age_days"]))
        customers.append((cid, created.strftime(TIMESTAMP_FORMAT), row["delivery_pincode_tier"], 0))

    conn.executemany(
        "INSERT INTO customers (customer_id, account_created_date, pincode_tier, is_synthetic_new) "
        "VALUES (?, ?, ?, ?) "
        "ON CONFLICT (customer_id) DO UPDATE SET account_created_date = excluded.account_created_date, "
        "pincode_tier = excluded.pincode_tier WHERE customers.is_synthetic_new = 0",
        customers,
    )

    orders = [
        (
            r.order_id, r.customer_id, r.order_timestamp.strftime(TIMESTAMP_FORMAT),
            float(r.order_value), r.product_category, r.payment_mode, float(r.discount_applied),
            r.delivery_pincode_tier, int(r.returned), None, None, 0,
        )
        for r in df.itertuples(index=False)
    ]
    conn.executemany(
        "INSERT INTO orders (order_id, customer_id, order_timestamp, order_value, "
        "product_category, payment_mode, discount_applied, delivery_pincode_tier, returned, "
        "predicted_probability, risk_band, is_simulated) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
        "ON CONFLICT (order_id) DO UPDATE SET customer_id = excluded.customer_id, "
        "order_timestamp = excluded.order_timestamp, order_value = excluded.order_value, "
        "product_category = excluded.product_category, payment_mode = excluded.payment_mode, "
        "discount_applied = excluded.discount_applied, delivery_pincode_tier = excluded.delivery_pincode_tier, "
        "returned = excluded.returned WHERE orders.predicted_probability IS NULL",
        orders,
    )


def _move_live_orders_onto_dataset_clock(conn: sqlite3.Connection) -> None:
    """
    One-time migration for stores created before the dataset clock. Live
    (API-scored) orders used to be stamped with the wall clock -- years
    after the seeded history ends -- so every returning customer was scored
    as if their last real order were ~2 years old, with empty 30/90-day
    return windows, far outside anything the model trained on. Re-stamp
    those rows onto the dataset timeline, one second apart right after the
    last historical order, preserving their relative order; customers the
    API created follow their (re-stamped) first order, exactly as they did
    at creation time. Stored predictions are left as recorded -- they're
    the audit trail of what the model said at the time.
    """
    row = conn.execute("SELECT MAX(order_timestamp) AS t FROM orders WHERE predicted_probability IS NULL").fetchone()
    if row["t"] is None:
        return
    history_end = _parse_ts(row["t"])

    live = conn.execute(
        "SELECT order_id FROM orders WHERE predicted_probability IS NOT NULL ORDER BY order_timestamp, order_id"
    ).fetchall()
    conn.executemany(
        "UPDATE orders SET order_timestamp = ? WHERE order_id = ?",
        [(_format_ts(history_end + timedelta(seconds=i + 1)), r["order_id"]) for i, r in enumerate(live)],
    )
    conn.execute(
        "UPDATE customers SET account_created_date = COALESCE("
        "(SELECT MIN(o.order_timestamp) FROM orders o WHERE o.customer_id = customers.customer_id), ?) "
        "WHERE is_synthetic_new = 1",
        (_format_ts(history_end),),
    )


def ensure_ready() -> None:
    init_db()
    conn = get_connection()
    try:
        # Re-sync seeded history whenever the processed CSVs differ from the
        # ones the store was last seeded from -- a brand-new store, one seeded
        # from fewer splits, or data regenerated since -- not only when it's
        # empty. Otherwise every customer keeps the return history of a
        # dataset that no longer exists, and live features drift from the
        # ones the model was trained and evaluated on.
        fingerprint = _processed_fingerprint()
        if _get_meta(conn, "seed_fingerprint") != fingerprint:
            seed_from_processed_csvs(conn)
            _set_meta(conn, "seed_fingerprint", fingerprint)
        if int(_get_meta(conn, "schema_version") or 0) < SCHEMA_VERSION:
            _move_live_orders_onto_dataset_clock(conn)
            _set_meta(conn, "schema_version", str(SCHEMA_VERSION))
        conn.commit()
    finally:
        conn.close()


def next_order_time(conn: sqlite3.Connection) -> datetime:
    """
    The store's clock: one second after the latest order on record.

    Live orders continue the synthetic dataset's own timeline rather than
    the wall clock. The seeded history ends mid-2024; scoring against
    datetime.utcnow() would put every returning customer's last order ~2
    years in the past -- recency, tenure and purchase frequency far beyond
    anything in training, and 30/90-day return windows that can never be
    non-zero. On the dataset clock, a returning customer's features are
    computed exactly as they were for the training rows. (An empty store
    has no timeline to continue, so it starts from the wall clock.)
    """
    row = conn.execute("SELECT MAX(order_timestamp) AS t FROM orders").fetchone()
    latest = _parse_ts(row["t"]) if row["t"] else utc_now().replace(microsecond=0)
    return latest + timedelta(seconds=1)


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
        (customer_id, _format_ts(account_created_date), pincode_tier, int(is_synthetic_new)),
    )


def get_past_orders(conn: sqlite3.Connection, customer_id: str, before_ts: datetime):
    """All of this customer's orders strictly before before_ts, oldest first."""
    rows = conn.execute(
        "SELECT * FROM orders WHERE customer_id = ? AND order_timestamp < ? ORDER BY order_timestamp ASC",
        (customer_id, _format_ts(before_ts)),
    ).fetchall()
    return [dict(r) for r in rows]


def insert_order(conn: sqlite3.Connection, order: dict) -> None:
    # Plain INSERT, not INSERT OR REPLACE: an order_id collision must fail
    # loudly rather than silently overwrite an order (and re-point every
    # decision already logged against it at a different order).
    conn.execute(
        "INSERT INTO orders (order_id, customer_id, order_timestamp, order_value, "
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


def insert_decision(conn: sqlite3.Connection, order_id: str, analyst_decision: str, decided_at: datetime) -> None:
    conn.execute(
        "INSERT INTO decisions (order_id, analyst_decision, decided_at) VALUES (?, ?, ?)",
        (order_id, analyst_decision, _format_ts(decided_at)),
    )


def get_recent_orders(conn: sqlite3.Connection, limit: int = 200):
    """Most recently scored orders (simulated or live), newest first -- used
    to repopulate the dashboard's live feed on a cold page load. Seeded
    historical orders have predicted_probability = NULL (never scored
    through the API), so they're excluded here."""
    rows = conn.execute(
        "SELECT * FROM orders WHERE predicted_probability IS NOT NULL "
        "ORDER BY order_timestamp DESC, rowid DESC LIMIT ?",
        (limit,),
    ).fetchall()
    return [dict(r) for r in rows]


def count_decisions(conn: sqlite3.Connection) -> int:
    return conn.execute("SELECT COUNT(*) AS c FROM decisions").fetchone()["c"]


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
    decisions = [dict(r) for r in rows]
    for d in decisions:
        d["decided_at"] = to_utc_iso(d["decided_at"])
    return decisions

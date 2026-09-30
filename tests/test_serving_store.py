"""
Regression guards for the serving store (api/db.py) and the endpoints that
read and write it -- the path a live order actually takes.

tests/test_feature_parity.py pins the feature *formulas*; it can't catch
skew in their *inputs*. A store seeded without one of the splits, or a
clock that puts every customer's history years in the past, feeds the same
correct formulas the wrong history -- which is exactly what happened
before: validation.csv was never seeded, and live orders were scored
against the wall clock (2026) while all history ends in mid-2024.
"""

import json
import os
import shutil
import sys
import threading
from datetime import datetime, timedelta

import pandas as pd
import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "api"))

import db  # noqa: E402
import features  # noqa: E402
import main  # noqa: E402
from features_core import compute_all_temporal_features  # noqa: E402

PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
FMT = db.TIMESTAMP_FORMAT
ORDER = {"order_value": 1800.0, "product_category": "apparel", "payment_mode": "COD", "delivery_pincode_tier": "tier2"}


def _splits():
    return {split: pd.read_csv(os.path.join(PROCESSED_DIR, f"{split}.csv")) for split in db.SEED_SPLITS}


def _all_orders():
    df = pd.concat(_splits().values(), ignore_index=True)
    df["ts"] = pd.to_datetime(df["order_timestamp"])
    return df


@pytest.fixture(scope="module")
def seeded_template(tmp_path_factory):
    """Seed once per module; each test works on its own copy."""
    path = str(tmp_path_factory.mktemp("store") / "template.db")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(db, "DB_PATH", path)
        db.ensure_ready()
    return path


@pytest.fixture
def store(seeded_template, tmp_path, monkeypatch):
    path = str(tmp_path / "riskguard.db")
    shutil.copy(seeded_template, path)
    monkeypatch.setattr(db, "DB_PATH", path)
    return path


@pytest.fixture
def client(store):
    with TestClient(main.app) as test_client:  # runs the lifespan's ensure_ready() against `store`
        yield test_client


def test_store_is_seeded_with_every_split(store):
    conn = db.get_connection()
    try:
        order_ids = {r["order_id"] for r in conn.execute("SELECT order_id FROM orders")}
        customer_ids = {r["customer_id"] for r in conn.execute("SELECT customer_id FROM customers")}
    finally:
        conn.close()
    for split, df in _splits().items():
        missing_orders = set(df["order_id"]) - order_ids
        missing_customers = set(df["customer_id"]) - customer_ids
        assert not missing_orders, f"{len(missing_orders)} {split}.csv orders missing from the store"
        assert not missing_customers, f"{len(missing_customers)} {split}.csv customers missing from the store"


def test_serving_path_reproduces_training_features(store):
    """Input parity: rebuild every held-out order's features the way a live
    order gets them -- that customer's stored history strictly before the
    order, through api/features.py -- and require the exact values the
    model was evaluated on."""
    conn = db.get_connection()
    try:
        mismatches = []
        for row in _splits()["test"].itertuples(index=False):
            ts = datetime.strptime(row.order_timestamp, FMT)
            customer = db.get_customer(conn, row.customer_id)
            customer["account_created_date"] = datetime.strptime(customer["account_created_date"], FMT)
            past = db.get_past_orders(conn, row.customer_id, ts)
            for o in past:
                o["order_timestamp"] = datetime.strptime(o["order_timestamp"], FMT)
            served = features.compute_features(customer, past, ts, row.order_value)
            mismatches += [(row.order_id, name, value, getattr(row, name))
                           for name, value in served.items() if value != getattr(row, name)]
    finally:
        conn.close()
    assert not mismatches, f"{len(mismatches)} serving/training feature mismatches, e.g. {mismatches[:3]}"


def test_live_orders_continue_the_dataset_clock(client):
    """A returning customer scored live must get features on the same
    timeline as training -- recency in days, not years, and return windows
    that can actually count their recent returns."""
    df = _all_orders()
    history_end = df["ts"].max()
    recent_returns = df[(df["returned"] == 1) & (df["ts"] >= history_end - pd.Timedelta(days=29))]
    customer_id = sorted(recent_returns["customer_id"])[0]

    result = client.post("/score", json={**ORDER, "customer_id": customer_id}).json()
    now = (history_end + pd.Timedelta(seconds=1)).to_pydatetime()
    assert result["order_timestamp"] == now.isoformat()

    history = df[df["customer_id"] == customer_id].sort_values("ts")
    first = history.iloc[0]
    created = (first["ts"].normalize() - pd.Timedelta(days=int(first["account_age_days"]))).to_pydatetime()
    past = [{"order_timestamp": t.to_pydatetime(), "order_value": v, "returned": int(r)}
            for t, v, r in zip(history["ts"], history["order_value"], history["returned"])]
    expected = compute_all_temporal_features(past, now, created, ORDER["order_value"])

    served = result["customer_features"]
    assert served["days_since_last_order"] == expected["days_since_last_order"] <= 30
    assert served["returns_last_30d"] == expected["returns_last_30d"] >= 1
    assert served["returns_last_90d"] == expected["returns_last_90d"]
    assert served["account_age_days"] == expected["account_age_days"]


def test_legacy_store_is_migrated(tmp_path, monkeypatch):
    """A store written before the fix: seeded without validation.csv, with
    live orders and API-created customers stamped with the wall clock."""
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "legacy.db"))
    db.init_db()
    conn = db.get_connection()
    try:
        with monkeypatch.context() as mp:
            mp.setattr(db, "SEED_SPLITS", ("train", "test"))
            db.seed_from_processed_csvs(conn)
        seeded_customer = conn.execute("SELECT customer_id, account_created_date FROM customers LIMIT 1").fetchone()
        db.insert_customer(conn, "SIM000001", datetime(2026, 9, 2, 8, 0, 5), "tier3", is_synthetic_new=True)
        for order_id, customer_id, ts in [
            ("SIMORD0000002", "SIM000001", "2026-09-02 08:00:05"),
            ("ORD-LEGACY001", seeded_customer["customer_id"], "2026-09-02 08:00:00"),
        ]:
            db.insert_order(conn, {
                "order_id": order_id, "customer_id": customer_id, "order_timestamp": ts, "order_value": 999.0,
                "product_category": "beauty", "payment_mode": "UPI", "discount_applied": 0.0,
                "delivery_pincode_tier": "tier3", "returned": None, "predicted_probability": 0.3,
                "risk_band": "high", "is_simulated": int(order_id.startswith("SIM")),
            })
        db.insert_decision(conn, "SIMORD0000002", "confirmed_normal", datetime(2026, 9, 2, 9, 0, 0))
        conn.commit()
    finally:
        conn.close()

    db.ensure_ready()
    db.ensure_ready()  # a second start must be a no-op

    history_end = _all_orders()["ts"].max().to_pydatetime()
    conn = db.get_connection()
    try:
        stamped = dict(conn.execute("SELECT order_id, order_timestamp FROM orders WHERE predicted_probability IS NOT NULL"))
        assert stamped == {  # re-stamped just after the history, original order preserved
            "ORD-LEGACY001": (history_end + timedelta(seconds=1)).strftime(FMT),
            "SIMORD0000002": (history_end + timedelta(seconds=2)).strftime(FMT),
        }
        created = dict(conn.execute("SELECT customer_id, account_created_date FROM customers"))
        assert created["SIM000001"] == stamped["SIMORD0000002"]
        assert created[seeded_customer["customer_id"]] == seeded_customer["account_created_date"]
        n_validation = conn.execute("SELECT COUNT(*) AS c FROM orders WHERE order_id IN (%s)" % ",".join(
            "?" * len(_splits()["validation"])), list(_splits()["validation"]["order_id"])).fetchone()["c"]
        assert n_validation == len(_splits()["validation"])
        assert [d["order_id"] for d in db.get_decisions(conn)] == ["SIMORD0000002"]
        assert db.next_order_time(conn) == history_end + timedelta(seconds=3)
    finally:
        conn.close()


def test_store_resyncs_when_the_dataset_changes(store, monkeypatch):
    """Regenerating the CSVs must reach an existing store: seeded rows take
    the new data, while live orders and decisions are left exactly as they
    were. (Without this, customers keep the return history of a dataset that
    no longer exists.)"""
    conn = db.get_connection()
    try:
        seeded = conn.execute(
            "SELECT order_id, returned FROM orders WHERE predicted_probability IS NULL LIMIT 1"
        ).fetchone()
        conn.execute("UPDATE orders SET returned = ? WHERE order_id = ?", (1 - seeded["returned"], seeded["order_id"]))
        live_ts = db.next_order_time(conn).strftime(FMT)
        db.insert_order(conn, {
            "order_id": "ORD-LIVE-0001", "customer_id": _splits()["train"].iloc[0]["customer_id"],
            "order_timestamp": live_ts, "order_value": 500.0, "product_category": "beauty", "payment_mode": "UPI",
            "discount_applied": 0.0, "delivery_pincode_tier": "metro", "returned": None,
            "predicted_probability": 0.42, "risk_band": "high", "is_simulated": 0,
        })
        conn.commit()
    finally:
        conn.close()

    monkeypatch.setattr(db, "_processed_fingerprint", lambda: "regenerated-dataset")
    db.ensure_ready()

    conn = db.get_connection()
    try:
        restored = conn.execute("SELECT returned FROM orders WHERE order_id = ?", (seeded["order_id"],)).fetchone()
        live = dict(conn.execute("SELECT * FROM orders WHERE order_id = 'ORD-LIVE-0001'").fetchone())
    finally:
        conn.close()
    assert restored["returned"] == seeded["returned"]
    assert (live["returned"], live["predicted_probability"], live["order_timestamp"]) == (None, 0.42, live_ts)


def test_stream_worker_commits_every_order_before_done(store):
    """/simulate/stream's batch runs on its own thread (which owns its SQLite
    connection) and commits whether or not anyone is still reading -- and
    "done" is only emitted after that commit."""
    emitted = []
    worker = threading.Thread(target=main._run_simulation, args=(25, 0.5, emitted.append))
    worker.start()
    worker.join(timeout=120)
    assert not worker.is_alive()
    assert emitted[-1]["done"] is True
    ids = [e["order_id"] for e in emitted[:-1]]
    assert len(ids) == len(set(ids)) == 25

    conn = db.get_connection()
    try:
        stored = {r["order_id"] for r in conn.execute(
            "SELECT order_id FROM orders WHERE order_id IN (%s)" % ",".join("?" * len(ids)), ids)}
    finally:
        conn.close()
    assert stored == set(ids)


def test_sse_stream_orders_are_decidable_after_done(client):
    with client.stream("GET", "/simulate/stream", params={"n": 12, "risk_shift": 0.3}) as resp:
        events = [json.loads(line[len("data: "):]) for line in resp.iter_lines() if line.startswith("data: ")]
    orders, done = events[:-1], events[-1]
    assert done["done"] is True and sum(done["band_counts"].values()) == len(orders) == 12
    for order in orders:
        response = client.post("/decide", json={"order_id": order["order_id"], "decision": "confirmed_normal"})
        assert response.status_code == 200


def test_abandoned_batch_does_not_recycle_order_ids(store):
    conn = db.get_connection()
    try:
        batch = main._simulate_batch(conn, n=5, risk_shift=0.0)
        abandoned = [next(batch)["order_id"] for _ in range(3)]
        batch.close()  # the viewer left before the batch committed
        committed = [o["order_id"] for o in main._simulate_batch(conn, n=5, risk_shift=0.0)]
    finally:
        conn.close()
    assert not set(abandoned) & set(committed)


def test_score_uses_the_tier_on_file_for_known_customers(client):
    train = _splits()["train"]
    customer_id, tier_on_file = train.iloc[0]["customer_id"], train.iloc[0]["delivery_pincode_tier"]
    other_tier = next(t for t in ("metro", "tier2", "tier3") if t != tier_on_file)
    seeded = client.post("/score", json={**ORDER, "delivery_pincode_tier": other_tier, "customer_id": customer_id})
    assert seeded.json()["delivery_pincode_tier"] == tier_on_file

    # A customer the API itself created is on file too (it used to be exempt by ID prefix).
    first = client.post("/score", json={**ORDER, "delivery_pincode_tier": "tier3"}).json()
    again = client.post("/score", json={**ORDER, "delivery_pincode_tier": "metro", "customer_id": first["customer_id"]})
    assert again.json()["delivery_pincode_tier"] == "tier3"

    brand_new = client.post("/score", json={**ORDER, "delivery_pincode_tier": "metro", "customer_id": "NEW-CUSTOMER-1"})
    assert brand_new.json()["delivery_pincode_tier"] == "metro"


def test_decision_timestamps_are_unambiguous_utc(client):
    order = client.post("/score", json=ORDER).json()
    logged = client.post("/decide", json={"order_id": order["order_id"], "decision": "flagged_for_verification"}).json()
    datetime.strptime(logged["decided_at"], "%Y-%m-%dT%H:%M:%SZ")

    listing = client.get("/decisions", params={"limit": 5}).json()
    assert listing["total"] == 1
    assert listing["decisions"][0]["decided_at"] == logged["decided_at"]
    assert client.get("/decisions", params={"limit": 0}).status_code == 422


def test_orders_feed_reconstructs_what_scoring_reported(client):
    scored = [
        client.post("/score", json={**ORDER, "order_value": value, "product_category": category}).json()
        for value in (350.0, 2900.0, 9000.0) for category in ("footwear", "groceries")
    ]
    feed = {o["order_id"]: o for o in client.get("/orders", params={"limit": 50}).json()["orders"]}
    fields = ("probability", "risk_band", "recommendation", "recommended_action", "optimal_threshold")
    for s in scored:
        assert tuple(feed[s["order_id"]][f] for f in fields) == tuple(s[f] for f in fields)

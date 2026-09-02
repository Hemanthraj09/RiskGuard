"""
Regression guard for _simulate_batch's two correctness guarantees (api/main.py):

  (a) Deferred commit -- nothing is written to the database until the
      generator has been driven to exhaustion. This is what makes
      GET /simulate/stream safe: a client can stop consuming mid-batch (or
      the batch can be scored slowly, one order at a time) without ever
      observing a partially-committed batch.
  (b) Batch isolation -- every order in one /simulate or /simulate/stream
      call for the SAME customer is scored against that customer's
      PRE-BATCH history only. Order 2 in a batch must not see order 1 from
      the same batch, even though order 1 has already been yielded to the
      caller by the time order 2 is scored.

Both properties existed before _simulate_batch was factored out of
POST /simulate to be shared with GET /simulate/stream; this test exercises
the shared generator directly so a future change to either endpoint can't
silently break the guarantee for the other.
"""

import os
import sys
from datetime import datetime, timedelta

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
API_DIR = os.path.join(PROJECT_ROOT, "api")
sys.path.insert(0, API_DIR)

import db  # noqa: E402
import main  # noqa: E402

CUSTOMER_ID = "TESTCUST001"


def _fresh_db(tmp_path):
    db_path = str(tmp_path / "isolation_test.db")
    db.DB_PATH = db_path
    db.init_db()
    return db_path


def _seed_customer_with_one_past_order(conn):
    created = datetime(2024, 1, 1, 0, 0, 0)
    db.insert_customer(conn, CUSTOMER_ID, created, "metro", is_synthetic_new=False)
    db.insert_order(conn, {
        "order_id": "ORD-SEED-0001",
        "customer_id": CUSTOMER_ID,
        "order_timestamp": "2024-02-01 00:00:00",
        "order_value": 1000.0,
        "product_category": "apparel",
        "payment_mode": "UPI",
        "discount_applied": 0.0,
        "delivery_pincode_tier": "metro",
        "returned": 1,
        "predicted_probability": None,
        "risk_band": None,
        "is_simulated": 0,
    })
    conn.commit()


def _order_count_for(conn, customer_id):
    return conn.execute(
        "SELECT COUNT(*) AS c FROM orders WHERE customer_id = ?", (customer_id,)
    ).fetchone()["c"]


def test_simulate_batch_defers_commit_until_exhausted(tmp_path, monkeypatch):
    _fresh_db(tmp_path)
    conn = db.get_connection()
    try:
        _seed_customer_with_one_past_order(conn)
        monkeypatch.setattr(main, "EXISTING_CUSTOMER_PROB", 1.0)
        monkeypatch.setattr(db, "random_existing_customer_ids", lambda _conn, _n: [CUSTOMER_ID])

        gen = main._simulate_batch(conn, n=3, risk_shift=0.0)

        next(gen)
        assert _order_count_for(conn, CUSTOMER_ID) == 1, "first yield must not have written anything yet"

        next(gen)
        assert _order_count_for(conn, CUSTOMER_ID) == 1, "second yield must still not have committed"

        next(gen)
        assert _order_count_for(conn, CUSTOMER_ID) == 1, "third (last) yield must still not have committed"

        try:
            next(gen)
            assert False, "generator should be exhausted after n=3 yields"
        except StopIteration:
            pass

        assert _order_count_for(conn, CUSTOMER_ID) == 4, "exhausting the generator must commit all 3 new orders"
    finally:
        conn.close()


def test_simulate_batch_isolates_same_customer_orders(tmp_path, monkeypatch):
    _fresh_db(tmp_path)
    conn = db.get_connection()
    try:
        _seed_customer_with_one_past_order(conn)
        monkeypatch.setattr(main, "EXISTING_CUSTOMER_PROB", 1.0)
        monkeypatch.setattr(db, "random_existing_customer_ids", lambda _conn, _n: [CUSTOMER_ID])

        results = list(main._simulate_batch(conn, n=3, risk_shift=0.0))

        assert len(results) == 3
        assert all(r["customer_id"] == CUSTOMER_ID for r in results)

        # History-derived features must be IDENTICAL across all 3 -- each was
        # scored against the same frozen 1-past-order snapshot, not against
        # the other orders already yielded earlier in this same batch. (If
        # isolation were broken, order 2 would see order 1 as history and
        # these would diverge.)
        history_fields = [
            "bayesian_return_rate",
            "customer_purchase_frequency",
            "account_age_days",
            "days_since_last_order",
            "returns_last_30d",
            "returns_last_90d",
        ]
        first_features = results[0]["customer_features"]
        for r in results[1:]:
            for field in history_fields:
                assert r["customer_features"][field] == first_features[field], (
                    f"{field} diverged within the same batch -- batch isolation is broken"
                )

        # Confirm it's not just a coincidence of the sentinel/cold-start path:
        # this customer really does have exactly the 1 seeded past order in
        # every one of the 3 snapshots, not 0 and not a growing count.
        assert first_features["days_since_last_order"] >= 0  # saw the seeded order, not a cold start

        # A call AFTER the batch (i.e. not part of it) must see all 3 new
        # orders as real history -- isolation is about same-batch visibility
        # only, not about hiding the orders forever.
        future_past_orders = db.get_past_orders(conn, CUSTOMER_ID, datetime(2030, 1, 1))
        assert len(future_past_orders) == 4  # 1 seeded + 3 simulated
    finally:
        conn.close()

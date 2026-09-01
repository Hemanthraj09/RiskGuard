"""
RiskGuard — FastAPI serving layer.

Minimal surface per the spec: POST /score, POST /simulate, GET /metrics.
No auth, no policy engine, no autonomous execution — this is a scoring /
decision-support tool only (defense-only constraint).

Run with:
    uvicorn api.main:app --reload --port 8000
(from the riskguard/ directory)
"""

import json
import os
import sys
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Literal, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import db  # noqa: E402
import features  # noqa: E402
import scoring  # noqa: E402
from simulate_gen import (  # noqa: E402
    CATEGORIES, PAYMENT_MODES, PINCODE_TIERS,
    EXISTING_CUSTOMER_PROB, sample_order_fields, make_new_customer_id,
)

CategoryLiteral = Literal[tuple(CATEGORIES)]
PaymentModeLiteral = Literal[tuple(PAYMENT_MODES)]
PincodeTierLiteral = Literal[tuple(PINCODE_TIERS)]


@asynccontextmanager
async def lifespan(app: FastAPI):
    db.ensure_ready()
    yield


app = FastAPI(title="RiskGuard API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────
# Schemas
# ─────────────────────────────────────────────────────────────
class ScoreRequest(BaseModel):
    order_value: float = Field(gt=0, le=100000)
    product_category: CategoryLiteral
    payment_mode: PaymentModeLiteral
    delivery_pincode_tier: PincodeTierLiteral
    discount_applied: float = Field(default=0.0, ge=0, le=100)
    customer_id: Optional[str] = None


class SimulateRequest(BaseModel):
    n: int = Field(default=100, ge=1, le=500)
    risk_shift: float = Field(default=0.0, ge=0.0, le=1.0)


AnalystDecisionLiteral = Literal["confirmed_normal", "flagged_for_verification"]


class DecideRequest(BaseModel):
    order_id: str
    decision: AnalystDecisionLiteral


# ─────────────────────────────────────────────────────────────
# Core order-scoring pipeline (shared by /score and /simulate)
# ─────────────────────────────────────────────────────────────
def _resolve_customer(conn, customer_id: Optional[str], order_timestamp: datetime, fallback_pincode_tier: str):
    """Returns (customer_id, customer_record, past_orders, is_new_customer)."""
    if customer_id:
        record = db.get_customer(conn, customer_id)
        if record is None:
            created = {"account_created_date": order_timestamp}
            db.insert_customer(conn, customer_id, order_timestamp, fallback_pincode_tier, is_synthetic_new=True)
            return customer_id, created, [], True
        record["account_created_date"] = datetime.strptime(record["account_created_date"], "%Y-%m-%d %H:%M:%S")
        past = db.get_past_orders(conn, customer_id, order_timestamp)
        for o in past:
            o["order_timestamp"] = datetime.strptime(o["order_timestamp"], "%Y-%m-%d %H:%M:%S")
        return customer_id, record, past, False

    new_id = f"ADHOC{uuid.uuid4().hex[:8].upper()}"
    db.insert_customer(conn, new_id, order_timestamp, fallback_pincode_tier, is_synthetic_new=True)
    return new_id, {"account_created_date": order_timestamp}, [], True


def _score_and_persist(conn, customer_id: str, customer_record: dict, past_orders: list,
                        order_fields: dict, order_timestamp: datetime, order_id: str, is_simulated: bool) -> dict:
    customer_features = features.compute_features(
        customer_record, past_orders, order_timestamp, order_fields["order_value"]
    )
    result = scoring.score_order(order_fields, customer_features)

    db.insert_order(conn, {
        "order_id": order_id,
        "customer_id": customer_id,
        "order_timestamp": order_timestamp.strftime("%Y-%m-%d %H:%M:%S"),
        "order_value": order_fields["order_value"],
        "product_category": order_fields["product_category"],
        "payment_mode": order_fields["payment_mode"],
        "discount_applied": order_fields.get("discount_applied", 0.0),
        "delivery_pincode_tier": order_fields["delivery_pincode_tier"],
        "returned": None,
        "predicted_probability": result["probability"],
        "risk_band": result["risk_band"],
        "is_simulated": int(is_simulated),
    })

    return {
        "order_id": order_id,
        "customer_id": customer_id,
        "order_timestamp": order_timestamp.isoformat(),
        **order_fields,
        **result,
        "customer_features": customer_features,
    }


# ─────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────
@app.post("/score")
def score(req: ScoreRequest):
    conn = db.get_connection()
    try:
        order_timestamp = datetime.utcnow()
        order_fields = req.model_dump(exclude={"customer_id"})

        customer_id, customer_record, past_orders, _ = _resolve_customer(
            conn, req.customer_id, order_timestamp, req.delivery_pincode_tier
        )
        if not customer_id.startswith(("ADHOC", "SIM")):
            order_fields["delivery_pincode_tier"] = customer_record.get("pincode_tier", req.delivery_pincode_tier)

        order_id = f"ORD-{uuid.uuid4().hex[:10].upper()}"
        result = _score_and_persist(
            conn, customer_id, customer_record, past_orders, order_fields, order_timestamp, order_id, False
        )
        conn.commit()
        return result
    finally:
        conn.close()


def _simulate_batch(conn, n: int, risk_shift: float):
    # Outcome-visibility policy: a real order's return outcome isn't known
    # for days or weeks, so orders generated within the SAME batch must not
    # see each other -- not their existence, not their (nonexistent-yet)
    # outcomes -- regardless of what order they're processed in. We enforce
    # this by snapshotting each customer's PRE-EXISTING persisted history
    # into an in-memory cache the first time that customer is touched in
    # this batch (which, since nothing from this batch has been written
    # yet, is necessarily pre-batch state), scoring every order against
    # that frozen snapshot only, and deferring ALL database writes (new
    # customers + new orders) until the whole batch has been scored. Only
    # *future* /simulate or /score calls will see these orders as history.
    #
    # Shared by /simulate (collects every yielded result before returning)
    # and /simulate/stream (forwards each yielded result to the client as
    # it's scored) -- both drive this generator to exhaustion before its
    # caller commits anything, so the isolation guarantee above holds
    # identically for either consumer.
    rng = np.random.RandomState()  # unseeded: each simulate call looks "live" and different
    base_ts = datetime.utcnow()
    seq_start = db.next_order_seq(conn)
    new_cust_seq = db.next_synthetic_customer_seq(conn)

    snapshot_cache: dict = {}  # customer_id -> (record, past_orders), pre-batch only
    pending_new_customers: list = []  # (customer_id, created_at, pincode_tier)
    pending_orders: list = []  # order dicts ready for db.insert_order

    def load_snapshot(customer_id: str):
        if customer_id in snapshot_cache:
            return snapshot_cache[customer_id]
        record = db.get_customer(conn, customer_id)
        record["account_created_date"] = datetime.strptime(record["account_created_date"], "%Y-%m-%d %H:%M:%S")
        past_orders = db.get_past_orders(conn, customer_id, base_ts)
        for o in past_orders:
            o["order_timestamp"] = datetime.strptime(o["order_timestamp"], "%Y-%m-%d %H:%M:%S")
        snapshot_cache[customer_id] = (record, past_orders)
        return record, past_orders

    for i in range(n):
        order_timestamp = base_ts + timedelta(seconds=i)  # strictly increasing order_id/timestamp bookkeeping only
        order_fields = sample_order_fields(risk_shift, rng)

        use_existing = rng.random() < EXISTING_CUSTOMER_PROB
        customer_id = None
        if use_existing:
            candidates = db.random_existing_customer_ids(conn, 1)
            if candidates:
                customer_id = candidates[0]

        if customer_id:
            record, past_orders = load_snapshot(customer_id)
            order_fields["delivery_pincode_tier"] = record["pincode_tier"]
        else:
            customer_id = make_new_customer_id(new_cust_seq)
            new_cust_seq += 1
            record = {"account_created_date": order_timestamp}
            past_orders = []
            pending_new_customers.append((customer_id, order_timestamp, order_fields["delivery_pincode_tier"]))

        customer_features = features.compute_features(
            record, past_orders, order_timestamp, order_fields["order_value"]
        )
        result = scoring.score_order(order_fields, customer_features)

        order_id = f"SIMORD{seq_start + i:07d}"
        pending_orders.append({
            "order_id": order_id,
            "customer_id": customer_id,
            "order_timestamp": order_timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "order_value": order_fields["order_value"],
            "product_category": order_fields["product_category"],
            "payment_mode": order_fields["payment_mode"],
            "discount_applied": order_fields.get("discount_applied", 0.0),
            "delivery_pincode_tier": order_fields["delivery_pincode_tier"],
            "returned": None,
            "predicted_probability": result["probability"],
            "risk_band": result["risk_band"],
            "is_simulated": 1,
        })
        yield {
            "order_id": order_id,
            "customer_id": customer_id,
            "order_timestamp": order_timestamp.isoformat(),
            **order_fields,
            **result,
            "customer_features": customer_features,
        }

    # Commit the whole batch atomically, only now that every order has been
    # scored against pre-batch history (i.e. only once this generator has
    # been driven to exhaustion by its caller).
    for cid, created_at, tier in pending_new_customers:
        db.insert_customer(conn, cid, created_at, tier, is_synthetic_new=True)
    for order in pending_orders:
        db.insert_order(conn, order)
    conn.commit()


@app.post("/simulate")
def simulate(req: SimulateRequest):
    conn = db.get_connection()
    try:
        results = list(_simulate_batch(conn, req.n, req.risk_shift))

        band_counts = {"low": 0, "medium": 0, "high": 0}
        for r in results:
            band_counts[r["risk_band"]] += 1

        return {"orders": results, "risk_shift": req.risk_shift, "band_counts": band_counts}
    finally:
        conn.close()


@app.get("/simulate/stream")
def simulate_stream(
    n: int = Query(default=100, ge=1, le=500),
    risk_shift: float = Query(default=0.0, ge=0.0, le=1.0),
):
    """SSE variant of /simulate for the Simulation Console's live feed: same
    batch-isolation guarantee and same DB-write timing as the POST endpoint
    (see _simulate_batch) -- this only changes how results reach the client,
    streaming each one as it's scored instead of waiting for the whole batch."""
    def event_stream():
        conn = db.get_connection()
        try:
            band_counts = {"low": 0, "medium": 0, "high": 0}
            for result in _simulate_batch(conn, n, risk_shift):
                band_counts[result["risk_band"]] += 1
                yield f"data: {json.dumps(result)}\n\n"
            yield f"data: {json.dumps({'done': True, 'band_counts': band_counts})}\n\n"
        finally:
            conn.close()

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache"},
    )


@app.get("/orders")
def orders(limit: int = Query(default=200, ge=1, le=1000)):
    """Repopulates the dashboard's live feed on a cold page load. Only
    orders that have actually been scored through the API (simulated or
    live) are returned -- customer_features and top_contributors require
    the original scoring pass (temporal features off frozen history, SHAP
    off the live model) and aren't reconstructed here, so they come back
    null; recommendation/recommended_action/optimal_threshold are cheap,
    deterministic derivations of the already-stored risk_band and are
    reconstructed exactly."""
    conn = db.get_connection()
    try:
        rows = db.get_recent_orders(conn, limit=limit)
        results = []
        for r in rows:
            probability = r["predicted_probability"]
            band = r["risk_band"]
            results.append({
                "order_id": r["order_id"],
                "customer_id": r["customer_id"],
                "order_timestamp": r["order_timestamp"].replace(" ", "T"),
                "order_value": r["order_value"],
                "product_category": r["product_category"],
                "payment_mode": r["payment_mode"],
                "discount_applied": r["discount_applied"],
                "delivery_pincode_tier": r["delivery_pincode_tier"],
                "probability": probability,
                "risk_band": band,
                "recommendation": "flag_for_verification" if probability >= scoring.OPTIMAL_THRESHOLD else "accept_normally",
                "recommended_action": scoring.recommend_action(band, r["payment_mode"], r["product_category"]),
                "optimal_threshold": scoring.OPTIMAL_THRESHOLD,
                "top_contributors": None,
                "customer_features": None,
            })
        return {"orders": results}
    finally:
        conn.close()


@app.get("/metrics")
def metrics():
    return scoring._eval_results


@app.post("/decide")
def decide(req: DecideRequest):
    """
    Logs a human analyst's verify/decide action on a scored order -- the
    "verifier" half of "detector, verifier, or auto-responder" from the
    track brief. This only RECORDS a decision; it never blocks, denies, or
    refunds anything itself. Re-deciding the same order logs a new row
    (an analyst can change their mind), so this is a genuine audit trail.
    """
    conn = db.get_connection()
    try:
        order_exists = conn.execute(
            "SELECT 1 FROM orders WHERE order_id = ?", (req.order_id,)
        ).fetchone()
        if not order_exists:
            raise HTTPException(status_code=404, detail=f"Order {req.order_id} not found")

        decided_at = datetime.utcnow()
        db.insert_decision(conn, req.order_id, req.decision, decided_at)
        conn.commit()
        return {"order_id": req.order_id, "decision": req.decision, "decided_at": decided_at.isoformat()}
    finally:
        conn.close()


@app.get("/decisions")
def decisions(limit: int = 100):
    """Outcome-vs-prediction view: recent analyst decisions alongside what
    the model predicted for that order at scoring time."""
    conn = db.get_connection()
    try:
        rows = db.get_decisions(conn, limit=limit)
        return {"decisions": rows}
    finally:
        conn.close()


@app.get("/health")
def health():
    return {"status": "ok"}

"""
RiskGuard -- real HTTP round-trip train/serve parity check (one-off, not a
regression guard; see tests/test_feature_parity.py and
tests/test_shap_calibration_consistency.py for the permanent in-process
guards this project already has).

A prior verification pass confirmed POST /score's train/serve consistency
by calling api/scoring.py's functions directly in-process -- a real check,
but not a literal HTTP round trip. This script does the real version:
starts nothing itself (point it at an already-running `uvicorn api.main:app`)
and sends an actual POST over the network.

Why this can't just be "pick a test.csv row, POST its fields with its real
customer_id, expect probabilities to match": /score computes a customer's
past orders as of datetime.utcnow() (see api/main.py's _resolve_customer),
not as of the historical row's original timestamp. For any customer with
orders after the chosen row, that pulls in orders from the future relative
to the row being tested -- a different, invalid feature vector, not the one
used at training time. Comparing against that would fail for the wrong
reason and prove nothing.

The fix: test a customer's FIRST order. A first order has no past orders
regardless of what "now" is, so the "past orders as of now" problem doesn't
arise -- and POST /score with customer_id=None hits the exact same cold-start
path (api/main.py's _resolve_customer, the `else` branch: brand-new ad-hoc
customer, past_orders=[]).

One subtlety this script does NOT paper over: cold-start account_age_days.
/score's ad-hoc path sets account_created_date = utcnow(), so it always
computes account_age_days=0. But data/generate_data.py gives every customer
a signup date measurably before their first order (min observed: 5 days
across the full dataset) -- so no real "first order" row has
account_age_days=0, and picking one whose *other* four cold-start features
match (bayesian_return_rate=0.2, purchase_frequency=0.0,
days_since_last_order=-1, returns_30d/90d=0) will still differ from the API
in account_age_days specifically. That's not a parity bug; it's the ad-hoc
path modeling a truly brand-new signup, which no historical row represents.
So this script reports TWO offline numbers:
  (1) using the row's own stored account_age_days, exactly as instructed
      ("using the row's actual stored feature values from test.csv") --
      this will legitimately NOT match the API, and that's expected, not a
      failure.
  (2) using account_age_days=0, i.e. the actual feature vector /score's
      ad-hoc path produces -- this is the real parity claim, and IS
      expected to match the API to floating-point tolerance.

Usage:
    uvicorn api.main:app --port 8000     # in one terminal
    python experiments/http_score_parity_check.py [base_url]
"""

import os
import sys
import json
import pickle

import numpy as np
import pandas as pd
import requests

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
MODEL_DIR = os.path.join(PROJECT_ROOT, "model")
ARTIFACTS_DIR = os.path.join(MODEL_DIR, "artifacts")
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

sys.path.insert(0, MODEL_DIR)
from train import prepare_features  # noqa: E402


def pick_first_order_row() -> pd.Series:
    """A test.csv row that is its customer's first order in the dataset:
    bayesian_return_rate at the prior default (~0.2), no purchase frequency,
    no prior order to measure recency against. Sidesteps the utcnow()
    "future orders" trap entirely, since a first order has no history under
    any current timestamp."""
    df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
    cold = df[(df["days_since_last_order"] == -1) & (df["customer_purchase_frequency"] == 0.0)]
    if len(cold) == 0:
        raise RuntimeError("No first-order rows found in test.csv -- can't run this check.")
    return cold.sort_values("order_id").iloc[0]


def build_raw_row(row: pd.Series, account_age_days) -> pd.DataFrame:
    """Exactly api/scoring.py's _build_raw_row, by hand, off the row's
    already-computed feature values instead of recomputing them."""
    return pd.DataFrame([{
        "order_value": float(row["order_value"]),
        "product_category": row["product_category"],
        "payment_mode": row["payment_mode"],
        "discount_applied": float(row["discount_applied"]),
        "delivery_pincode_tier": row["delivery_pincode_tier"],
        "bayesian_return_rate": float(row["bayesian_return_rate"]),
        "customer_purchase_frequency": float(row["customer_purchase_frequency"]),
        "account_age_days": account_age_days,
        "days_since_last_order": int(row["days_since_last_order"]),
        "returns_last_30d": int(row["returns_last_30d"]),
        "returns_last_90d": int(row["returns_last_90d"]),
        "category_x_payment_mode": f"{row['product_category']}_{row['payment_mode']}",
        "order_value_vs_customer_avg": float(row["order_value_vs_customer_avg"]),
        "pincode_tier_x_category": f"{row['delivery_pincode_tier']}_{row['product_category']}",
    }])


def offline_probability(raw_df: pd.DataFrame, model, calibrator, encoded_columns) -> float:
    """Exactly api/scoring.py's score_order(), by hand: encode -> raw
    predict_proba -> isotonic calibration -> round(., 4) (score_order's
    response field is itself rounded to 4dp, so matching that rounding here
    is part of reproducing it exactly, not a tolerance fudge). No SHAP --
    this check is about the probability the dashboard shows, not the
    explanation panel."""
    X, _ = prepare_features(raw_df, encoded_columns=encoded_columns)
    raw_prob = float(model.predict_proba(X)[:, 1][0])
    calibrated_prob = float(calibrator.predict([raw_prob])[0])
    return round(calibrated_prob, 4)


def main(base_url: str):
    print("=" * 70)
    print("RiskGuard -- real HTTP /score train/serve parity check")
    print("=" * 70)

    row = pick_first_order_row()
    print(f"\nChosen first-order row: {row['order_id']} (customer {row['customer_id']})")
    print(f"  order_value={row['order_value']}, category={row['product_category']}, "
          f"payment={row['payment_mode']}, tier={row['delivery_pincode_tier']}, "
          f"discount={row['discount_applied']}")
    print(f"  stored features: bayesian_return_rate={row['bayesian_return_rate']}, "
          f"purchase_frequency={row['customer_purchase_frequency']}, "
          f"account_age_days={row['account_age_days']} (row's real signup tenure), "
          f"days_since_last_order={row['days_since_last_order']}, "
          f"returns_30d/90d={row['returns_last_30d']}/{row['returns_last_90d']}, "
          f"value_vs_avg={row['order_value_vs_customer_avg']}")

    with open(os.path.join(ARTIFACTS_DIR, "model.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(ARTIFACTS_DIR, "calibrator.pkl"), "rb") as f:
        calibrator = pickle.load(f)
    with open(os.path.join(ARTIFACTS_DIR, "metadata.json")) as f:
        metadata = json.load(f)
    encoded_columns = metadata["encoded_columns"]

    # (1) Offline, using the row's OWN stored account_age_days -- literal
    # instruction, but NOT expected to match the API (see module docstring).
    raw_as_stored = build_raw_row(row, account_age_days=int(row["account_age_days"]))
    prob_as_stored = offline_probability(raw_as_stored, model, calibrator, encoded_columns)

    # (2) Offline, using account_age_days=0 -- the feature vector /score's
    # customer_id=None ad-hoc path actually produces. This IS the real
    # parity claim.
    raw_cold_start = build_raw_row(row, account_age_days=0)
    prob_cold_start_offline = offline_probability(raw_cold_start, model, calibrator, encoded_columns)

    # Live HTTP round trip: customer_id=None -> genuinely new ad-hoc
    # customer -> API computes the same cold-start features itself.
    body = {
        "order_value": float(row["order_value"]),
        "product_category": row["product_category"],
        "payment_mode": row["payment_mode"],
        "delivery_pincode_tier": row["delivery_pincode_tier"],
        "discount_applied": float(row["discount_applied"]),
        "customer_id": None,
    }
    resp = requests.post(f"{base_url}/score", json=body, timeout=30)
    resp.raise_for_status()
    api_result = resp.json()
    api_probability = api_result["probability"]
    api_customer_features = api_result["customer_features"]

    print(f"\nAPI's computed cold-start customer_features: {api_customer_features}")

    print("\n" + "-" * 70)
    print("RESULTS")
    print("-" * 70)
    print(f"(1) Offline, row's real account_age_days={int(row['account_age_days'])}: "
          f"probability = {prob_as_stored:.6f}")
    print(f"(2) Offline, account_age_days=0 (matches /score's ad-hoc path): "
          f"probability = {prob_cold_start_offline:.6f}")
    print(f"    Live API   POST /score (customer_id=None):     "
          f"probability = {api_probability:.6f}")

    diff_vs_stored = abs(prob_as_stored - api_probability)
    diff_vs_cold_start = abs(prob_cold_start_offline - api_probability)
    print(f"\n|(1) - API| = {diff_vs_stored:.6f}  "
          f"(expected to be nonzero -- different account_age_days, not a bug)")
    print(f"|(2) - API| = {diff_vs_cold_start:.6f}  "
          f"(expected to be ~0 -- this is the actual serving-path parity check)")

    tolerance = 1e-4  # score_order()'s response field is itself round(., 4)
    if diff_vs_cold_start <= tolerance:
        print(f"\nPASS: live HTTP /score matches the offline model+calibrator artifacts "
              f"to within {tolerance} for an identical cold-start feature vector.")
    else:
        print(f"\nFAIL: live HTTP /score diverges from the offline artifacts by "
              f"{diff_vs_cold_start:.6f}, exceeding the {tolerance} tolerance -- "
              f"investigate before trusting /score's output.")
        sys.exit(1)


if __name__ == "__main__":
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
    main(base_url)

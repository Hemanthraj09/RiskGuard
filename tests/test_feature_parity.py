"""
Regression guard for the shared feature module (features_core.py).

Constructs one synthetic customer history and feeds it through both usage
paths -- data.generate_data's compute_customer_features_for_order (offline
generation) and api.features.compute_features (online scoring) -- and
asserts identical output. Since both now call the same underlying
features_core functions, this mostly guards the thin wrapping/rounding
code around them, but it's a cheap, permanent tripwire against the two
call sites drifting apart again in the future.
"""

import os
import sys
from datetime import datetime, timedelta

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "data"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "api"))

from generate_data import compute_customer_features_for_order  # noqa: E402
from features import compute_features  # noqa: E402
from features_core import compute_returns_in_window  # noqa: E402


def _build_history(base_time: datetime):
    """A customer with 4 past orders (mixed outcomes), oldest first."""
    return [
        {"order_timestamp": base_time - timedelta(days=120), "order_value": 1500.0, "returned": 0},
        {"order_timestamp": base_time - timedelta(days=80), "order_value": 2200.0, "returned": 1},
        {"order_timestamp": base_time - timedelta(days=45), "order_value": 1800.0, "returned": 0},
        {"order_timestamp": base_time - timedelta(days=10), "order_value": 3000.0, "returned": 1},
    ]


def test_parity_repeat_customer():
    now = datetime(2024, 6, 1, 12, 0, 0)
    account_created = now - timedelta(days=400)
    past = _build_history(now)
    order_value = 2500.0

    generator_result = compute_customer_features_for_order(past, now, account_created, order_value)
    api_result = compute_features({"account_created_date": account_created}, past, now, order_value)

    assert generator_result == api_result


def test_parity_cold_start_customer():
    now = datetime(2024, 6, 1, 12, 0, 0)
    account_created = now  # brand new account, first order
    past = []
    order_value = 999.0

    generator_result = compute_customer_features_for_order(past, now, account_created, order_value)
    api_result = compute_features({"account_created_date": account_created}, past, now, order_value)

    assert generator_result == api_result
    assert generator_result["days_since_last_order"] == -1
    assert generator_result["order_value_vs_customer_avg"] == 1.0
    assert generator_result["customer_purchase_frequency"] == 0.0
    assert generator_result["bayesian_return_rate"] == round(2 / 10, 6)  # prior mean 0.20


def test_parity_with_unresolved_recent_order():
    """The API-only case: a just-simulated order with returned=None must not
    be counted as a known non-return in the customer's history stats."""
    now = datetime(2024, 6, 1, 12, 0, 0)
    account_created = now - timedelta(days=200)
    past = _build_history(now) + [
        {"order_timestamp": now - timedelta(days=1), "order_value": 2000.0, "returned": None},
    ]
    order_value = 1700.0

    api_result = compute_features({"account_created_date": account_created}, past, now, order_value)

    # bayesian_return_rate must match the 4-resolved-order history, not 5
    resolved_only = _build_history(now)
    expected = compute_customer_features_for_order(resolved_only, now, account_created, order_value)
    assert api_result["bayesian_return_rate"] == expected["bayesian_return_rate"]
    assert api_result["returns_last_30d"] == expected["returns_last_30d"]
    assert api_result["returns_last_90d"] == expected["returns_last_90d"]
    # but value-ratio and purchase-frequency DO count the unresolved order --
    # its value and existence are known immediately, only the outcome isn't
    assert api_result["customer_purchase_frequency"] != expected["customer_purchase_frequency"]
    assert api_result["days_since_last_order"] == 1


# ── Edge cases added in the final evaluation pass -- these boundary
# conditions are exactly where two independent implementations were most
# likely to have quietly diverged before the features_core.py refactor. ──

def test_parity_exactly_one_past_order():
    now = datetime(2024, 6, 1, 12, 0, 0)
    account_created = now - timedelta(days=60)
    past = [{"order_timestamp": now - timedelta(days=20), "order_value": 1200.0, "returned": 0}]
    order_value = 1800.0

    generator_result = compute_customer_features_for_order(past, now, account_created, order_value)
    api_result = compute_features({"account_created_date": account_created}, past, now, order_value)

    assert generator_result == api_result
    assert generator_result["days_since_last_order"] == 20
    assert generator_result["order_value_vs_customer_avg"] == round(1800.0 / 1200.0, 4)
    # 1 past order, 0 returns: (0 + 2) / (1 + 2 + 8) = 2/11
    assert generator_result["bayesian_return_rate"] == round(2 / 11, 6)


def test_parity_same_day_repeat_order():
    """Most recent past order is 0 days before the current one (two orders
    placed the same day) -- days_since_last_order must be 0, not treated as
    a cold-start / no-history sentinel (-1)."""
    now = datetime(2024, 6, 1, 18, 0, 0)
    account_created = now - timedelta(days=100)
    past = [{"order_timestamp": now - timedelta(hours=3), "order_value": 900.0, "returned": 0}]
    order_value = 1100.0

    generator_result = compute_customer_features_for_order(past, now, account_created, order_value)
    api_result = compute_features({"account_created_date": account_created}, past, now, order_value)

    assert generator_result == api_result
    assert generator_result["days_since_last_order"] == 0


def test_window_boundary_is_inclusive_in_both_paths():
    """An order timestamped EXACTLY 30 (or 90) days before `now` must be
    counted in the returns_last_30d/90d window -- features_core.py's
    boundary check is `order_timestamp >= cutoff`, i.e. inclusive. Verify
    this directly against the shared function (both call sites use it, so
    this pins the contract itself, not just that the two sides agree)."""
    now = datetime(2024, 6, 1, 0, 0, 0)

    on_30d_boundary = [{"order_timestamp": now - timedelta(days=30), "order_value": 1000.0, "returned": 1}]
    just_outside_30d = [{"order_timestamp": now - timedelta(days=30, seconds=1), "order_value": 1000.0, "returned": 1}]
    on_90d_boundary = [{"order_timestamp": now - timedelta(days=90), "order_value": 1000.0, "returned": 1}]
    just_outside_90d = [{"order_timestamp": now - timedelta(days=90, seconds=1), "order_value": 1000.0, "returned": 1}]

    assert compute_returns_in_window(on_30d_boundary, now, 30) == 1
    assert compute_returns_in_window(just_outside_30d, now, 30) == 0
    assert compute_returns_in_window(on_90d_boundary, now, 90) == 1
    assert compute_returns_in_window(just_outside_90d, now, 90) == 0

    # And confirm both usage paths still agree at this exact boundary.
    account_created = now - timedelta(days=200)
    order_value = 1500.0
    generator_result = compute_customer_features_for_order(on_30d_boundary, now, account_created, order_value)
    api_result = compute_features({"account_created_date": account_created}, on_30d_boundary, now, order_value)
    assert generator_result == api_result
    assert generator_result["returns_last_30d"] == 1

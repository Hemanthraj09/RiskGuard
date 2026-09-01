"""
RiskGuard — shared temporal-feature core.

Pure feature-computation functions used by BOTH the offline data generator
(data/generate_data.py) and the online scoring API (api/features.py). Before
this module existed, the two implemented the same formulas independently —
any drift between them (a different prior, a different window, a different
cold-start default) would silently produce train/serve skew that shows up
only in live demo behavior, never in reported metrics. Now there is exactly
one implementation to keep correct.

No pandas, no SQLite, no I/O — every function takes plain inputs so it can
be tested and reasoned about in isolation (see tests/test_feature_parity.py).

`past_orders` is a list of dict-like objects, oldest-first, each exposing:
  - order_timestamp: datetime
  - order_value: float
  - returned: Optional[int]   # 0, 1, or None if the outcome isn't known yet
Callers are responsible for the boundary (only orders strictly before the
current order's timestamp); these functions don't filter by time except for
the 30/90-day windows, which are relative to `current_time`.
"""

from datetime import datetime, timedelta
from statistics import mean
from typing import Sequence

PRIOR_ALPHA_DEFAULT = 2
PRIOR_BETA_DEFAULT = 8


def _get(order, key):
    return order[key] if isinstance(order, dict) else getattr(order, key)


def _resolved(past_orders: Sequence) -> list:
    """Only orders with a KNOWN outcome count toward return-rate statistics —
    an unresolved (pending) order can't yet tell us anything about whether
    it will be returned."""
    return [o for o in past_orders if _get(o, "returned") is not None]


def compute_bayesian_return_rate(
    past_orders: Sequence, prior_alpha: float = PRIOR_ALPHA_DEFAULT, prior_beta: float = PRIOR_BETA_DEFAULT
) -> float:
    resolved = _resolved(past_orders)
    past_returns = sum(1 for o in resolved if _get(o, "returned") == 1)
    past_count = len(resolved)
    return (past_returns + prior_alpha) / (past_count + prior_alpha + prior_beta)


def compute_returns_in_window(past_orders: Sequence, current_time: datetime, window_days: int) -> int:
    cutoff = current_time - timedelta(days=window_days)
    resolved = _resolved(past_orders)
    return sum(1 for o in resolved if _get(o, "order_timestamp") >= cutoff and _get(o, "returned") == 1)


def compute_order_value_ratio(order_value: float, past_orders: Sequence) -> float:
    if len(past_orders) == 0:
        return 1.0
    avg_value = mean(_get(o, "order_value") for o in past_orders)
    return order_value / avg_value if avg_value > 0 else 1.0


def compute_purchase_frequency(past_orders: Sequence, account_created_date: datetime, current_time: datetime) -> float:
    if len(past_orders) == 0:
        return 0.0
    span_months = max((current_time - account_created_date).days / 30.0, 0.5)
    return len(past_orders) / span_months


def compute_days_since_last_order(past_orders: Sequence, current_time: datetime) -> int:
    if len(past_orders) == 0:
        return -1  # sentinel: first order, not an imputed fake value
    last_ts = _get(past_orders[-1], "order_timestamp")
    return (current_time - last_ts).days


def compute_account_age_days(account_created_date: datetime, current_time: datetime) -> int:
    return (current_time - account_created_date).days


def compute_all_temporal_features(
    past_orders: Sequence,
    current_time: datetime,
    account_created_date: datetime,
    order_value: float,
    prior_alpha: float = PRIOR_ALPHA_DEFAULT,
    prior_beta: float = PRIOR_BETA_DEFAULT,
) -> dict:
    """Convenience wrapper bundling every feature above into the exact dict
    shape consumed by both the generator and the API — the single entry
    point exercised by the parity test."""
    return {
        "bayesian_return_rate": compute_bayesian_return_rate(past_orders, prior_alpha, prior_beta),
        "customer_purchase_frequency": compute_purchase_frequency(past_orders, account_created_date, current_time),
        "account_age_days": compute_account_age_days(account_created_date, current_time),
        "days_since_last_order": compute_days_since_last_order(past_orders, current_time),
        "returns_last_30d": compute_returns_in_window(past_orders, current_time, 30),
        "returns_last_90d": compute_returns_in_window(past_orders, current_time, 90),
        "order_value_vs_customer_avg": compute_order_value_ratio(order_value, past_orders),
    }

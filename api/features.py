"""
RiskGuard — Inference-time feature computation.

Thin wrapper around features_core.py (the single shared implementation used
by both this API and data/generate_data.py) that applies the API's rounding
precision. See features_core.py for the actual formulas.
"""

import os
import sys
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from features_core import compute_all_temporal_features  # noqa: E402


def compute_features(customer: dict, past_orders: list, order_timestamp: datetime, order_value: float) -> dict:
    """
    customer: dict with 'account_created_date' (datetime) key.
    past_orders: list of dicts (oldest first) with 'order_timestamp' (datetime),
                 'order_value' (float), 'returned' (0/1/None) keys.
    Returns the customer-level feature dict used by the model.
    """
    feats = compute_all_temporal_features(
        past_orders, order_timestamp, customer["account_created_date"], order_value
    )
    return {
        "bayesian_return_rate": round(feats["bayesian_return_rate"], 6),
        "customer_purchase_frequency": round(feats["customer_purchase_frequency"], 4),
        "account_age_days": feats["account_age_days"],
        "days_since_last_order": feats["days_since_last_order"],
        "returns_last_30d": feats["returns_last_30d"],
        "returns_last_90d": feats["returns_last_90d"],
        "order_value_vs_customer_avg": round(feats["order_value_vs_customer_avg"], 4),
    }

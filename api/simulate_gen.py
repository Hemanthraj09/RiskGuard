"""
RiskGuard — Simulation Console order generator.

Generates genuinely new orders (new or existing customers) to demonstrate
the model scoring unseen data, separate from both train and test sets.
`risk_shift` in [0, 1] skews category / payment-mode / delivery-tier mix
and order value toward higher-risk conditions, giving the live demo a
visibly shifting risk distribution as the slider moves — this is just a
parameter on the same population-level draw used elsewhere, no new
engineering complexity.

Outcome-visibility policy: in production, a return outcome isn't known
until days or weeks after purchase. api/main.py's /simulate handler scores
every order in a batch against a snapshot of each customer's history taken
BEFORE the batch started, and defers all database writes until the whole
batch is scored -- so two orders generated in the same /simulate call for
the same customer never see each other's existence or outcome, even though
this module may hand out the same customer_id twice in one call. Only
future /simulate or /score calls see these orders as real history. This
generator itself is stateless and doesn't need to know about the policy;
it's enforced entirely by the caller.
"""

import os
import sys

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

if DATA_DIR not in sys.path:
    sys.path.insert(0, DATA_DIR)

from generate_data import CATEGORIES, PAYMENT_MODES, PINCODE_TIERS  # noqa: E402

# Baseline mix ~= the population-level draw used in data/generate_data.py's
# customer-profile generation (roughly uniform categories, payment Dirichlet
# centered at [.30,.25,.35,.10], pincode tiers [.40,.35,.25]).
BASELINE_CATEGORY_WEIGHTS = np.array([1.0 / len(CATEGORIES)] * len(CATEGORIES))
SHIFTED_CATEGORY_WEIGHTS = np.array([0.34, 0.30, 0.06, 0.06, 0.10, 0.14])  # footwear/apparel heavy

BASELINE_PAYMENT_WEIGHTS = np.array([0.30, 0.25, 0.35, 0.10])  # COD, prepaid, UPI, wallet
SHIFTED_PAYMENT_WEIGHTS = np.array([0.75, 0.08, 0.12, 0.05])   # COD heavy

BASELINE_TIER_WEIGHTS = np.array([0.40, 0.35, 0.25])  # metro, tier2, tier3
SHIFTED_TIER_WEIGHTS = np.array([0.10, 0.30, 0.60])   # tier3 heavy

EXISTING_CUSTOMER_PROB = 0.30


def _interp(baseline: np.ndarray, shifted: np.ndarray, shift: float) -> np.ndarray:
    w = (1 - shift) * baseline + shift * shifted
    return w / w.sum()


def sample_order_fields(risk_shift: float, rng: np.random.RandomState) -> dict:
    shift = float(np.clip(risk_shift, 0.0, 1.0))

    category = str(rng.choice(CATEGORIES, p=_interp(BASELINE_CATEGORY_WEIGHTS, SHIFTED_CATEGORY_WEIGHTS, shift)))
    payment_mode = str(rng.choice(PAYMENT_MODES, p=_interp(BASELINE_PAYMENT_WEIGHTS, SHIFTED_PAYMENT_WEIGHTS, shift)))
    pincode_tier = str(rng.choice(PINCODE_TIERS, p=_interp(BASELINE_TIER_WEIGHTS, SHIFTED_TIER_WEIGHTS, shift)))

    # Order value: log-normal, center drifts from median ~Rs.1,800 toward the
    # Rs.3,000 "impulse zone" (highest non-monotonic risk) as shift increases.
    base_log_mean = np.log(1800)
    shifted_log_mean = np.log(3000)
    log_mean = (1 - shift) * base_log_mean + shift * shifted_log_mean
    order_value = float(np.clip(rng.lognormal(mean=log_mean, sigma=0.4), 200, 15000))

    discount = round(float(rng.uniform(5, 30)), 1) if rng.random() < 0.30 else 0.0

    return {
        "order_value": round(order_value, 2),
        "product_category": category,
        "payment_mode": payment_mode,
        "discount_applied": discount,
        "delivery_pincode_tier": pincode_tier,
    }


def make_new_customer_id(seq: int) -> str:
    return f"SIM{seq:06d}"

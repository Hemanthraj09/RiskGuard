"""
RiskGuard — Synthetic Data Generator

Generates ~12,000 e-commerce orders across ~2,500 customers with:
- Realistic multi-order customer histories
- Temporal features computed strictly from past orders (no leakage)
- Labels from the locked function in Section 5.2 of the spec
- 6% label noise to cap achievable AUC at realistic levels
- Temporal train/test split (80/20 by timestamp)

All random processes use fixed seeds for reproducibility (Section 5.4).
"""

import numpy as np
import random
import pandas as pd
from datetime import datetime, timedelta
from collections import defaultdict
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # riskguard/
from features_core import compute_all_temporal_features  # noqa: E402

# ─────────────────────────────────────────────────────────────
# Fixed Seeds (Section 5.4)
# ─────────────────────────────────────────────────────────────
np.random.seed(42)
random.seed(42)

# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────
SIM_START = datetime(2024, 1, 1)
SIM_END = datetime(2024, 6, 30)
SIM_DAYS = (SIM_END - SIM_START).days  # 181 days

N_CUSTOMERS = 2500
LABEL_NOISE_RATE = 0.06

CATEGORIES = [
    "footwear", "apparel", "electronics_accessories",
    "groceries", "home_goods", "beauty",
]
PAYMENT_MODES = ["COD", "prepaid_card", "UPI", "wallet"]
PINCODE_TIERS = ["metro", "tier2", "tier3"]

# Bayesian prior for customer return rate
PRIOR_ALPHA = 2
PRIOR_BETA = 8  # Prior mean = 2 / (2 + 8) = 0.20


# ─────────────────────────────────────────────────────────────
# Locked Label Function (Section 5.2 — retuned, see below)
# ─────────────────────────────────────────────────────────────
# NOTE ON THIS RETUNE: the original constants (footwear 0.22, COD 1.55x,
# etc.) produced an oracle AUC ceiling of only ~0.60-0.63 (verified by
# scoring the true generative probability against the sampled label) --
# well below the "believable ~0.75-0.92" sanity check this spec itself
# calls for. A systematic search over the same functional form (still
# multi-signal, non-monotonic, noisy -- no single dominant rule) found
# that closing the gap purely by widening spread pushes the *overall*
# return rate to 27-31%, unrealistic for a catalog mostly made of
# groceries/electronics. These constants ("SWEET-3") instead land on a
# balance: real trained-model AUC ~0.72, overall return rate ~22%,
# footwear/apparel ~35-42% (COD/fashion returns run high in Indian
# e-commerce), groceries/electronics ~7-12%. Approved by the user after
# reviewing the tradeoff -- do not modify again without new sign-off.
CATEGORY_BASE = {
    "footwear": 0.32,
    "apparel": 0.25,
    "electronics_accessories": 0.022,
    "groceries": 0.007,
    "home_goods": 0.048,
    "beauty": 0.088,
}

PAYMENT_MULTIPLIER = {
    "COD": 2.3,
    "prepaid_card": 1.0,
    "UPI": 0.83,
    "wallet": 0.68,
}


def non_monotonic_value_factor(order_value: float) -> float:
    """
    Peaked curve over order value.  Returns multiplier in [0.55, 1.55].
    Peak at ₹3,000 (impulse / mid-range zone).
    Very cheap and very expensive orders get lower multipliers.
    """
    peak = 3000
    spread = 1500
    factor = 1.55 * np.exp(-0.5 * ((order_value - peak) / spread) ** 2)
    return max(0.55, min(1.55, factor))


def generate_return_probability(row: dict) -> float:
    """
    Locked label-generation function from Section 5.2 (retuned, see note above).
    Combines category base rate, payment multiplier, customer history,
    non-monotonic value effect, and pincode × category interaction.
    """
    base = CATEGORY_BASE[row["product_category"]]
    prob = base * PAYMENT_MULTIPLIER[row["payment_mode"]]

    # Blend with Bayesian-smoothed customer history (22% weight)
    bayes_rate = row["bayesian_return_rate"]
    prob = 0.78 * prob + 0.22 * bayes_rate

    # Non-monotonic order value effect
    prob *= non_monotonic_value_factor(row["order_value"])

    # Pincode × category modifier
    if row["delivery_pincode_tier"] == "tier3" and row["product_category"] in [
        "apparel",
        "footwear",
    ]:
        prob *= 1.55
    elif row["delivery_pincode_tier"] == "metro":
        prob *= 0.78

    # Clip to valid probability
    prob = float(np.clip(prob, 0.01, 0.90))
    return prob


# ─────────────────────────────────────────────────────────────
# Customer Profile Generation
# ─────────────────────────────────────────────────────────────
def generate_customer_profiles(n_customers: int) -> list:
    """
    Create customer pool with latent purchasing preferences.
    These preferences control WHAT orders customers generate
    (categories, value, payment modes), NOT the label function
    (which is locked and unchanged).
    """
    profiles = []

    for i in range(n_customers):
        cid = f"C{i + 1:05d}"

        # Account created 0–540 days (~18 months) before simulation start
        days_before = int(np.random.randint(0, 540))
        account_created = SIM_START - timedelta(days=days_before)

        # Fixed delivery location (pincode tier — customer lives somewhere)
        pincode_tier = np.random.choice(PINCODE_TIERS, p=[0.40, 0.35, 0.25])

        # Category preferences: Dirichlet with one dominant category
        alpha = np.ones(len(CATEGORIES)) * 0.5
        dominant_idx = np.random.randint(0, len(CATEGORIES))
        alpha[dominant_idx] = 3.0
        category_prefs = np.random.dirichlet(alpha)

        # Personal order-value center (log-normal, median ~₹1,800)
        personal_log_mean = np.random.normal(np.log(1800), 0.3)

        # Payment mode preferences (Dirichlet around global distribution)
        payment_alpha = np.array([0.30, 0.25, 0.35, 0.10]) * 5.0
        payment_prefs = np.random.dirichlet(payment_alpha)

        # Number of orders in the 6-month window (gamma → Poisson)
        expected_orders = max(1.0, np.random.gamma(shape=2.0, scale=2.4))
        n_orders = max(1, int(np.random.poisson(expected_orders)))

        profiles.append(
            {
                "customer_id": cid,
                "account_created_date": account_created,
                "pincode_tier": pincode_tier,
                "category_prefs": category_prefs,
                "personal_log_mean": personal_log_mean,
                "payment_prefs": payment_prefs,
                "n_orders": n_orders,
            }
        )

    return profiles


# ─────────────────────────────────────────────────────────────
# Order Skeleton Generation
# ─────────────────────────────────────────────────────────────
def generate_order_skeletons(profiles: list) -> list:
    """
    Generate order metadata (no labels, no temporal features yet).
    Each order gets a timestamp, category, payment mode, value, etc.
    drawn from the customer's preferences.
    """
    orders = []
    order_counter = 0

    for p in profiles:
        cid = p["customer_id"]
        n = p["n_orders"]

        # Distribute timestamps uniformly across the simulation window
        raw_days = sorted(np.random.uniform(0, SIM_DAYS, size=n))
        timestamps = [SIM_START + timedelta(days=float(d)) for d in raw_days]

        for ts in timestamps:
            order_counter += 1

            # Category from customer preferences
            category = np.random.choice(CATEGORIES, p=p["category_prefs"])

            # Payment mode from customer preferences
            payment = np.random.choice(PAYMENT_MODES, p=p["payment_prefs"])

            # Order value: log-normal centered at customer's personal mean
            order_value = float(
                np.random.lognormal(mean=p["personal_log_mean"], sigma=0.4)
            )
            order_value = round(float(np.clip(order_value, 200, 15000)), 2)

            # Discount: 70% none, 30% get 5–30%
            if np.random.random() < 0.30:
                discount = round(float(np.random.uniform(5, 30)), 1)
            else:
                discount = 0.0

            orders.append(
                {
                    "order_id": f"ORD{order_counter:06d}",
                    "customer_id": cid,
                    "order_value": order_value,
                    "product_category": category,
                    "payment_mode": payment,
                    "discount_applied": discount,
                    "order_timestamp": ts,
                    "delivery_pincode_tier": p["pincode_tier"],
                    "_account_created_date": p["account_created_date"],
                }
            )

    return orders


# ─────────────────────────────────────────────────────────────
# Shared feature computation (see features_core.py)
# ─────────────────────────────────────────────────────────────
def compute_customer_features_for_order(past: list, T: datetime, acct_created: datetime, order_value: float) -> dict:
    """
    Same shape/semantics as api.features.compute_features -- this is the
    "generator's usage path" exercised alongside the API's in
    tests/test_feature_parity.py. All the actual formulas live in
    features_core.py; this just applies the generator's rounding precision.
    """
    feats = compute_all_temporal_features(past, T, acct_created, order_value, PRIOR_ALPHA, PRIOR_BETA)
    return {
        "bayesian_return_rate": round(feats["bayesian_return_rate"], 6),
        "customer_purchase_frequency": round(feats["customer_purchase_frequency"], 4),
        "account_age_days": feats["account_age_days"],
        "days_since_last_order": feats["days_since_last_order"],
        "returns_last_30d": feats["returns_last_30d"],
        "returns_last_90d": feats["returns_last_90d"],
        "order_value_vs_customer_avg": round(feats["order_value_vs_customer_avg"], 4),
    }


# ─────────────────────────────────────────────────────────────
# Chronological Feature Computation & Labeling
# ─────────────────────────────────────────────────────────────
def compute_temporal_features_and_labels(orders: list) -> list:
    """
    Process orders in strict chronological order.
    For each order, compute temporal features from ONLY that
    customer's prior orders (timestamp < current), then generate
    the label and record it into history.

    This structure makes temporal leakage structurally impossible:
    customer_history[cid] is only appended to AFTER the current
    order is fully labeled.
    """
    # Sort globally by timestamp, tiebreak by order_id for determinism
    orders.sort(key=lambda o: (o["order_timestamp"], o["order_id"]))

    # customer_id → list of past order dicts (already labeled)
    customer_history = defaultdict(list)

    for idx, order in enumerate(orders):
        cid = order["customer_id"]
        T = order["order_timestamp"]
        acct_created = order["_account_created_date"]
        past = customer_history[cid]  # All have timestamp < T by construction

        # ── Temporal customer-level features (shared with api/features.py
        #    via features_core.py — one implementation, not two) ──
        feats = compute_customer_features_for_order(past, T, acct_created, order["order_value"])
        order.update(feats)

        # ── Interaction features (no temporal dependency) ──
        order["category_x_payment_mode"] = (
            f"{order['product_category']}_{order['payment_mode']}"
        )
        order["pincode_tier_x_category"] = (
            f"{order['delivery_pincode_tier']}_{order['product_category']}"
        )

        # ── Generate label using locked function ──
        prob = generate_return_probability(order)
        order["return_probability"] = round(prob, 6)  # Keep for analysis only
        label = int(np.random.binomial(1, prob))

        # Apply 6% label noise
        if np.random.random() < LABEL_NOISE_RATE:
            label = 1 - label

        order["returned"] = label

        # Record into history AFTER labeling
        customer_history[cid].append(order)

        # Progress logging
        if (idx + 1) % 3000 == 0:
            print(f"      Processed {idx + 1:,} / {len(orders):,} orders...")

    return orders


# ─────────────────────────────────────────────────────────────
# Output Column Order
# ─────────────────────────────────────────────────────────────
OUTPUT_COLUMNS = [
    # Identifiers
    "order_id",
    "customer_id",
    "order_timestamp",
    # Order-level features
    "order_value",
    "product_category",
    "payment_mode",
    "discount_applied",
    "delivery_pincode_tier",
    # Customer-level features (temporal, leakage-free)
    "bayesian_return_rate",
    "customer_purchase_frequency",
    "account_age_days",
    "days_since_last_order",
    # Temporal aggregates
    "returns_last_30d",
    "returns_last_90d",
    # Engineered / interaction features
    "category_x_payment_mode",
    "order_value_vs_customer_avg",
    "pincode_tier_x_category",
    # Ground-truth probability (for analysis, NOT a model feature)
    "return_probability",
    # Label
    "returned",
]


# ─────────────────────────────────────────────────────────────
# Main Pipeline
# ─────────────────────────────────────────────────────────────
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "processed")

    print("=" * 60)
    print("RiskGuard — Synthetic Data Generator")
    print("=" * 60)

    # 1. Generate customer profiles
    print("\n[1/5] Generating customer profiles...")
    profiles = generate_customer_profiles(N_CUSTOMERS)
    total_planned = sum(p["n_orders"] for p in profiles)
    print(f"      {len(profiles):,} customers, ~{total_planned:,} planned orders")

    # 2. Generate order skeletons
    print("[2/5] Generating order skeletons...")
    orders = generate_order_skeletons(profiles)
    print(f"      {len(orders):,} order skeletons created")

    # 3. Chronological processing
    print("[3/5] Computing temporal features + labels (chronological)...")
    orders = compute_temporal_features_and_labels(orders)
    print(f"      {len(orders):,} orders processed")

    # 4. Build DataFrame
    df = pd.DataFrame(orders)
    df = df[OUTPUT_COLUMNS]
    df["order_timestamp"] = df["order_timestamp"].apply(
        lambda x: x.strftime("%Y-%m-%d %H:%M:%S")
    )
    df = df.sort_values("order_timestamp").reset_index(drop=True)

    # 5. Temporal 3-way split (65% train / 15% validation / 20% test)
    #
    # A 2-way split tempts you into selecting the cost-optimal threshold by
    # sweeping it against the same test set you then report precision/recall
    # on -- letting the test set influence a modeling decision before it
    # evaluates that decision. The validation slice exists so threshold
    # selection (model/evaluate.py) never touches test. The 65/80 cut points
    # keep the test set's absolute boundary identical to the old 80/20 split
    # (same 2,437 held-out orders throughout this project's history).
    print("[4/5] Temporal 3-way split (65% train / 15% validation / 20% test)...")
    train_end = int(len(df) * 0.65)
    val_end = int(len(df) * 0.80)
    train_df = df.iloc[:train_end].reset_index(drop=True)
    val_df = df.iloc[train_end:val_end].reset_index(drop=True)
    test_df = df.iloc[val_end:].reset_index(drop=True)

    # 6. Save
    print("[5/5] Saving CSVs...")
    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "train.csv")
    val_path = os.path.join(output_dir, "validation.csv")
    test_path = os.path.join(output_dir, "test.csv")
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    # ── Summary Statistics ──
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    print(f"Total orders:        {len(df):,}")
    print(f"Unique customers:    {df['customer_id'].nunique():,}")
    print(f"Train set:           {len(train_df):,} ({len(train_df) / len(df) * 100:.1f}%)")
    print(f"Validation set:      {len(val_df):,} ({len(val_df) / len(df) * 100:.1f}%)")
    print(f"Test set:            {len(test_df):,} ({len(test_df) / len(df) * 100:.1f}%)")
    print(f"Overall return rate: {df['returned'].mean() * 100:.2f}%")
    print(f"Train return rate:   {train_df['returned'].mean() * 100:.2f}%")
    print(f"Validation return rate: {val_df['returned'].mean() * 100:.2f}%")
    print(f"Test return rate:    {test_df['returned'].mean() * 100:.2f}%")

    print(f"\nReturn rates by category:")
    for cat in CATEGORIES:
        cat_df = df[df["product_category"] == cat]
        rate = cat_df["returned"].mean() * 100 if len(cat_df) > 0 else 0
        print(f"  {cat:30s}  {rate:5.2f}%  (n={len(cat_df):,})")

    print(f"\nReturn rates by payment mode:")
    for pm in PAYMENT_MODES:
        pm_df = df[df["payment_mode"] == pm]
        rate = pm_df["returned"].mean() * 100 if len(pm_df) > 0 else 0
        print(f"  {pm:15s}  {rate:5.2f}%  (n={len(pm_df):,})")

    print(f"\nOrders per customer distribution:")
    opc = df.groupby("customer_id").size()
    print(
        f"  min={opc.min()}, median={opc.median():.0f}, "
        f"mean={opc.mean():.1f}, max={opc.max()}"
    )

    print(f"\nOrder value distribution:")
    print(
        f"  min=Rs.{df['order_value'].min():,.0f}, "
        f"median=Rs.{df['order_value'].median():,.0f}, "
        f"mean=Rs.{df['order_value'].mean():,.0f}, "
        f"max=Rs.{df['order_value'].max():,.0f}"
    )

    print(f"\nFiles saved:")
    print(f"  {train_path}")
    print(f"  {val_path}")
    print(f"  {test_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

"""
RiskGuard — Scoring core.

Loads the trained model + isotonic calibrator + SHAP explainer once at
import time, and exposes score_order() used by both POST /score (a single
order) and POST /simulate (looped over a generated batch). Keeping this in
one place guarantees /score and /simulate can never drift out of sync.
"""

import os
import sys
import json
import pickle

import pandas as pd
import shap

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
MODEL_DIR = os.path.join(PROJECT_ROOT, "model")
ARTIFACTS_DIR = os.path.join(MODEL_DIR, "artifacts")

if MODEL_DIR not in sys.path:
    sys.path.insert(0, MODEL_DIR)

from train import prepare_features  # noqa: E402 (reuse the exact training-time encoder)

with open(os.path.join(ARTIFACTS_DIR, "model.pkl"), "rb") as f:
    _model = pickle.load(f)
with open(os.path.join(ARTIFACTS_DIR, "calibrator.pkl"), "rb") as f:
    _calibrator = pickle.load(f)
with open(os.path.join(ARTIFACTS_DIR, "metadata.json")) as f:
    _metadata = json.load(f)
with open(os.path.join(ARTIFACTS_DIR, "eval_results.json")) as f:
    _eval_results = json.load(f)

_encoded_columns = _metadata["encoded_columns"]
_categorical_columns = _metadata["categorical_columns"]
_explainer = shap.TreeExplainer(_model)

OPTIMAL_THRESHOLD = _eval_results["threshold_selection"]["optimal_threshold"]
FRICTION_COST = _eval_results["threshold_selection"]["friction_cost"]
RETURN_COST = _eval_results["threshold_selection"]["return_cost"]
REVIEW_COST = _eval_results["threshold_selection"]["review_cost"]
# Low/Medium split scales with the actual cost-optimal threshold rather than
# a hardcoded constant, since that threshold moves whenever the label
# generator or cost assumptions change.
LOW_CUTOFF = round(OPTIMAL_THRESHOLD * 0.5, 4)

_FEATURE_LABELS = {
    "order_value": "Order value",
    "discount_applied": "Discount applied",
    "bayesian_return_rate": "Customer return history",
    "customer_purchase_frequency": "Purchase frequency",
    "account_age_days": "Account age",
    "days_since_last_order": "Days since last order",
    "returns_last_30d": "Returns in last 30 days",
    "returns_last_90d": "Returns in last 90 days",
    "order_value_vs_customer_avg": "Order value vs. customer average",
}


def _humanize_feature_name(encoded_name: str, is_active: bool) -> str:
    if encoded_name in _FEATURE_LABELS:
        return _FEATURE_LABELS[encoded_name]
    for prefix in sorted(_categorical_columns, key=len, reverse=True):
        if encoded_name.startswith(prefix + "_"):
            value = encoded_name[len(prefix) + 1:].replace("_", " ")
            label = _FEATURE_LABELS.get(prefix, prefix.replace("_", " ").title())
            # A one-hot dummy that's 0 for this row still carries a SHAP
            # contribution (tree models use "not this category" as signal
            # too) -- phrase it as an absence, not a false-looking match.
            return f"{label}: {value}" if is_active else f"{label}: not {value}"
    return encoded_name.replace("_", " ").title()


def _build_raw_row(order_fields: dict, customer_features: dict) -> pd.DataFrame:
    row = {
        "order_value": order_fields["order_value"],
        "product_category": order_fields["product_category"],
        "payment_mode": order_fields["payment_mode"],
        "discount_applied": order_fields.get("discount_applied", 0.0),
        "delivery_pincode_tier": order_fields["delivery_pincode_tier"],
        "bayesian_return_rate": customer_features["bayesian_return_rate"],
        "customer_purchase_frequency": customer_features["customer_purchase_frequency"],
        "account_age_days": customer_features["account_age_days"],
        "days_since_last_order": customer_features["days_since_last_order"],
        "returns_last_30d": customer_features["returns_last_30d"],
        "returns_last_90d": customer_features["returns_last_90d"],
        "category_x_payment_mode": f"{order_fields['product_category']}_{order_fields['payment_mode']}",
        "order_value_vs_customer_avg": customer_features["order_value_vs_customer_avg"],
        "pincode_tier_x_category": f"{order_fields['delivery_pincode_tier']}_{order_fields['product_category']}",
    }
    return pd.DataFrame([row])


def risk_band(probability: float) -> str:
    if probability < LOW_CUTOFF:
        return "low"
    if probability < OPTIMAL_THRESHOLD:
        return "medium"
    return "high"


_SIZING_SENSITIVE_CATEGORIES = {"footwear", "apparel"}


def recommend_action(band: str, payment_mode: str, product_category: str) -> str:
    """
    A thin, transparent, RULE-BASED recommendation layer on top of the ML
    risk score -- not another model, and not an autonomous action. This is
    the "responder" half of "detector, verifier, or auto-responder" from the
    track brief: it upgrades the score into a specific, auditable suggestion
    a human can act on, but it never executes anything itself (no auto-block,
    no auto-refund, no auto-cancel -- defense-only, human always in the loop).
    """
    if band == "high":
        if payment_mode == "COD" and product_category in _SIZING_SENSITIVE_CATEGORIES:
            return "Recommend prepaid payment or a size/fit verification call before shipping."
        if payment_mode != "COD":
            return "Flag for manual review before dispatch."
        return "Flag for a verification call before shipping."
    if band == "medium":
        return "Optional: confirm via an automated message before shipping."
    return "Process normally."


def score_order(order_fields: dict, customer_features: dict) -> dict:
    raw_df = _build_raw_row(order_fields, customer_features)
    X, _ = prepare_features(raw_df, encoded_columns=_encoded_columns)

    raw_prob = float(_model.predict_proba(X)[:, 1][0])
    calibrated_prob = float(_calibrator.predict([raw_prob])[0])

    shap_values = _explainer.shap_values(X)
    if isinstance(shap_values, list):
        sv = shap_values[1][0]
    else:
        sv = shap_values[0]
        if getattr(sv, "ndim", 1) == 2:
            sv = sv[:, 1] if sv.shape[1] > 1 else sv[:, 0]

    contributions = sorted(
        zip(X.columns.tolist(), [float(v) for v in sv]),
        key=lambda x: abs(x[1]),
        reverse=True,
    )
    row = X.iloc[0]
    top_contributors = [
        {
            "feature": _humanize_feature_name(name, bool(row[name])),
            "shap_value": round(val, 4),
            "direction": "increases_risk" if val > 0 else "decreases_risk",
        }
        for name, val in contributions[:5]
    ]

    band = risk_band(calibrated_prob)
    recommendation = "flag_for_verification" if calibrated_prob >= OPTIMAL_THRESHOLD else "accept_normally"

    return {
        "probability": round(calibrated_prob, 4),
        "risk_band": band,
        "recommendation": recommendation,
        "recommended_action": recommend_action(band, order_fields["payment_mode"], order_fields["product_category"]),
        "optimal_threshold": OPTIMAL_THRESHOLD,
        "top_contributors": top_contributors,
    }

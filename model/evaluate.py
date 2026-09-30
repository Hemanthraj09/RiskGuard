"""
RiskGuard — Model Evaluation

Implements a leakage-safe 3-way evaluation protocol:
  - The cost-optimal decision threshold is selected by sweeping thresholds
    against VALIDATION set predictions only, and frozen there.
  - That frozen threshold is then applied to the TEST set, which has never
    influenced any modeling decision (not training, not calibration, not
    threshold selection) -- only final reporting.
  - Threshold-independent metrics (ROC-AUC, PR curve, calibration curve,
    Brier score, ECE, lift curve) are computed directly on test, since they
    don't involve a selection step that could leak.

Produces every artifact needed by the Model Performance dashboard:
  - Precision, Recall, F1, Confusion Matrix (at the frozen threshold, on test)
  - ROC-AUC (+ bootstrap 95% CI) and ROC curve data
  - Precision-recall curve
  - Calibration curve, Brier score, Expected Calibration Error (ECE)
  - Cost-based threshold sweep on validation (selection) AND on test
    (reporting what that frozen threshold actually does on held-out data)
  - Lift / gains curve
  - One honest failure case

Saves everything to artifacts/eval_results.json.

Usage:
    python model/evaluate.py
"""

import os
import sys
import json
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    brier_score_loss,
)
from sklearn.calibration import calibration_curve

warnings.filterwarnings("ignore", category=UserWarning)

# ─────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
ARTIFACTS_DIR = os.path.join(SCRIPT_DIR, "artifacts")

TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
VAL_PATH = os.path.join(DATA_DIR, "validation.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")

if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

DATA_SCRIPT_DIR = os.path.join(PROJECT_ROOT, "data")
API_SCRIPT_DIR = os.path.join(PROJECT_ROOT, "api")
if DATA_SCRIPT_DIR not in sys.path:
    sys.path.insert(0, DATA_SCRIPT_DIR)
if API_SCRIPT_DIR not in sys.path:
    sys.path.insert(0, API_SCRIPT_DIR)

from train import prepare_features  # noqa: E402 (reuse the exact training-time encoder)
from generate_data import generate_return_probability, PRIOR_ALPHA, PRIOR_BETA, LABEL_NOISE_RATE  # noqa: E402
from simulate_gen import sample_order_fields  # noqa: E402 (for the risk-shifted calibration probe)

# ─────────────────────────────────────────────────────────────
# Fixed Cost Assumptions (Section 6.2)
# ─────────────────────────────────────────────────────────────
DEFAULT_FRICTION_COST = 180   # Rs. per false positive
DEFAULT_RETURN_COST = 650     # Rs. per false negative
DEFAULT_REVIEW_COST = 50      # Rs. per flagged order (analyst review time), Tier-2 addition

N_BOOTSTRAP = 1000
RNG_SEED = 42


def load_artifacts():
    with open(os.path.join(ARTIFACTS_DIR, "model.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(ARTIFACTS_DIR, "calibrator.pkl"), "rb") as f:
        calibrator = pickle.load(f)
    with open(os.path.join(ARTIFACTS_DIR, "metadata.json"), "r") as f:
        metadata = json.load(f)
    return model, calibrator, metadata


def prepare_split(path, metadata):
    df = pd.read_csv(path)
    y = df[metadata["label_column"]].values
    X, _ = prepare_features(df, encoded_columns=metadata["encoded_columns"])
    return X, y, df


def get_calibrated_probs(model, calibrator, X):
    raw = model.predict_proba(X)[:, 1]
    return calibrator.predict(raw)


def compute_cost_curve(y_true, y_prob, n_thresholds=200):
    """
    Sweep thresholds and return per-threshold confusion-matrix counts plus
    precision/recall. Total cost is NOT baked in here -- the caller (or the
    frontend, via its adjustable sliders) combines fp/fn/flag_count with
    whatever cost assumptions are current. Keeping raw counts instead of a
    single cost number is what lets the friction/return/review-cost sliders
    recompute live without another round trip.
    """
    thresholds = np.linspace(0.01, 0.99, n_thresholds)
    results = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        results.append({
            "threshold": round(float(t), 4),
            "fp": int(fp), "fn": int(fn), "tp": int(tp), "tn": int(tn),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
        })
    return results


def total_cost(entry, friction_cost, return_cost, review_cost=0):
    return entry["fp"] * friction_cost + entry["fn"] * return_cost + (entry["fp"] + entry["tp"]) * review_cost


def select_optimal_threshold(cost_curve, friction_cost, return_cost, review_cost=0):
    scored = [(e, total_cost(e, friction_cost, return_cost, review_cost)) for e in cost_curve]
    best_entry, best_cost = min(scored, key=lambda pair: pair[1])
    return best_entry, best_cost


def compute_ece(y_true, y_prob, n_bins=10):
    """Expected Calibration Error: bin predictions into n_bins equal-width
    bins, weight each bin's |predicted - actual| gap by its share of samples."""
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(y_prob, bin_edges[1:-1], right=True), 0, n_bins - 1)
    n = len(y_true)
    ece = 0.0
    per_bin = []
    for b in range(n_bins):
        mask = bin_idx == b
        count = int(mask.sum())
        if count == 0:
            continue
        mean_pred = float(y_prob[mask].mean())
        actual_rate = float(y_true[mask].mean())
        weight = count / n
        ece += weight * abs(mean_pred - actual_rate)
        per_bin.append({
            "bin_range": [round(float(bin_edges[b]), 2), round(float(bin_edges[b + 1]), 2)],
            "count": count,
            "mean_predicted": round(mean_pred, 4),
            "actual_rate": round(actual_rate, 4),
        })
    return round(ece, 4), per_bin


def downsample_curve(xs, ys, max_points=100):
    """Every k-th point of a curve for plotting, always keeping the last one
    -- a plain [::k] slice silently drops the endpoint (ROC's (1, 1))."""
    step = max(1, len(xs) // max_points)
    idx = list(range(0, len(xs), step))
    if idx[-1] != len(xs) - 1:
        idx.append(len(xs) - 1)
    return [round(float(xs[i]), 4) for i in idx], [round(float(ys[i]), 4) for i in idx]


def compute_lift_curve(y_true, y_prob, n_deciles=10):
    """Sort by predicted risk descending; at each decile of orders reviewed,
    what fraction of ALL actual returns in the test set have been caught."""
    order = np.argsort(-y_prob)
    y_sorted = y_true[order]
    total_positives = y_sorted.sum()
    n = len(y_sorted)
    deciles, capture_rates, random_baseline = [], [], []
    for i in range(1, n_deciles + 1):
        frac = i / n_deciles
        cutoff = int(round(n * frac))
        captured = y_sorted[:cutoff].sum()
        deciles.append(round(frac * 100, 1))
        capture_rates.append(round(float(captured / total_positives), 4) if total_positives > 0 else 0.0)
        random_baseline.append(round(frac, 4))
    return {"decile_pct": deciles, "capture_rate": capture_rates, "random_baseline": random_baseline}


def bootstrap_ci(y_true, y_prob, threshold, n_bootstrap=N_BOOTSTRAP, seed=RNG_SEED):
    """Bootstrap-resample (with replacement, same size as the test set) to
    get 95% CIs on AUC and on precision/recall/F1 at the frozen threshold.
    Meaningful mainly because the test set here is only ~2.4k rows -- point
    estimates on a set this size carry real sampling uncertainty."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    aucs, precisions, recalls, f1s = [], [], [], []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        yt, yp = y_true[idx], y_prob[idx]
        if len(np.unique(yt)) < 2:
            continue  # AUC undefined for a resample with only one class
        aucs.append(roc_auc_score(yt, yp))
        pred = (yp >= threshold).astype(int)
        precisions.append(precision_score(yt, pred, zero_division=0))
        recalls.append(recall_score(yt, pred, zero_division=0))
        f1s.append(f1_score(yt, pred, zero_division=0))

    def ci(values):
        lo, hi = np.percentile(values, [2.5, 97.5])
        return [round(float(lo), 4), round(float(hi), 4)]

    return {
        "roc_auc_ci": ci(aucs),
        "precision_ci": ci(precisions),
        "recall_ci": ci(recalls),
        "f1_ci": ci(f1s),
        "n_bootstrap": len(aucs),
    }


HEURISTIC_HIGH_RISK_CATEGORIES = ["footwear", "apparel"]  # the two highest category base rates


def heuristic_baseline(df, friction_cost, return_cost, review_cost):
    """
    A hand-written rule using the two strongest known signals: flag if
    payment is COD AND the category is one of the two highest-base-rate
    categories. This is a fixed decision rule, not a ranked score -- it has
    exactly one implied operating point, so we report precision/recall/cost
    at that point rather than forcing an AUC out of a binary rule.
    """
    flagged = (df["payment_mode"] == "COD") & (df["product_category"].isin(HEURISTIC_HIGH_RISK_CATEGORIES))
    y_pred = flagged.astype(int).values
    y_true = df["returned"].values
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    cost = fp * friction_cost + fn * return_cost + (fp + tp) * review_cost
    return {
        "rule": f"Flag if payment_mode == COD and product_category in {HEURISTIC_HIGH_RISK_CATEGORIES}",
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "total_cost": float(cost),
        "flag_rate": round(float(y_pred.mean()), 4),
    }


def logistic_regression_baseline(X_train, y_train, X_test, y_test):
    """Plain logistic regression on the identical one-hot feature set and
    the same train/test split -- no calibration, no threshold tuning. A
    linear-model floor to compare LightGBM's ranking quality against."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    lr = LogisticRegression(max_iter=2000, random_state=RNG_SEED)
    lr.fit(X_train_scaled, y_train)
    test_prob = lr.predict_proba(X_test_scaled)[:, 1]
    return {
        "model": "LogisticRegression(max_iter=2000) on StandardScaler-ed one-hot features, same train/test split as LightGBM",
        "test_auc": round(float(roc_auc_score(y_test, test_prob)), 4),
    }


def compute_segment_metrics(y_true, y_prob, threshold, segment_series, min_sample=100):
    """AUC/precision/recall broken out by a categorical cut, on the SAME
    test set and SAME frozen threshold as the headline numbers. A segment
    below min_sample is still reported (never silently dropped) but flagged
    insufficient_sample so a thin slice isn't shown with the same visual
    confidence as the global metric."""
    segments = {}
    for value in sorted(segment_series.dropna().unique().tolist(), key=str):
        mask = (segment_series == value).values
        n = int(mask.sum())
        if n == 0:
            continue
        yt, yp = y_true[mask], y_prob[mask]
        y_pred = (yp >= threshold).astype(int)
        auc = round(float(roc_auc_score(yt, yp)), 4) if len(np.unique(yt)) > 1 else None
        segments[str(value)] = {
            "n": n,
            "insufficient_sample": n < min_sample,
            "positive_rate": round(float(yt.mean()), 4),
            # With zero flags, precision/recall read 0.0 -- not "every flag was wrong".
            "n_flagged": int(y_pred.sum()),
            "auc": auc,
            "precision": round(float(precision_score(yt, y_pred, zero_division=0)), 4),
            "recall": round(float(recall_score(yt, y_pred, zero_division=0)), 4),
        }
    return segments


def compute_headline_savings(y_true, cm_frozen, friction_cost, return_cost, review_cost):
    """Two boundary cases -- flag nothing, flag everything -- against the
    model's actual cost at the chosen threshold, normalized per 1,000
    orders. This is the number a hiring panel remembers, not an AUC decimal."""
    tn, fp = cm_frozen[0]
    fn, tp = cm_frozen[1]
    n = len(y_true)
    n_positive = int(y_true.sum())
    n_negative = n - n_positive

    model_cost = fp * friction_cost + fn * return_cost + (fp + tp) * review_cost
    flag_nothing_cost = n_positive * return_cost
    flag_everything_cost = n_negative * friction_cost + n * review_cost

    scale = 1000.0 / n
    return {
        "n_orders": n,
        "model_cost_per_1000": round(model_cost * scale, 2),
        "flag_nothing_cost_per_1000": round(flag_nothing_cost * scale, 2),
        "flag_everything_cost_per_1000": round(flag_everything_cost * scale, 2),
        "savings_vs_flag_nothing_per_1000": round((flag_nothing_cost - model_cost) * scale, 2),
        "savings_vs_flag_everything_per_1000": round((flag_everything_cost - model_cost) * scale, 2),
    }


def bootstrap_threshold_stability(y_val, val_prob, friction_cost, return_cost, review_cost,
                                   n_bootstrap=N_BOOTSTRAP, seed=RNG_SEED, n_thresholds=200):
    """
    Re-run cost-optimal threshold SELECTION on n_bootstrap resamples of the
    validation set (with replacement), vectorized across all thresholds at
    once per resample for speed. A stable threshold across resamples is a
    quiet but real credibility signal; a threshold that swings wildly would
    mean "the optimal threshold" is mostly noise in this validation slice.
    """
    rng = np.random.RandomState(seed)
    n = len(y_val)
    thresholds = np.linspace(0.01, 0.99, n_thresholds)
    selected = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        yt, yp = y_val[idx], val_prob[idx]
        pred = yp[:, None] >= thresholds[None, :]  # (n, n_thresholds)
        pos = (yt == 1)[:, None]
        fp = (pred & ~pos).sum(axis=0)
        fn = (~pred & pos).sum(axis=0)
        tp = (pred & pos).sum(axis=0)
        cost = fp * friction_cost + fn * return_cost + (fp + tp) * review_cost
        selected.append(float(thresholds[np.argmin(cost)]))
    selected = np.array(selected)
    q25, median, q75 = np.percentile(selected, [25, 50, 75])
    return {
        "n_bootstrap": len(selected),
        "median_threshold": round(float(median), 4),
        "iqr": [round(float(q25), 4), round(float(q75), 4)],
        "min": round(float(selected.min()), 4),
        "max": round(float(selected.max()), 4),
    }


def probe_shifted_calibration(model, calibrator, metadata, risk_shift, n_orders=1500, seed=123):
    """
    Diagnostic-only, not wired into the live API: generates a batch of
    cold-start synthetic orders using the SAME risk_shift sampling the
    Simulation Console's slider drives, applies the LOCKED label-generation
    formula to get genuine ground truth for this probe, scores them through
    the trained model+calibrator, and checks whether calibration (ECE)
    holds up under the shifted population the live demo will actually show.
    Cold-start-only customers isolate exactly what risk_shift changes --
    order-level field distribution -- since risk_shift never touches
    customer history in api/simulate_gen.py. That also means the test set's
    ECE is NOT the baseline to compare against (different population,
    different sample size); a risk_shift=0.0 probe is -- see
    summarize_shift_probes.
    """
    field_rng = np.random.RandomState(seed)
    label_rng = np.random.RandomState(seed + 1)
    cold_start_bayes_rate = PRIOR_ALPHA / (PRIOR_ALPHA + PRIOR_BETA)

    rows = []
    for _ in range(n_orders):
        fields = sample_order_fields(risk_shift, field_rng)
        row = {
            "order_value": fields["order_value"],
            "product_category": fields["product_category"],
            "payment_mode": fields["payment_mode"],
            "discount_applied": fields["discount_applied"],
            "delivery_pincode_tier": fields["delivery_pincode_tier"],
            "bayesian_return_rate": cold_start_bayes_rate,
            "customer_purchase_frequency": 0.0,
            "account_age_days": 0,
            "days_since_last_order": -1,
            "returns_last_30d": 0,
            "returns_last_90d": 0,
            "order_value_vs_customer_avg": 1.0,
        }
        prob = generate_return_probability(row)
        label = int(label_rng.binomial(1, prob))
        if label_rng.random() < LABEL_NOISE_RATE:
            label = 1 - label
        row["returned"] = label
        row["category_x_payment_mode"] = f"{row['product_category']}_{row['payment_mode']}"
        row["pincode_tier_x_category"] = f"{row['delivery_pincode_tier']}_{row['product_category']}"
        rows.append(row)

    shifted_df = pd.DataFrame(rows)
    X_shifted, _ = prepare_features(shifted_df, encoded_columns=metadata["encoded_columns"])
    shifted_prob = get_calibrated_probs(model, calibrator, X_shifted)
    y_shifted = shifted_df["returned"].values

    ece_shifted, _ = compute_ece(y_shifted, shifted_prob, n_bins=10)
    auc_shifted = round(float(roc_auc_score(y_shifted, shifted_prob)), 4) if len(np.unique(y_shifted)) > 1 else None
    return {
        "risk_shift": risk_shift,
        "n_orders": n_orders,
        "positive_rate": round(float(y_shifted.mean()), 4),
        "ece": ece_shifted,
        "auc": auc_shifted,
    }


PROBE_SHIFTS = (0.0, 0.7, 1.0)  # 0.0 is the like-for-like baseline for the other two
PROBE_SEEDS = (123, 456, 789, 1011, 1213)
PROBE_N_ORDERS = 1500


def summarize_shift_probes(model, calibrator, metadata, risk_shift):
    """Median over several independent probes at one risk_shift: a single
    1,500-order probe's ECE swings by roughly +/-0.01 from seed to seed --
    about the size of the effect being measured -- so one draw can't tell
    a real calibration shift from sampling noise."""
    runs = [
        probe_shifted_calibration(model, calibrator, metadata, risk_shift, n_orders=PROBE_N_ORDERS, seed=seed)
        for seed in PROBE_SEEDS
    ]
    eces = [r["ece"] for r in runs]
    return {
        "risk_shift": risk_shift,
        "n_orders": PROBE_N_ORDERS,
        "n_probes": len(runs),
        "ece_median": round(float(np.median(eces)), 4),
        "ece_range": [min(eces), max(eces)],
        "auc_median": round(float(np.median([r["auc"] for r in runs])), 4),
        "positive_rate_median": round(float(np.median([r["positive_rate"] for r in runs])), 4),
    }


def bayes_optimal_at_flag_count(y_true, true_prob, n_flagged):
    """The ceiling at the model's own operating point: flag the same number
    of orders, ranked by the label generator's true return probability.
    That catches the most returns any model can expect to with that many
    flags, so it bounds the precision AND recall achievable at this flag
    rate -- the right yardstick for "are these precision/recall numbers
    low?", since both are capped by the data's noise, not just the model."""
    flagged = np.argsort(-true_prob, kind="stable")[:n_flagged]
    y_pred = np.zeros_like(y_true)
    y_pred[flagged] = 1
    return {
        "n_flagged": int(n_flagged),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
    }


def find_honest_failure(df, y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    wrong_mask = y_pred != y_true
    if not wrong_mask.any():
        return None

    confidence = np.abs(y_prob - threshold)
    wrong_confidence = confidence.copy()
    wrong_confidence[~wrong_mask] = -1
    worst_idx = np.argmax(wrong_confidence)

    row = df.iloc[worst_idx]
    is_false_negative = y_pred[worst_idx] == 0 and y_true[worst_idx] == 1

    if is_false_negative:
        explanation = (
            f"This {row['product_category']} order (Rs.{row['order_value']:,.0f}, "
            f"{row['payment_mode']}) was predicted low-risk "
            f"(probability {y_prob[worst_idx]:.1%}) but was actually returned. "
            f"Likely cause: an unobservable factor such as sizing dissatisfaction, "
            f"product defect, or buyer's remorse that our feature set cannot capture. "
            f"The customer's prior return rate ({row['bayesian_return_rate']:.1%}) "
            f"didn't flag elevated risk."
        )
        failure_type = "false_negative"
    else:
        explanation = (
            f"This {row['product_category']} order (Rs.{row['order_value']:,.0f}, "
            f"{row['payment_mode']}) was predicted high-risk "
            f"(probability {y_prob[worst_idx]:.1%}) but was NOT returned. "
            f"Likely cause: the risk signals (category, payment mode, customer history) "
            f"aligned toward high risk, but this particular buyer was satisfied with "
            f"the product despite the statistical profile suggesting otherwise."
        )
        failure_type = "false_positive"

    return {
        "order_id": row["order_id"],
        "customer_id": row["customer_id"],
        "product_category": row["product_category"],
        "order_value": float(row["order_value"]),
        "payment_mode": row["payment_mode"],
        "delivery_pincode_tier": row["delivery_pincode_tier"],
        "bayesian_return_rate": float(row["bayesian_return_rate"]),
        "predicted_probability": round(float(y_prob[worst_idx]), 4),
        "predicted_label": int(y_pred[worst_idx]),
        "actual_label": int(y_true[worst_idx]),
        "failure_type": failure_type,
        "explanation": explanation,
    }


def main():
    print("=" * 60)
    print("RiskGuard — Model Evaluation")
    print("=" * 60)

    print("\n[1/14] Loading model, calibrator, metadata...")
    model, calibrator, metadata = load_artifacts()

    print("[2/14] Loading train + validation + test data...")
    X_train, y_train, df_train = prepare_split(TRAIN_PATH, metadata)
    X_val, y_val, df_val = prepare_split(VAL_PATH, metadata)
    X_test, y_test, df_test = prepare_split(TEST_PATH, metadata)
    print(f"      Train:      {len(X_train):,} samples, positive rate {y_train.mean() * 100:.2f}%")
    print(f"      Validation: {len(X_val):,} samples, positive rate {y_val.mean() * 100:.2f}%")
    print(f"      Test:       {len(X_test):,} samples, positive rate {y_test.mean() * 100:.2f}%")

    print("[3/14] Generating calibrated predictions...")
    val_prob = get_calibrated_probs(model, calibrator, X_val)
    test_prob = get_calibrated_probs(model, calibrator, X_test)
    roc_auc = roc_auc_score(y_test, test_prob)
    print(f"      Test ROC-AUC: {roc_auc:.4f}")

    # PR-AUC (average precision): the area under the precision-recall curve,
    # summarized as one number the same way ROC-AUC summarizes the ROC curve.
    # More informative than ROC-AUC as a single ranking-quality number on an
    # imbalanced problem like this one (22% positive rate) -- ROC-AUC can look
    # deceptively strong when negatives dominate, since a low false-positive
    # RATE is easy to achieve by construction. The PR curve itself was already
    # computed and shown on the dashboard; this just gives it a headline number.
    pr_auc = average_precision_score(y_test, test_prob)
    print(f"      Test PR-AUC (average precision): {pr_auc:.4f}")

    # ── Reconcile the Bayes-optimal ceiling to one number, on this exact
    # test split. `return_probability` is the true generative probability
    # saved by data/generate_data.py (not a model feature); scoring it
    # against the sampled label is the AUC ceiling no classifier can beat. ──
    ceiling_auc = round(float(roc_auc_score(y_test, df_test["return_probability"])), 4)
    print(f"      Bayes-optimal ceiling AUC (test): {ceiling_auc:.4f}")
    # Share of the achievable ranking signal the model captures. Measured
    # from 0.5, not from 0: a coin flip already scores 0.5 AUC, so the raw
    # ratio roc_auc / ceiling_auc would credit a random classifier with
    # 0.5 / ceiling_auc of the ceiling -- well over half.
    ceiling_signal_captured = round(float((roc_auc - 0.5) / (ceiling_auc - 0.5)), 4)
    print(f"      Share of achievable signal captured: {ceiling_signal_captured:.1%}")

    # ── Calibration sanity check: is_unbalance=True during training skews
    # raw probabilities well above the true base rate. Confirm isotonic
    # calibration (fit on validation) actually corrects this on test, an
    # independent split neither the base model nor the calibrator saw. ──
    raw_test_prob = model.predict_proba(X_test)[:, 1]
    calibration_sanity_check = {
        "mean_raw_probability": round(float(raw_test_prob.mean()), 4),
        "mean_calibrated_probability": round(float(test_prob.mean()), 4),
        "actual_positive_rate": round(float(y_test.mean()), 4),
        "raw_gap": round(float(abs(raw_test_prob.mean() - y_test.mean())), 4),
        "calibrated_gap": round(float(abs(test_prob.mean() - y_test.mean())), 4),
    }
    print(f"      Mean raw probability:        {calibration_sanity_check['mean_raw_probability']:.4f} "
          f"(gap {calibration_sanity_check['raw_gap']:.4f} from actual rate)")
    print(f"      Mean calibrated probability: {calibration_sanity_check['mean_calibrated_probability']:.4f} "
          f"(gap {calibration_sanity_check['calibrated_gap']:.4f} from actual rate)")

    # ── Threshold selection on VALIDATION only (leakage-safe) ──
    print("[4/14] Selecting cost-optimal threshold on VALIDATION set (test never touched)...")
    validation_cost_curve = compute_cost_curve(y_val, val_prob)
    optimal_entry, optimal_val_cost = select_optimal_threshold(
        validation_cost_curve, DEFAULT_FRICTION_COST, DEFAULT_RETURN_COST, DEFAULT_REVIEW_COST
    )
    frozen_threshold = optimal_entry["threshold"]
    print(f"      Frozen threshold (selected on validation): {frozen_threshold:.4f}")
    print(f"      Validation cost at that threshold: Rs.{optimal_val_cost:,.0f}")

    # Same threshold grid on TEST, purely for reporting/plotting -- the
    # frontend looks up this curve at whatever threshold validation selects,
    # it never re-selects using test.
    test_cost_curve = compute_cost_curve(y_test, test_prob)
    test_entry_at_frozen = min(test_cost_curve, key=lambda e: abs(e["threshold"] - frozen_threshold))
    test_cost_at_frozen = total_cost(test_entry_at_frozen, DEFAULT_FRICTION_COST, DEFAULT_RETURN_COST, DEFAULT_REVIEW_COST)

    # ── Standard metrics at default 0.5 (fixed a priori, no selection -> test is fine) ──
    print("[5/14] Computing metrics at default threshold 0.5 (test)...")
    y_pred_05 = (test_prob >= 0.5).astype(int)
    metrics_at_05 = {
        "threshold": 0.5,
        "precision": round(float(precision_score(y_test, y_pred_05, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred_05, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, y_pred_05, zero_division=0)), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred_05, labels=[0, 1]).tolist(),
    }

    # ── Final metrics at the FROZEN threshold, on TEST ──
    print("[6/14] Computing final metrics at the frozen threshold (test)...")
    y_pred_frozen = (test_prob >= frozen_threshold).astype(int)
    precision_frozen = precision_score(y_test, y_pred_frozen, zero_division=0)
    recall_frozen = recall_score(y_test, y_pred_frozen, zero_division=0)
    f1_frozen = f1_score(y_test, y_pred_frozen, zero_division=0)
    cm_frozen = confusion_matrix(y_test, y_pred_frozen, labels=[0, 1])
    print(f"      Precision: {precision_frozen:.4f}  Recall: {recall_frozen:.4f}  F1: {f1_frozen:.4f}")
    ceiling_at_threshold = bayes_optimal_at_flag_count(
        y_test, df_test["return_probability"].values, int(y_pred_frozen.sum())
    )
    print(f"      Bayes-optimal flagging of the same {ceiling_at_threshold['n_flagged']} orders: "
          f"precision {ceiling_at_threshold['precision']:.4f}  recall {ceiling_at_threshold['recall']:.4f}")

    # ── Threshold-independent diagnostics, all on TEST ──
    print("[7/14] Computing calibration curve, Brier score, ECE, PR/ROC/lift curves...")
    prob_true, prob_pred = calibration_curve(y_test, test_prob, n_bins=10, strategy="uniform")
    fpr, tpr, _ = roc_curve(y_test, test_prob)
    pr_precision, pr_recall, _ = precision_recall_curve(y_test, test_prob)
    brier = brier_score_loss(y_test, test_prob)
    ece, ece_bins = compute_ece(y_test, test_prob, n_bins=10)
    lift = compute_lift_curve(y_test, test_prob, n_deciles=10)
    print(f"      Brier score: {brier:.4f}   ECE: {ece:.4f}")

    # ── Bootstrap 95% CIs on test ──
    print("[8/14] Bootstrapping 95% confidence intervals (test, n=1000)...")
    ci = bootstrap_ci(y_test, test_prob, frozen_threshold)
    print(f"      AUC 95% CI: {ci['roc_auc_ci']}")

    # ── Honest failure case, at the frozen threshold, on test ──
    print("[9/14] Finding honest failure case...")
    failure_case = find_honest_failure(df_test, y_test, test_prob, frozen_threshold)

    # ── Baselines: a floor (heuristic rule + logistic regression) to go with
    # the ceiling above -- the pitch needs both ends, not just the model. ──
    print("[10/14] Computing baselines (heuristic rule + logistic regression)...")
    heuristic = heuristic_baseline(df_test, DEFAULT_FRICTION_COST, DEFAULT_RETURN_COST, DEFAULT_REVIEW_COST)
    lr_baseline = logistic_regression_baseline(X_train, y_train, X_test, y_test)
    print(f"       Heuristic cost: Rs.{heuristic['total_cost']:,.0f}  (model: Rs.{test_cost_at_frozen:,.0f})")
    print(f"       Logistic regression test AUC: {lr_baseline['test_auc']:.4f}  (LightGBM: {roc_auc:.4f})")

    # ── Segment-level metrics on test, at the frozen threshold ──
    print("[11/14] Computing segment-level metrics (category / payment mode / tenure)...")
    tenure_series = pd.Series(
        np.where(df_test["days_since_last_order"].values == -1, "new", "returning"), index=df_test.index
    )
    segments = {
        "product_category": compute_segment_metrics(y_test, test_prob, frozen_threshold, df_test["product_category"]),
        "payment_mode": compute_segment_metrics(y_test, test_prob, frozen_threshold, df_test["payment_mode"]),
        "customer_tenure": compute_segment_metrics(y_test, test_prob, frozen_threshold, tenure_series),
    }

    # ── Headline Rs. number: model cost vs. the two boundary cases ──
    print("[12/14] Computing headline savings vs. flag-nothing / flag-everything...")
    headline_savings = compute_headline_savings(
        y_test, cm_frozen.tolist(), DEFAULT_FRICTION_COST, DEFAULT_RETURN_COST, DEFAULT_REVIEW_COST
    )
    print(f"       Savings vs flag-nothing:    Rs.{headline_savings['savings_vs_flag_nothing_per_1000']:,.0f} per 1,000 orders")
    print(f"       Savings vs flag-everything: Rs.{headline_savings['savings_vs_flag_everything_per_1000']:,.0f} per 1,000 orders")

    # ── Threshold stability across validation resamples ──
    print("[13/14] Bootstrapping threshold stability on validation (n=1000)...")
    threshold_stability = bootstrap_threshold_stability(
        y_val, val_prob, DEFAULT_FRICTION_COST, DEFAULT_RETURN_COST, DEFAULT_REVIEW_COST
    )
    print(f"       Threshold median {threshold_stability['median_threshold']:.4f}, "
          f"IQR {threshold_stability['iqr']}, range [{threshold_stability['min']:.4f}, {threshold_stability['max']:.4f}]")

    # ── Probe: does calibration hold up under the Simulation Console's
    # risk-shift slider (a shifted population, not the standard test set)? ──
    print(f"[14/14] Probe: ECE under risk-shifted synthetic batches "
          f"(cold-start customers, {len(PROBE_SEEDS)} probes per shift)...")
    shifted_probes = [summarize_shift_probes(model, calibrator, metadata, shift) for shift in PROBE_SHIFTS]
    for p in shifted_probes:
        print(f"       shift={p['risk_shift']}: median ECE={p['ece_median']:.4f} (range {p['ece_range']})  "
              f"median AUC={p['auc_median']}  positive_rate={p['positive_rate_median']:.4f}")

    eval_results = {
        "test_set_size": int(len(y_test)),
        "validation_set_size": int(len(y_val)),
        "positive_rate": round(float(y_test.mean()), 4),
        "validation_positive_rate": round(float(y_val.mean()), 4),

        "roc_auc": round(float(roc_auc), 4),
        "pr_auc": round(float(pr_auc), 4),
        "bayes_optimal_ceiling_auc": ceiling_auc,
        "ceiling_signal_captured": ceiling_signal_captured,
        "calibration_sanity_check": calibration_sanity_check,
        "brier_score": round(float(brier), 4),
        "ece": ece,
        "ece_bins": ece_bins,

        "metrics_at_05": metrics_at_05,

        "headline_savings": headline_savings,

        "baselines": {
            "heuristic": heuristic,
            "logistic_regression": lr_baseline,
            "lightgbm_test_auc": round(float(roc_auc), 4),
            "lightgbm_test_cost_at_frozen_threshold": float(test_cost_at_frozen),
        },

        "segments": segments,

        "threshold_selection": {
            "method": (
                "Cost-optimal threshold is selected by sweeping thresholds against the "
                "VALIDATION set only, then frozen and applied to the test set below. "
                "The test set never influences threshold selection."
            ),
            "friction_cost": DEFAULT_FRICTION_COST,
            "return_cost": DEFAULT_RETURN_COST,
            "review_cost": DEFAULT_REVIEW_COST,
            "optimal_threshold": frozen_threshold,
            "optimal_validation_cost": float(optimal_val_cost),
            "validation_cost_curve": validation_cost_curve,
            "threshold_stability": threshold_stability,
        },

        "test_metrics": {
            "threshold": frozen_threshold,
            "note": "Threshold was frozen from validation-set selection above; these are its results on held-out test data it never influenced.",
            "precision": round(float(precision_frozen), 4),
            "recall": round(float(recall_frozen), 4),
            "f1": round(float(f1_frozen), 4),
            "confusion_matrix": cm_frozen.tolist(),
            "total_cost_at_default_assumptions": float(test_cost_at_frozen),
            "test_cost_curve": test_cost_curve,
            **ci,
        },

        "bayes_optimal_at_threshold": ceiling_at_threshold,

        "calibration_curve": {
            "predicted_probability": [round(float(p), 4) for p in prob_pred],
            "actual_return_rate": [round(float(p), 4) for p in prob_true],
            "n_bins": 10,
        },

        "roc_curve": dict(zip(("fpr", "tpr"), downsample_curve(fpr, tpr))),

        "pr_curve": dict(zip(("precision", "recall"), downsample_curve(pr_precision, pr_recall))),

        "lift_curve": lift,

        "shifted_calibration_probes": shifted_probes,

        "failure_case": failure_case,
    }

    results_path = os.path.join(ARTIFACTS_DIR, "eval_results.json")
    with open(results_path, "w") as f:
        json.dump(eval_results, f, indent=2)

    print(f"\n{'=' * 60}")
    print("EVALUATION SUMMARY")
    print(f"{'=' * 60}")
    print(f"Bayes-optimal ceiling AUC: {ceiling_auc:.4f}  (model captures {ceiling_signal_captured:.1%} of the achievable signal)")
    print(f"Mean calibrated probability vs actual rate: "
          f"{calibration_sanity_check['mean_calibrated_probability']:.4f} vs "
          f"{calibration_sanity_check['actual_positive_rate']:.4f} "
          f"(raw was {calibration_sanity_check['mean_raw_probability']:.4f})")
    print(f"ROC-AUC (test):            {roc_auc:.4f}  95% CI {ci['roc_auc_ci']}")
    print(f"PR-AUC (test):             {pr_auc:.4f}")
    print(f"  vs. logistic regression: {lr_baseline['test_auc']:.4f}")
    print(f"Brier score (test):        {brier:.4f}")
    print(f"ECE (test):                {ece:.4f}")
    print(f"Frozen threshold (from validation): {frozen_threshold:.4f}  "
          f"(bootstrap median {threshold_stability['median_threshold']:.4f}, IQR {threshold_stability['iqr']})")
    print(f"Precision @ frozen (test): {precision_frozen:.4f}  95% CI {ci['precision_ci']}")
    print(f"Recall @ frozen (test):    {recall_frozen:.4f}  95% CI {ci['recall_ci']}")
    print(f"  Bayes-optimal, same {ceiling_at_threshold['n_flagged']} flags: precision "
          f"{ceiling_at_threshold['precision']:.4f}, recall {ceiling_at_threshold['recall']:.4f}")
    print(f"F1 @ frozen (test):        {f1_frozen:.4f}  95% CI {ci['f1_ci']}")
    print(f"Confusion matrix (test):   {cm_frozen.tolist()}")
    print(f"Model cost/1000:           Rs.{headline_savings['model_cost_per_1000']:,.0f}  "
          f"(flag-nothing Rs.{headline_savings['flag_nothing_cost_per_1000']:,.0f}, "
          f"flag-everything Rs.{headline_savings['flag_everything_cost_per_1000']:,.0f})")
    print(f"Heuristic baseline cost:   Rs.{heuristic['total_cost']:,.0f}  (rule: {heuristic['rule']})")
    if failure_case:
        print(f"\nHonest failure case: {failure_case['order_id']} ({failure_case['failure_type']})")
    print(f"\nResults saved to: {results_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

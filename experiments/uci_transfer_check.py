"""
RiskGuard — Real-data transfer sanity check (optional, exploratory).

Everything in the main pipeline is evaluated on self-generated synthetic
data -- a fair critique is "you proved the model can recover the process
you wrote." This script is a rough, honestly-caveated check of whether the
same *kind* of feature set (order value, customer history, recency, an
interaction with a categorical dimension) carries real signal on a public
dataset with a genuine, non-synthetic cancellation signal: the UCI Online
Retail dataset (Dec 2010 - Dec 2011, UK-based online gift retailer).

This is NOT wired into the RiskGuard product (no new API endpoint, no new
dashboard page) -- it's a standalone, one-off analysis. Its result is
reported in the README as a directional signal, not a claim that this
number is comparable to the main model's held-out AUC.

Label construction (an approximation, stated plainly): the dataset has no
explicit "was this order returned" flag. It does have cancellation invoices
(InvoiceNo prefixed 'C', negative quantities). We treat a normal invoice as
a "returned-ish" order if that same customer has ANY cancellation invoice
within the following 30 days -- a proxy for "this purchase episode was
followed by a return/cancellation," not a verified per-item return match.

Customer history has to respect that label's latency: a past invoice's
label depends on cancellations up to 30 days AFTER it, so at time T it's
only known once that window has closed. customer_prior_cancel_rate counts
only those resolved invoices. (An earlier version counted every past
invoice, reading the future -- one cancellation at T+10d labels both a
past invoice at T-5d and this one -- and reported an inflated 0.78 test
AUC; that leaky variant is still computed below, clearly labeled, so the
size of the leak stays visible.)

Usage:
    python experiments/uci_transfer_check.py /path/to/Online Retail.xlsx
"""

import sys
import warnings

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

np.random.seed(42)

PRIOR_ALPHA = 1
PRIOR_BETA = 4
CANCELLATION_WINDOW_DAYS = 30


def load_raw(path: str) -> pd.DataFrame:
    df = pd.read_excel(path, engine="openpyxl")
    df = df.dropna(subset=["CustomerID"]).copy()
    df["CustomerID"] = df["CustomerID"].astype(int).astype(str)
    df["InvoiceNo"] = df["InvoiceNo"].astype(str)
    df["is_cancellation"] = df["InvoiceNo"].str.startswith("C")
    return df


def build_invoice_level(df: pd.DataFrame):
    normal = df[~df["is_cancellation"] & (df["Quantity"] > 0) & (df["UnitPrice"] > 0)].copy()
    normal["line_total"] = normal["Quantity"] * normal["UnitPrice"]

    invoices = (
        normal.groupby("InvoiceNo")
        .agg(
            customer_id=("CustomerID", "first"),
            country=("Country", "first"),
            invoice_date=("InvoiceDate", "min"),
            order_value=("line_total", "sum"),
            n_items=("StockCode", "nunique"),
            avg_unit_price=("UnitPrice", "mean"),
        )
        .reset_index()
    )
    invoices = invoices.sort_values("invoice_date").reset_index(drop=True)

    cancellations = (
        df[df["is_cancellation"]]
        .groupby("CustomerID")["InvoiceDate"]
        .apply(lambda s: sorted(s.tolist()))
        .to_dict()
    )
    return invoices, cancellations


def label_and_engineer(invoices: pd.DataFrame, cancellations: dict) -> pd.DataFrame:
    window = pd.Timedelta(days=CANCELLATION_WINDOW_DAYS)

    # Chronological per-customer history for the causal features below.
    invoices = invoices.sort_values(["customer_id", "invoice_date"]).reset_index(drop=True)

    bayes_rates, leaky_bayes_rates, freqs, days_since, labels = [], [], [], [], []
    history: dict = {}  # customer_id -> list of (date, was_labeled_return)

    for row in invoices.itertuples(index=False):
        cid = row.customer_id
        T = row.invoice_date
        past = history.get(cid, [])

        # Only past invoices whose 30-day label window has closed by T have a
        # known label at T (see module docstring).
        resolved = [r for (d, r) in past if d + window <= T]
        bayes_rates.append((sum(resolved) + PRIOR_ALPHA) / (len(resolved) + PRIOR_ALPHA + PRIOR_BETA))
        leaky_bayes_rates.append((sum(r for (_, r) in past) + PRIOR_ALPHA) / (len(past) + PRIOR_ALPHA + PRIOR_BETA))
        freqs.append(len(past))
        days_since.append((T - past[-1][0]).days if past else -1)

        # Label: any cancellation for this customer within the next 30 days.
        cust_cancellations = cancellations.get(cid, [])
        label = int(any(T < c_date <= T + window for c_date in cust_cancellations))
        labels.append(label)

        history.setdefault(cid, []).append((T, label))

    invoices = invoices.copy()
    invoices["customer_prior_cancel_rate"] = bayes_rates
    invoices["customer_prior_cancel_rate_leaky"] = leaky_bayes_rates  # diagnostic only
    invoices["customer_prior_order_count"] = freqs
    invoices["days_since_last_order"] = days_since
    invoices["returned"] = labels
    return invoices.sort_values("invoice_date").reset_index(drop=True)


def check_window_overlap(invoices: pd.DataFrame, window_days: int = 15) -> float:
    """
    Diagnostic: the 30-day-cancellation-proxy label can cluster -- if a
    customer cancels once, EVERY invoice of theirs in the preceding 30 days
    gets label=1, so nearby invoices don't get independent labels. Measures
    what fraction of positive-labeled invoices have another positive-labeled
    invoice from the SAME customer within `window_days` -- a high fraction
    means the model can partly learn "this customer is currently inside a
    cancellation episode" rather than per-order risk, which would inflate
    AUC without representing genuine order-level predictive signal.
    """
    pos = invoices[invoices["returned"] == 1].sort_values(["customer_id", "invoice_date"])
    total_pos = len(pos)
    if total_pos == 0:
        return 0.0
    clustered = 0
    for _, grp in pos.groupby("customer_id"):
        dates = grp["invoice_date"].tolist()
        for i, d in enumerate(dates):
            if any(d != d2 and abs((d - d2).days) <= window_days for d2 in dates):
                clustered += 1
    return clustered / total_pos


def fit_and_auc(invoices: pd.DataFrame, feature_cols: list, split_idx: int):
    X = pd.get_dummies(invoices[feature_cols + ["country_grp"]], columns=["country_grp"])
    y = invoices["returned"].values
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    model = lgb.LGBMClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.05, num_leaves=8,
        min_child_samples=30, subsample=0.7, colsample_bytree=0.7,
        reg_alpha=1.0, reg_lambda=1.0, random_state=42, verbose=-1, is_unbalance=True,
    )
    model.fit(X_train, y_train)
    train_auc = roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])
    test_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    importances = sorted(zip(X.columns.tolist(), model.feature_importances_.tolist()), key=lambda x: -x[1])
    return train_auc, test_auc, importances


def main(path: str):
    print("=" * 60)
    print("RiskGuard -- Real-data transfer sanity check (UCI Online Retail)")
    print("=" * 60)

    print("\n[1/4] Loading raw transactions...")
    df = load_raw(path)
    print(f"      {len(df):,} line items with a CustomerID")

    print("[2/4] Aggregating to invoice level + labeling (30-day cancellation proxy)...")
    invoices, cancellations = build_invoice_level(df)
    invoices = label_and_engineer(invoices, cancellations)
    print(f"      {len(invoices):,} invoices, {invoices['customer_id'].nunique():,} customers")
    print(f"      Positive rate (proxy label): {invoices['returned'].mean() * 100:.2f}%")

    # Keep the top N countries as categories, fold the rest into "other" --
    # this dataset is >90% UK, a long tail of countries would just be noise.
    top_countries = invoices["country"].value_counts().nlargest(8).index
    invoices["country_grp"] = invoices["country"].where(invoices["country"].isin(top_countries), "other")
    split_idx = int(len(invoices) * 0.80)

    print("[3/4] Label-window diagnostic: how clustered is the 30-day proxy label?")
    overlap_frac = check_window_overlap(invoices, window_days=15)
    print(
        f"      {overlap_frac * 100:.1f}% of positive-labeled invoices have another "
        f"positive-labeled invoice from the SAME customer within 15 days -- one "
        f"cancellation event labels a whole cluster of nearby orders, so a history "
        f"feature that can see still-open labels partly learns 'this customer is "
        f"mid-cancellation-episode' rather than per-order risk."
    )

    order_only_cols = ["order_value", "n_items", "avg_unit_price"]
    history_cols = ["customer_prior_order_count", "days_since_last_order"]
    full_cols = order_only_cols + ["customer_prior_cancel_rate"] + history_cols
    leaky_cols = order_only_cols + ["customer_prior_cancel_rate_leaky"] + history_cols

    print("[4/4] Training FULL (causal history) vs. ORDER-LEVEL-ONLY, plus the leaky variant as a diagnostic...")
    train_auc_full, test_auc_full, imp_full = fit_and_auc(invoices, full_cols, split_idx)
    train_auc_order, test_auc_order, _ = fit_and_auc(invoices, order_only_cols, split_idx)
    train_auc_leaky, test_auc_leaky, imp_leaky = fit_and_auc(invoices, leaky_cols, split_idx)

    print(f"\n{'=' * 60}")
    print("RESULT")
    print(f"{'=' * 60}")
    print(f"FULL (causal history)         -- train AUC {train_auc_full:.4f}  test AUC {test_auc_full:.4f}")
    print(f"  top feature: {imp_full[0][0]} (importance {imp_full[0][1]:.0f}, next highest {imp_full[1][1]:.0f})")
    print(f"ORDER-LEVEL-ONLY (no history) -- train AUC {train_auc_order:.4f}  test AUC {test_auc_order:.4f}")
    print(f"[diagnostic] LEAKY history    -- train AUC {train_auc_leaky:.4f}  test AUC {test_auc_leaky:.4f}")
    print(f"  top feature: {imp_leaky[0][0]} (importance {imp_leaky[0][1]:.0f}, next highest {imp_leaky[1][1]:.0f})")
    print(f"Same-customer label-window overlap: {overlap_frac * 100:.1f}%")

    print(
        f"\nFINDING: with customer history restricted to labels that are actually known "
        f"at order time, the full model scores {test_auc_full:.4f} test AUC vs. "
        f"{test_auc_order:.4f} for order-level features alone -- customer history adds "
        f"{test_auc_full - test_auc_order:+.4f}. The leaky variant's {test_auc_leaky:.4f} "
        f"is an artifact of reading still-open 30-day label windows ({overlap_frac * 100:.0f}% "
        f"of positives sit within 15 days of another positive from the same customer), "
        f"not transferable signal -- {test_auc_leaky - test_auc_full:+.4f} of it came "
        f"from the future.\n"
        f"\nThe number structurally comparable to the main synthetic model is the "
        f"ORDER-LEVEL-ONLY result: {test_auc_order:.4f} test AUC using just order value, "
        f"item count, unit price, and country -- meaningfully above 0.5 on genuine "
        f"real-world data, without leaning on this dataset's label-construction mechanics."
    )
    print(f"{'=' * 60}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python uci_transfer_check.py /path/to/Online Retail.xlsx")
        sys.exit(1)
    main(sys.argv[1])

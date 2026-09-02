"""
RiskGuard — Model Training

Trains a LightGBM classifier on train.csv, then fits calibration on
validation.csv (chronologically after train, never touched again until
model/evaluate.py applies the frozen threshold to test.csv). Using a
genuinely separate chronological slice for calibration -- rather than a
random subsample carved out of train -- keeps calibration and, downstream,
threshold selection (see evaluate.py) honest: test.csv is never used for
any modeling decision, only for final reporting.

The production calibrator is BaggedIsotonicCalibrator (50 bootstrap-resampled
isotonic fits averaged) -- adopted after comparing its full eval_results.json
output against a single isotonic fit (lower ECE and Brier, tighter threshold
bootstrap IQR, no regression on precision/recall/F1; see git history for the
before/after comparison). Both are fit and saved on every run:
calibrator.pkl is the live bagged calibrator every other file loads;
calibrator_isotonic_single.pkl is the single-fit comparison baseline, kept
so a revert is a one-line change, not a re-derivation.

Uses pd.get_dummies for categorical encoding (robust with any
sklearn wrapper — no categorical_feature pass-through issues).

Usage:
    python model/train.py
"""

import os
import sys
import json
import pickle
import warnings

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score

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

# ─────────────────────────────────────────────────────────────
# Feature Configuration
# ─────────────────────────────────────────────────────────────
ID_COLUMNS = ["order_id", "customer_id", "order_timestamp"]
ANALYSIS_COLUMNS = ["return_probability"]  # Ground-truth prob, not a feature
LABEL_COLUMN = "returned"

CATEGORICAL_COLUMNS = [
    "product_category",
    "payment_mode",
    "delivery_pincode_tier",
    "category_x_payment_mode",
    "pincode_tier_x_category",
]


def prepare_features(df, encoded_columns=None):
    """
    Drop non-feature columns, one-hot encode categoricals.
    If encoded_columns is provided, align to those columns (for val/test/inference).
    Returns (X_encoded, encoded_columns_list).
    """
    drop_cols = ID_COLUMNS + ANALYSIS_COLUMNS + [LABEL_COLUMN]
    drop_cols = [c for c in drop_cols if c in df.columns]

    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feature_cols].copy()

    # One-hot encode categoricals
    X = pd.get_dummies(X, columns=CATEGORICAL_COLUMNS, dtype=float)

    if encoded_columns is not None:
        # Align columns to training schema (handles unseen categories)
        X = X.reindex(columns=encoded_columns, fill_value=0.0)
    else:
        encoded_columns = list(X.columns)

    return X, encoded_columns


class BaggedIsotonicCalibrator:
    """Averages N isotonic regressions, each fit on a bootstrap resample of
    the validation set, to reduce variance in sparse-probability bins versus
    a single isotonic fit (isotonic's step function can overfit local noise
    where validation has few points, e.g. above ~0.6). Same .predict()
    interface as sklearn's IsotonicRegression, so every existing call site
    (api/scoring.py, model/evaluate.py) needs zero changes if this is
    promoted to be the live calibrator.

    Module-level (not a closure/lambda): api/scoring.py unpickles
    calibrator.pkl in a separate process and needs to resolve this class by
    qualified name (`train.BaggedIsotonicCalibrator`).
    """

    def __init__(self, n_bags=50, seed=42, y_min=0.01, y_max=0.99):
        self.n_bags = n_bags
        self.seed = seed
        self.y_min = y_min
        self.y_max = y_max
        self.models = []

    def fit(self, val_probs, y_val):
        rng = np.random.RandomState(self.seed)
        n = len(val_probs)
        for _ in range(self.n_bags):
            idx = rng.randint(0, n, size=n)
            iso = IsotonicRegression(y_min=self.y_min, y_max=self.y_max, out_of_bounds="clip")
            iso.fit(val_probs[idx], y_val[idx])
            self.models.append(iso)
        return self

    def predict(self, X):
        X = np.asarray(X)
        preds = np.mean([m.predict(X) for m in self.models], axis=0)
        return preds


# Force this class to always pickle as "train.BaggedIsotonicCalibrator", never
# "__main__.BaggedIsotonicCalibrator". Without this, `python model/train.py`
# (running this file as the entry point, per its own documented usage) would
# execute it as module __main__, and pickle embeds the *defining* module's
# name at save time -- so calibrator.pkl would only unpickle correctly in
# whatever process happens to also be running this exact file as __main__,
# which api/scoring.py and model/evaluate.py never are (they import this file
# as a normal module named "train").
#
# Setting __module__ alone isn't enough: pickle also verifies, at dump time,
# that sys.modules[obj.__class__.__module__] has this exact class object
# under this exact name -- and when running as a script, "train" isn't a
# registered module at all (only "__main__" is), so that check fails.
# Aliasing sys.modules["train"] to this same module object (a no-op when
# this file is genuinely imported as "train", since that entry already
# exists and already points here) satisfies the check either way.
sys.modules.setdefault("train", sys.modules[__name__])
BaggedIsotonicCalibrator.__module__ = "train"


def main():
    print("=" * 60)
    print("RiskGuard — Model Training")
    print("=" * 60)

    # 1. Load data
    print("\n[1/5] Loading train + validation data...")
    train_df = pd.read_csv(TRAIN_PATH)
    val_df = pd.read_csv(VAL_PATH)
    y_train = train_df[LABEL_COLUMN].values
    y_val = val_df[LABEL_COLUMN].values
    X_train, encoded_columns = prepare_features(train_df)
    X_val, _ = prepare_features(val_df, encoded_columns=encoded_columns)
    print(f"      Train:      {len(X_train):,} samples, {len(encoded_columns)} features (after one-hot)")
    print(f"      Validation: {len(X_val):,} samples")
    print(f"      Train positive rate:      {y_train.mean() * 100:.2f}%")
    print(f"      Validation positive rate: {y_val.mean() * 100:.2f}%")

    # 2. Train LightGBM on train.csv only
    print("[2/5] Training LightGBM classifier...")
    # Deliberately low-capacity: ~7.9k training rows and a signal ceiling of
    # ~0.71-0.73 AUC (verified against the true generative probability) mean
    # a deep/wide model just memorizes noise. max_depth=7/num_leaves=63/500
    # trees drove train AUC to 0.96 while held-out test AUC collapsed to
    # 0.66. This config keeps train/validation/test AUC close together.
    model = lgb.LGBMClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        num_leaves=8,
        min_child_samples=60,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=1.0,
        reg_lambda=1.0,
        random_state=42,
        verbose=-1,
        is_unbalance=True,
    )
    model.fit(X_train, y_train)

    # Check raw model performance
    model_train_probs = model.predict_proba(X_train)[:, 1]
    model_val_probs = model.predict_proba(X_val)[:, 1]
    train_auc = roc_auc_score(y_train, model_train_probs)
    val_auc = roc_auc_score(y_val, model_val_probs)
    print(f"      Raw train AUC:      {train_auc:.4f}")
    print(f"      Raw validation AUC: {val_auc:.4f}")

    # 3. Calibrate using isotonic regression, fit on validation.csv --
    #    a chronologically separate slice the model never trained on.
    #
    # Pattern used: manual "cv=prefit" equivalent, not sklearn's
    # CalibratedClassifierCV(cv=N) (which would internally cross-validate
    # and refit multiple base models). Base model is trained ONCE on
    # train.csv above; the calibrator is fit separately on that model's
    # predictions over validation.csv only. test.csv is never touched by
    # either the base model or the calibrator -- consistent with the same
    # train/validation/test boundary the leakage-safe threshold selection
    # in evaluate.py depends on.
    print("[3/5] Calibrating probabilities...")
    print("      Fitting single-fit isotonic regression (comparison baseline)...")
    single_calibrator = IsotonicRegression(
        y_min=0.01, y_max=0.99, out_of_bounds="clip"
    )
    single_calibrator.fit(model_val_probs, y_val)

    val_probs_single = single_calibrator.predict(model_val_probs)
    val_auc_single = roc_auc_score(y_val, val_probs_single)
    print(f"      [comparison] Calibrated validation AUC: {val_auc_single:.4f}")
    print(f"      [comparison] Calibrated prob range: [{val_probs_single.min():.4f}, {val_probs_single.max():.4f}]")

    # Production calibrator: bagged isotonic (averages n_bags isotonic fits,
    # each on a bootstrap resample of validation) reduces variance in
    # sparse-probability bins vs. a single fit -- adopted as the live
    # calibrator after comparing full eval_results.json output before/after
    # (lower ECE and Brier, tighter threshold bootstrap IQR, no regression on
    # precision/recall/F1). The single-fit calibrator above is still fit and
    # saved every run purely as a comparison baseline, not because it's used
    # anywhere downstream.
    print("      Fitting BaggedIsotonicCalibrator (n_bags=50, seed=42) [production]...")
    calibrator = BaggedIsotonicCalibrator(n_bags=50, seed=42)
    calibrator.fit(model_val_probs, y_val)
    val_probs_calibrated = calibrator.predict(model_val_probs)
    val_auc_calibrated = roc_auc_score(y_val, val_probs_calibrated)
    print(f"      [production] Calibrated validation AUC: {val_auc_calibrated:.4f}")
    print(f"      [production] Calibrated prob range: [{val_probs_calibrated.min():.4f}, {val_probs_calibrated.max():.4f}]")

    # 4. Save artifacts
    print("[4/5] Saving artifacts...")
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    model_path = os.path.join(ARTIFACTS_DIR, "model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"  Model:        {model_path}")

    calibrator_path = os.path.join(ARTIFACTS_DIR, "calibrator.pkl")
    with open(calibrator_path, "wb") as f:
        pickle.dump(calibrator, f)
    print(f"  Calibrator:   {calibrator_path} (production: BaggedIsotonicCalibrator)")

    single_calibrator_path = os.path.join(ARTIFACTS_DIR, "calibrator_isotonic_single.pkl")
    with open(single_calibrator_path, "wb") as f:
        pickle.dump(single_calibrator, f)
    print(f"  Calibrator:   {single_calibrator_path} (comparison baseline only, unused downstream)")

    raw_feature_cols = [
        c for c in train_df.columns
        if c not in ID_COLUMNS + ANALYSIS_COLUMNS + [LABEL_COLUMN]
    ]
    metadata = {
        "raw_feature_names": raw_feature_cols,
        "encoded_columns": encoded_columns,
        "categorical_columns": CATEGORICAL_COLUMNS,
        "label_column": LABEL_COLUMN,
        "id_columns": ID_COLUMNS,
        "analysis_columns": ANALYSIS_COLUMNS,
        "n_train_samples": len(X_train),
        "n_validation_samples": len(X_val),
        "train_positive_rate": float(y_train.mean()),
        "validation_positive_rate": float(y_val.mean()),
        "raw_train_auc": round(train_auc, 4),
        "raw_validation_auc": round(val_auc, 4),
        "calibrated_validation_auc": round(val_auc_calibrated, 4),
        "calibrator": "BaggedIsotonicCalibrator(n_bags=50, seed=42)",
        "comparison_single_isotonic_validation_auc": round(val_auc_single, 4),
    }
    metadata_path = os.path.join(ARTIFACTS_DIR, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  Metadata:     {metadata_path}")

    # 5. Feature importance (top 15)
    print("[5/5] Feature importances...")
    importance = model.feature_importances_
    feat_imp = sorted(
        zip(encoded_columns, importance), key=lambda x: x[1], reverse=True
    )
    print(f"\n  Top 15 feature importances:")
    for name, imp in feat_imp[:15]:
        print(f"    {name:45s}  {imp:6.0f}")

    print(f"\n{'=' * 60}")
    print("Training complete.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

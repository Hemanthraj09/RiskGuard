"""
Verifies the SHAP-vs-calibration relationship the dashboard's explanation
UI depends on getting right (Tier 1, item 1 of the verification pass).

TreeExplainer explains the raw LightGBM booster's margin output; isotonic
calibration is a separate, non-linear wrapper the explainer has no
visibility into. Two things must both be true:

  (a) SHAP is internally well-formed: sigmoid(base_value + sum(shap_values))
      must equal the model's raw (uncalibrated) predict_proba output.
  (b) That raw output must differ from the calibrated probability shown to
      the user -- confirming the two are genuinely distinct quantities, so
      the dashboard's "relative influence on raw risk score" caption
      (rather than implying the waterfall sums to the calibrated %) is the
      correct one, not just a defensive rephrasing of nothing.
"""

import os
import sys
import pickle
import json

import numpy as np
import pandas as pd
import shap

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(PROJECT_ROOT, "model")
ARTIFACTS_DIR = os.path.join(MODEL_DIR, "artifacts")
sys.path.insert(0, MODEL_DIR)

from train import prepare_features  # noqa: E402


def _load():
    with open(os.path.join(ARTIFACTS_DIR, "model.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(os.path.join(ARTIFACTS_DIR, "calibrator.pkl"), "rb") as f:
        calibrator = pickle.load(f)
    with open(os.path.join(ARTIFACTS_DIR, "metadata.json")) as f:
        metadata = json.load(f)
    return model, calibrator, metadata


def test_shap_sum_matches_raw_uncalibrated_output():
    model, calibrator, metadata = _load()
    test_df = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "processed", "test.csv"))
    X, _ = prepare_features(test_df, encoded_columns=metadata["encoded_columns"])
    sample = X.iloc[:25]

    raw_prob = model.predict_proba(sample)[:, 1]
    calibrated_prob = calibrator.predict(raw_prob)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)
    if isinstance(shap_values, list):
        sv = np.asarray(shap_values[1])
        base = explainer.expected_value[1]
    else:
        sv = np.asarray(shap_values)
        if sv.ndim == 3:
            sv = sv[:, :, 1]
        base = explainer.expected_value
        base = base[1] if isinstance(base, (list, np.ndarray)) and len(np.atleast_1d(base)) > 1 else float(np.atleast_1d(base)[0])

    margin_sum = base + sv.sum(axis=1)
    reconstructed_prob = 1 / (1 + np.exp(-margin_sum))

    # (a) SHAP is well-formed: reconstructing from base_value + contributions
    # (in margin/log-odds space, then sigmoid) must match predict_proba exactly.
    assert np.allclose(reconstructed_prob, raw_prob, atol=1e-6), (
        "sigmoid(base_value + sum(shap_values)) should equal the raw model's "
        "predict_proba output -- if this fails, SHAP itself is misconfigured "
        "(e.g. explaining a different output than expected)."
    )

    # (b) Raw and calibrated probabilities are genuinely, substantially
    # different -- confirming the SHAP waterfall does NOT sum to the
    # calibrated percentage shown on the dashboard, which is exactly the
    # gap the "relative influence on raw risk score" caption exists to cover.
    max_gap = np.max(np.abs(raw_prob - calibrated_prob))
    assert max_gap > 0.05, (
        f"Expected raw vs. calibrated probability to diverge meaningfully "
        f"(max gap was only {max_gap:.4f}) -- if calibration stops doing "
        f"anything, re-check the caption is still necessary."
    )


def test_mean_calibrated_probability_tracks_true_base_rate():
    """
    is_unbalance=True during training skews raw probabilities well above the
    true base rate (mean raw ~0.44 vs a ~22% actual positive rate) -- this
    confirms isotonic calibration, fit on validation, actually corrects that
    skew back toward the truth on a fully independent test set.
    """
    model, calibrator, metadata = _load()
    test_df = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "processed", "test.csv"))
    X, _ = prepare_features(test_df, encoded_columns=metadata["encoded_columns"])
    y = test_df["returned"].values

    raw_prob = model.predict_proba(X)[:, 1]
    calibrated_prob = calibrator.predict(raw_prob)

    actual_rate = y.mean()
    raw_gap = abs(raw_prob.mean() - actual_rate)
    calibrated_gap = abs(calibrated_prob.mean() - actual_rate)

    assert calibrated_gap < raw_gap, (
        f"Calibration should pull the mean predicted probability closer to "
        f"the true rate ({actual_rate:.4f}) than the raw model does "
        f"(raw mean {raw_prob.mean():.4f}, calibrated mean {calibrated_prob.mean():.4f})."
    )
    assert calibrated_gap < 0.03, (
        f"Calibrated mean probability ({calibrated_prob.mean():.4f}) drifted "
        f">3pp from the true test-set positive rate ({actual_rate:.4f})."
    )

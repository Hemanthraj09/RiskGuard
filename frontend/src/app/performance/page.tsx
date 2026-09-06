"use client";

import { useEffect, useMemo, useState } from "react";
import { getMetrics } from "@/lib/api";
import type { CostCurvePoint, EvalResults } from "@/lib/types";
import { StatTile } from "@/components/StatTile";
import { PerformanceSkeleton } from "@/components/PerformanceSkeleton";
import { ConfusionMatrix } from "@/components/ConfusionMatrix";
import { RocCurveChart } from "@/components/charts/RocCurveChart";
import { PrCurveChart } from "@/components/charts/PrCurveChart";
import { CalibrationChart } from "@/components/charts/CalibrationChart";
import { CostCurveChart } from "@/components/charts/CostCurveChart";
import { LiftChart } from "@/components/charts/LiftChart";
import { SegmentBreakdown } from "@/components/SegmentBreakdown";

const pct = (v: number) => `${(v * 100).toFixed(1)}%`;
// AUC, precision, recall, and F1 are ALL reported as decimals throughout this
// page and in eval_results.json -- mixing a decimal point-estimate with a
// percentage-formatted CI (or percentages for some metrics and decimals for
// others) reads as a bug to anyone who checks the numbers. Population/rate
// stats that aren't part of that four-metric cluster (positive rate, flag
// rate, relative cost deltas) still use `pct`.
const dec = (v: number) => v.toFixed(3);
const rupees = (v: number) =>
  `Rs.${v.toLocaleString(undefined, { maximumFractionDigits: 0 })}`;
const ciLabel = (point: number, ci: [number, number], fmt: (v: number) => string) =>
  `${fmt(point)} [${fmt(ci[0])}–${fmt(ci[1])}]`;

function costOf(e: CostCurvePoint, friction: number, ret: number, review: number) {
  return e.fp * friction + e.fn * ret + (e.fp + e.tp) * review;
}

function nearestByThreshold(curve: CostCurvePoint[], threshold: number) {
  return curve.reduce((best, e) => (Math.abs(e.threshold - threshold) < Math.abs(best.threshold - threshold) ? e : best), curve[0]);
}

export default function PerformancePage() {
  const [metrics, setMetrics] = useState<EvalResults | null>(null);
  const [error, setError] = useState<string | null>(null);

  const [frictionCost, setFrictionCost] = useState<number>(180);
  const [returnCost, setReturnCost] = useState<number>(650);
  const [reviewCost, setReviewCost] = useState<number>(50);

  useEffect(() => {
    getMetrics()
      .then((m) => {
        setMetrics(m);
        setFrictionCost(m.threshold_selection.friction_cost);
        setReturnCost(m.threshold_selection.return_cost);
        setReviewCost(m.threshold_selection.review_cost);
      })
      .catch((e) => setError(String(e)));
  }, []);

  const derived = useMemo(() => {
    if (!metrics) return null;
    // Threshold is SELECTED against the validation cost curve only...
    const validationCurve = metrics.threshold_selection.validation_cost_curve.map((e) => ({
      threshold: e.threshold,
      total_cost: costOf(e, frictionCost, returnCost, reviewCost),
    }));
    const selected = validationCurve.reduce((best, e) => (e.total_cost < best.total_cost ? e : best), validationCurve[0]);

    // ...then LOOKED UP (never re-selected) on the test cost curve, so the
    // displayed test performance is what that validation-chosen threshold
    // actually does on data it never influenced.
    const testEntry = nearestByThreshold(metrics.test_metrics.test_cost_curve, selected.threshold);
    const testCost = costOf(testEntry, frictionCost, returnCost, reviewCost);
    const f1 = (2 * testEntry.precision * testEntry.recall) / (testEntry.precision + testEntry.recall || 1);

    return { validationCurve, selected, testEntry, testCost, f1 };
  }, [metrics, frictionCost, returnCost, reviewCost]);

  if (error) {
    return (
      <div className="panel p-6 text-sm" style={{ color: "var(--status-critical)" }}>
        Failed to load metrics from the API: {error}. Is the FastAPI server running at{" "}
        {process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000"}?
      </div>
    );
  }

  if (!metrics || !derived) {
    return <PerformanceSkeleton />;
  }

  const { selected, testEntry, testCost, f1 } = derived;
  const isDefaultCosts =
    frictionCost === metrics.threshold_selection.friction_cost &&
    returnCost === metrics.threshold_selection.return_cost &&
    reviewCost === metrics.threshold_selection.review_cost;
  const testMatrix: [[number, number], [number, number]] = [
    [testEntry.tn, testEntry.fp],
    [testEntry.fn, testEntry.tp],
  ];
  const tm = metrics.test_metrics;

  return (
    <div className="flex flex-col gap-8">
      <div>
        <h1 className="text-xl font-semibold" style={{ color: "var(--text-primary)" }}>
          Model Performance
        </h1>
        <p className="mt-1 text-sm" style={{ color: "var(--text-secondary)" }}>
          Held-out test set &middot; {metrics.test_set_size.toLocaleString()} orders &middot;{" "}
          {pct(metrics.positive_rate)} actually returned. Validation set (
          {metrics.validation_set_size.toLocaleString()} orders) is used only to select the
          decision threshold below &mdash; never to evaluate it.
        </p>
      </div>

      <div>
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4 lg:grid-cols-7">
          <StatTile label="ROC-AUC" value={metrics.roc_auc.toFixed(3)} />
          <StatTile label="PR-AUC" value={metrics.pr_auc.toFixed(3)} sublabel="Avg. precision" />
          <StatTile label="Ceiling AUC" value={metrics.bayes_optimal_ceiling_auc.toFixed(3)} sublabel="Bayes-optimal" />
          <StatTile
            label="Ceiling captured"
            value={`${((metrics.roc_auc / metrics.bayes_optimal_ceiling_auc) * 100).toFixed(0)}%`}
          />
          <StatTile label="Brier score" value={metrics.brier_score.toFixed(3)} sublabel="Lower is better" />
          <StatTile label="ECE" value={metrics.ece.toFixed(3)} sublabel="Calibration error" />
          <StatTile label="Threshold" value={metrics.threshold_selection.optimal_threshold.toFixed(3)} sublabel="Cost-optimal" />
        </div>
        <p className="mt-2 text-xs" style={{ color: "var(--text-muted)" }}>
          AUC captures {((metrics.roc_auc / metrics.bayes_optimal_ceiling_auc) * 100).toFixed(0)}% of the
          theoretically achievable ceiling.
        </p>
      </div>

      <section
        className="rounded-xl p-6 text-white"
        style={{ background: "linear-gradient(135deg, var(--series-1), #184f95)" }}
      >
        <div className="text-xs font-medium uppercase tracking-wide opacity-80">
          Per 1,000 orders, at default cost assumptions
        </div>
        <div className="mt-2 flex flex-wrap items-baseline gap-x-8 gap-y-2">
          <div>
            <div className="text-3xl font-bold tabular">
              {rupees(metrics.headline_savings.savings_vs_flag_nothing_per_1000)}
            </div>
            <div className="text-sm opacity-90">saved vs. screening nothing</div>
          </div>
          <div>
            <div className="text-3xl font-bold tabular">
              {rupees(metrics.headline_savings.savings_vs_flag_everything_per_1000)}
            </div>
            <div className="text-sm opacity-90">saved vs. screening every order</div>
          </div>
        </div>
        <div className="mt-3 text-xs opacity-75">
          Model cost {rupees(metrics.headline_savings.model_cost_per_1000)}/1000 vs.{" "}
          {rupees(metrics.headline_savings.flag_nothing_cost_per_1000)}/1000 (flag nothing) and{" "}
          {rupees(metrics.headline_savings.flag_everything_cost_per_1000)}/1000 (flag everything).
        </div>
      </section>

      <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
        <StatTile
          label="ROC-AUC"
          value={metrics.roc_auc.toFixed(3)}
          sublabel={`95% CI ${tm.roc_auc_ci[0].toFixed(3)}–${tm.roc_auc_ci[1].toFixed(3)} · ceiling ${metrics.bayes_optimal_ceiling_auc.toFixed(3)}`}
        />
        <StatTile
          label="Precision"
          value={ciLabel(testEntry.precision, tm.precision_ci, dec)}
          sublabel="Cost-optimized, not maximized"
        />
        <StatTile
          label="Recall"
          value={ciLabel(testEntry.recall, tm.recall_ci, dec)}
          sublabel="Cost-optimized, not maximized"
        />
        <StatTile
          label="F1"
          value={ciLabel(f1, tm.f1_ci, dec)}
          sublabel={`95% CI, n=${tm.n_bootstrap} bootstrap`}
        />
      </div>

      <section className="panel p-6">
        <div className="flex flex-wrap items-baseline justify-between gap-2">
          <h2 className="text-base font-semibold" style={{ color: "var(--text-primary)" }}>
            Cost-based threshold analysis
          </h2>
          {!isDefaultCosts && (
            <button
              onClick={() => {
                setFrictionCost(metrics.threshold_selection.friction_cost);
                setReturnCost(metrics.threshold_selection.return_cost);
                setReviewCost(metrics.threshold_selection.review_cost);
              }}
              className="text-xs font-medium underline"
              style={{ color: "var(--series-1)" }}
            >
              Reset to defaults
            </button>
          )}
        </div>
        <p className="mt-1 text-sm" style={{ color: "var(--text-secondary)" }}>
          Instead of an arbitrary 0.5 cutoff, we pick the threshold that minimizes expected total
          cost = (false positives &times; friction cost) + (false negatives &times; return cost) +
          (flagged orders &times; review cost) &mdash; <strong>selected on the validation set</strong>,
          then applied to test. Move the sliders to see the threshold recompute live.
        </p>

        <div className="mt-5 grid grid-cols-1 gap-6 sm:grid-cols-3">
          <div>
            <div className="flex items-center justify-between text-sm">
              <label style={{ color: "var(--text-primary)" }} className="font-medium">
                Friction cost (FP)
              </label>
              <span className="tabular font-semibold" style={{ color: "var(--series-1)" }}>
                {rupees(frictionCost)}
              </span>
            </div>
            <p className="mt-0.5 text-xs" style={{ color: "var(--text-muted)" }}>
              Flagging a legitimate order for extra verification.
            </p>
            <input
              type="range" min={20} max={1000} step={10} value={frictionCost}
              onChange={(e) => setFrictionCost(Number(e.target.value))}
              className="mt-2 w-full accent-[--series-1]"
            />
          </div>
          <div>
            <div className="flex items-center justify-between text-sm">
              <label style={{ color: "var(--text-primary)" }} className="font-medium">
                Return cost (FN)
              </label>
              <span className="tabular font-semibold" style={{ color: "var(--series-1)" }}>
                {rupees(returnCost)}
              </span>
            </div>
            <p className="mt-0.5 text-xs" style={{ color: "var(--text-muted)" }}>
              Missing a genuine high-risk order (reverse logistics + write-off).
            </p>
            <input
              type="range" min={50} max={2000} step={10} value={returnCost}
              onChange={(e) => setReturnCost(Number(e.target.value))}
              className="mt-2 w-full accent-[--series-1]"
            />
          </div>
          <div>
            <div className="flex items-center justify-between text-sm">
              <label style={{ color: "var(--text-primary)" }} className="font-medium">
                Review cost (flagged)
              </label>
              <span className="tabular font-semibold" style={{ color: "var(--series-1)" }}>
                {rupees(reviewCost)}
              </span>
            </div>
            <p className="mt-0.5 text-xs" style={{ color: "var(--text-muted)" }}>
              Analyst time to review every flagged order &mdash; flagging isn&apos;t free.
            </p>
            <input
              type="range" min={0} max={300} step={5} value={reviewCost}
              onChange={(e) => setReviewCost(Number(e.target.value))}
              className="mt-2 w-full accent-[--series-1]"
            />
          </div>
        </div>

        <div className="mt-6 grid grid-cols-1 gap-6 lg:grid-cols-2">
          <div>
            <h3 className="text-sm font-semibold" style={{ color: "var(--text-primary)" }}>
              1. Select threshold on validation set
            </h3>
            <CostCurveChart
              data={derived.validationCurve}
              optimalThreshold={selected.threshold}
              optimalCost={selected.total_cost}
            />
            <div className="mt-2 grid grid-cols-2 gap-3">
              <StatTile label="Selected threshold" value={selected.threshold.toFixed(3)} sublabel="Chosen on validation" />
              <StatTile label="Validation cost" value={rupees(selected.total_cost)} sublabel="At selected threshold" />
            </div>
          </div>
          <div>
            <h3 className="text-sm font-semibold" style={{ color: "var(--text-primary)" }}>
              2. Apply it, blind, to held-out test set
            </h3>
            <div className="mt-4">
              <ConfusionMatrix matrix={testMatrix} />
            </div>
            <div className="mt-4 grid grid-cols-3 gap-3">
              <StatTile label="Precision" value={dec(testEntry.precision)} />
              <StatTile label="Recall" value={dec(testEntry.recall)} />
              <StatTile label="Test cost" value={rupees(testCost)} />
            </div>
          </div>
        </div>
        <p className="mt-4 text-xs" style={{ color: "var(--text-muted)" }}>
          {metrics.threshold_selection.method}
        </p>
      </section>

      <section className="panel p-6">
        <h2 className="text-base font-semibold" style={{ color: "var(--text-primary)" }}>
          Floor and ceiling: baselines
        </h2>
        <p className="mt-1 text-sm" style={{ color: "var(--text-secondary)" }}>
          A ranked score needs to be judged against something simpler, not just against itself.
        </p>
        <div className="mt-4 overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b text-left text-xs" style={{ borderColor: "var(--border)", color: "var(--text-muted)" }}>
                <th className="py-2 pr-4 font-medium">Approach</th>
                <th className="py-2 pr-4 font-medium">AUC</th>
                <th className="py-2 pr-4 font-medium">Precision</th>
                <th className="py-2 pr-4 font-medium">Recall</th>
                <th className="py-2 pr-4 font-medium text-right">Cost on test</th>
              </tr>
            </thead>
            <tbody>
              <tr className="border-b" style={{ borderColor: "var(--gridline)" }}>
                <td className="py-2 pr-4" style={{ color: "var(--text-primary)" }}>
                  Heuristic rule
                  <div className="text-xs" style={{ color: "var(--text-muted)" }}>{metrics.baselines.heuristic.rule}</div>
                </td>
                <td className="py-2 pr-4" style={{ color: "var(--text-muted)" }}>&mdash;</td>
                <td className="tabular py-2 pr-4">{dec(metrics.baselines.heuristic.precision)}</td>
                <td className="tabular py-2 pr-4">{dec(metrics.baselines.heuristic.recall)}</td>
                <td className="tabular py-2 pr-4 text-right">{rupees(metrics.baselines.heuristic.total_cost)}</td>
              </tr>
              <tr className="border-b" style={{ borderColor: "var(--gridline)" }}>
                <td className="py-2 pr-4" style={{ color: "var(--text-primary)" }}>
                  Logistic regression
                  <div className="text-xs" style={{ color: "var(--text-muted)" }}>Same features, same split, no calibration</div>
                </td>
                <td className="tabular py-2 pr-4">{metrics.baselines.logistic_regression.test_auc.toFixed(3)}</td>
                <td className="py-2 pr-4" colSpan={2} style={{ color: "var(--text-muted)" }}>
                  ranked score, no single operating point
                </td>
                <td className="py-2 pr-4 text-right" style={{ color: "var(--text-muted)" }}>&mdash;</td>
              </tr>
              <tr>
                <td className="py-2 pr-4 font-semibold" style={{ color: "var(--series-1)" }}>
                  LightGBM (this model)
                </td>
                <td className="tabular py-2 pr-4 font-semibold" style={{ color: "var(--series-1)" }}>
                  {metrics.baselines.lightgbm_test_auc.toFixed(3)}
                </td>
                <td className="tabular py-2 pr-4 font-semibold" style={{ color: "var(--series-1)" }}>{dec(testEntry.precision)}</td>
                <td className="tabular py-2 pr-4 font-semibold" style={{ color: "var(--series-1)" }}>{dec(testEntry.recall)}</td>
                <td className="tabular py-2 pr-4 text-right font-semibold" style={{ color: "var(--series-1)" }}>
                  {rupees(metrics.baselines.lightgbm_test_cost_at_frozen_threshold)}
                </td>
              </tr>
            </tbody>
          </table>
        </div>
        <p className="mt-3 text-xs" style={{ color: "var(--text-muted)" }}>
          The heuristic has higher precision but much lower recall &mdash; given return cost far
          exceeds friction cost, that trade-off costs{" "}
          {pct((metrics.baselines.heuristic.total_cost - metrics.baselines.lightgbm_test_cost_at_frozen_threshold) / metrics.baselines.heuristic.total_cost)}{" "}
          more overall. Logistic regression on identical features scores statistically the same
          AUC as LightGBM &mdash; evidence the achievable signal here is close to linear, and that
          the model wasn&apos;t over-fit to appear more sophisticated than the data supports.
        </p>
      </section>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <section className="panel p-6">
          <h2 className="text-base font-semibold" style={{ color: "var(--text-primary)" }}>
            ROC curve
          </h2>
          <p className="mt-1 text-sm" style={{ color: "var(--text-secondary)" }}>
            Ranking quality across all thresholds, independent of the operating point above.
          </p>
          <RocCurveChart fpr={metrics.roc_curve.fpr} tpr={metrics.roc_curve.tpr} auc={metrics.roc_auc} />
        </section>

        <section className="panel p-6">
          <h2 className="text-base font-semibold" style={{ color: "var(--text-primary)" }}>
            Precision-recall curve
          </h2>
          <p className="mt-1 text-sm" style={{ color: "var(--text-secondary)" }}>
            PR-AUC (average precision): <strong style={{ color: "var(--text-primary)" }}>{dec(metrics.pr_auc)}</strong> &mdash;
            more informative than ROC-AUC as a single ranking-quality number on this imbalanced
            ({pct(metrics.positive_rate)} positive) problem.
          </p>
          <PrCurveChart
            precision={metrics.pr_curve.precision}
            recall={metrics.pr_curve.recall}
            positiveRate={metrics.positive_rate}
          />
        </section>

        <section className="panel p-6">
          <h2 className="text-base font-semibold" style={{ color: "var(--text-primary)" }}>
            Calibration curve
          </h2>
          <p className="mt-1 text-sm" style={{ color: "var(--text-secondary)" }}>
            Does a predicted 70% probability actually mean ~70% of those orders return?
          </p>
          <CalibrationChart
            predicted={metrics.calibration_curve.predicted_probability}
            actual={metrics.calibration_curve.actual_return_rate}
          />
          <div className="mt-2 flex gap-4 text-sm" style={{ color: "var(--text-secondary)" }}>
            <span>
              <strong style={{ color: "var(--text-primary)" }}>Brier score:</strong> {metrics.brier_score.toFixed(3)}
            </span>
            <span>
              <strong style={{ color: "var(--text-primary)" }}>ECE:</strong> {metrics.ece.toFixed(3)}
            </span>
          </div>
        </section>

        <section className="panel p-6">
          <h2 className="text-base font-semibold" style={{ color: "var(--text-primary)" }}>
            Lift / gains curve
          </h2>
          <p className="mt-1 text-sm" style={{ color: "var(--text-secondary)" }}>
            If an analyst reviews only the riskiest 10% of orders, how many actual returns get caught?
          </p>
          <LiftChart
            decilePct={metrics.lift_curve.decile_pct}
            captureRate={metrics.lift_curve.capture_rate}
            randomBaseline={metrics.lift_curve.random_baseline}
          />
        </section>
      </div>

      <SegmentBreakdown segments={metrics.segments} />

      {metrics.failure_case && (
        <section className="panel p-6">
          <h2 className="text-base font-semibold" style={{ color: "var(--text-primary)" }}>
            Honest failure case
          </h2>
          <p className="mt-1 text-sm" style={{ color: "var(--text-secondary)" }}>
            One high-confidence miss from the held-out set, surfaced deliberately &mdash; not every
            order fits the model.
          </p>
          <div className="mt-4 rounded-lg p-4" style={{ background: "var(--status-warning-bg)" }}>
            <div className="flex flex-wrap items-center gap-2 text-sm font-medium" style={{ color: "var(--text-primary)" }}>
              <span>{metrics.failure_case.order_id}</span>
              <span style={{ color: "var(--text-muted)" }}>&middot;</span>
              <span>{metrics.failure_case.product_category}</span>
              <span style={{ color: "var(--text-muted)" }}>&middot;</span>
              <span>{rupees(metrics.failure_case.order_value)}</span>
              <span style={{ color: "var(--text-muted)" }}>&middot;</span>
              <span>{metrics.failure_case.payment_mode}</span>
              <span
                className="ml-auto rounded-full px-2 py-0.5 text-xs font-semibold"
                style={{ background: "var(--status-warning)", color: "white" }}
              >
                {metrics.failure_case.failure_type === "false_positive" ? "False positive" : "False negative"}
              </span>
            </div>
            <p className="mt-2 text-sm" style={{ color: "var(--text-secondary)" }}>
              {metrics.failure_case.explanation}
            </p>
          </div>
        </section>
      )}
    </div>
  );
}

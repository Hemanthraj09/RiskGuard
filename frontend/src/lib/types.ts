export type ProductCategory =
  | "footwear"
  | "apparel"
  | "electronics_accessories"
  | "groceries"
  | "home_goods"
  | "beauty";

export type PaymentMode = "COD" | "prepaid_card" | "UPI" | "wallet";
export type PincodeTier = "metro" | "tier2" | "tier3";
export type RiskBand = "low" | "medium" | "high";
export type Recommendation = "accept_normally" | "flag_for_verification";
export type AnalystDecision = "confirmed_normal" | "flagged_for_verification";

export interface DecisionLogEntry {
  id: number;
  order_id: string;
  analyst_decision: AnalystDecision;
  decided_at: string;
  customer_id: string;
  product_category: ProductCategory;
  payment_mode: PaymentMode;
  order_value: number;
  predicted_probability: number;
  risk_band: RiskBand;
  returned: number | null;
}

export interface ShapContributor {
  feature: string;
  shap_value: number;
  direction: "increases_risk" | "decreases_risk";
}

export interface CustomerFeatures {
  has_return_history: boolean;
  bayesian_return_rate: number;
  customer_purchase_frequency: number;
  account_age_days: number;
  days_since_last_order: number;
  returns_last_30d: number;
  returns_last_90d: number;
  order_value_vs_customer_avg: number;
}

export interface ScoredOrder {
  order_id: string;
  customer_id: string;
  order_timestamp: string;
  order_value: number;
  product_category: ProductCategory;
  payment_mode: PaymentMode;
  discount_applied: number;
  delivery_pincode_tier: PincodeTier;
  probability: number;
  risk_band: RiskBand;
  recommendation: Recommendation;
  recommended_action: string;
  optimal_threshold: number;
  // Null for orders repopulated via GET /orders (cold-start dashboard load):
  // reconstructing these would require re-running feature computation and
  // SHAP against the order's original scoring-time history, which GET
  // /orders deliberately skips. Always present on /score and /simulate
  // responses.
  top_contributors: ShapContributor[] | null;
  customer_features: CustomerFeatures | null;
}

export interface SimulateResponse {
  orders: ScoredOrder[];
  risk_shift: number;
  band_counts: Record<RiskBand, number>;
}

export interface ConfusionMatrixMetrics {
  threshold: number;
  precision: number;
  recall: number;
  f1: number;
  confusion_matrix: [[number, number], [number, number]]; // [[TN, FP], [FN, TP]]
}

export interface CostCurvePoint {
  threshold: number;
  fp: number;
  fn: number;
  tp: number;
  tn: number;
  precision: number;
  recall: number;
}

export type ConfidenceInterval = [number, number];

export interface SegmentMetric {
  n: number;
  insufficient_sample: boolean;
  positive_rate: number;
  auc: number | null;
  precision: number;
  recall: number;
}

export interface EvalResults {
  test_set_size: number;
  validation_set_size: number;
  positive_rate: number;
  validation_positive_rate: number;
  roc_auc: number;
  pr_auc: number;
  bayes_optimal_ceiling_auc: number;
  brier_score: number;
  ece: number;
  ece_bins: {
    bin_range: [number, number];
    count: number;
    mean_predicted: number;
    actual_rate: number;
  }[];
  metrics_at_05: ConfusionMatrixMetrics;

  headline_savings: {
    n_orders: number;
    model_cost_per_1000: number;
    flag_nothing_cost_per_1000: number;
    flag_everything_cost_per_1000: number;
    savings_vs_flag_nothing_per_1000: number;
    savings_vs_flag_everything_per_1000: number;
  };

  baselines: {
    heuristic: {
      rule: string;
      precision: number;
      recall: number;
      f1: number;
      confusion_matrix: [[number, number], [number, number]];
      total_cost: number;
      flag_rate: number;
    };
    logistic_regression: {
      model: string;
      test_auc: number;
    };
    lightgbm_test_auc: number;
    lightgbm_test_cost_at_frozen_threshold: number;
  };

  segments: {
    product_category: Record<string, SegmentMetric>;
    payment_mode: Record<string, SegmentMetric>;
    customer_tenure: Record<string, SegmentMetric>;
  };

  shifted_calibration_probes: {
    risk_shift: number;
    n_orders: number;
    positive_rate: number;
    ece: number;
    auc: number | null;
  }[];

  threshold_selection: {
    method: string;
    friction_cost: number;
    return_cost: number;
    review_cost: number;
    optimal_threshold: number;
    optimal_validation_cost: number;
    validation_cost_curve: CostCurvePoint[];
    threshold_stability: {
      n_bootstrap: number;
      median_threshold: number;
      iqr: ConfidenceInterval;
      min: number;
      max: number;
    };
  };

  test_metrics: {
    threshold: number;
    note: string;
    precision: number;
    recall: number;
    f1: number;
    confusion_matrix: [[number, number], [number, number]];
    total_cost_at_default_assumptions: number;
    test_cost_curve: CostCurvePoint[];
    roc_auc_ci: ConfidenceInterval;
    precision_ci: ConfidenceInterval;
    recall_ci: ConfidenceInterval;
    f1_ci: ConfidenceInterval;
    n_bootstrap: number;
  };

  calibration_curve: {
    predicted_probability: number[];
    actual_return_rate: number[];
    n_bins: number;
  };
  roc_curve: { fpr: number[]; tpr: number[] };
  pr_curve: { precision: number[]; recall: number[] };
  lift_curve: {
    decile_pct: number[];
    capture_rate: number[];
    random_baseline: number[];
  };
  failure_case: {
    order_id: string;
    customer_id: string;
    product_category: ProductCategory;
    order_value: number;
    payment_mode: PaymentMode;
    delivery_pincode_tier: PincodeTier;
    bayesian_return_rate: number;
    predicted_probability: number;
    predicted_label: number;
    actual_label: number;
    failure_type: "false_positive" | "false_negative";
    explanation: string;
  } | null;
}

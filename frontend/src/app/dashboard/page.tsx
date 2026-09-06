"use client";

import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import { useOrders } from "@/context/OrdersContext";
import { RiskBadge } from "@/components/RiskBadge";
import { DecisionsLog } from "@/components/DecisionsLog";
import type { AnalystDecision, ScoredOrder } from "@/lib/types";
import { getOrders, postDecision, scoreOrder } from "@/lib/api";

const rupees = (v: number) => `Rs.${v.toLocaleString(undefined, { maximumFractionDigits: 0 })}`;

const CATEGORIES = ["footwear", "apparel", "electronics_accessories", "groceries", "home_goods", "beauty"];
const PAYMENT_MODES = ["COD", "prepaid_card", "UPI", "wallet"];
const TIERS = ["metro", "tier2", "tier3"];

function ManualScoreForm({ onScored }: { onScored: (order: ScoredOrder) => void }) {
  const [open, setOpen] = useState(false);
  const [orderValue, setOrderValue] = useState(2999);
  const [category, setCategory] = useState(CATEGORIES[0]);
  const [payment, setPayment] = useState(PAYMENT_MODES[0]);
  const [tier, setTier] = useState(TIERS[0]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleSubmit() {
    setLoading(true);
    setError(null);
    try {
      const result = await scoreOrder({
        order_value: orderValue,
        product_category: category,
        payment_mode: payment,
        delivery_pincode_tier: tier,
      });
      onScored(result);
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }

  if (!open) {
    return (
      <button
        onClick={() => setOpen(true)}
        className="text-sm font-medium underline"
        style={{ color: "var(--series-1)" }}
      >
        + Score a one-off order
      </button>
    );
  }

  return (
    <div className="panel p-4">
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <div>
          <label className="text-xs font-medium" style={{ color: "var(--text-muted)" }}>
            Order value (Rs.)
          </label>
          <input
            type="number"
            value={orderValue}
            onChange={(e) => setOrderValue(Number(e.target.value))}
            className="mt-1 w-full rounded-md border px-2 py-1.5 text-sm tabular"
            style={{ borderColor: "var(--border)" }}
          />
        </div>
        <div>
          <label className="text-xs font-medium" style={{ color: "var(--text-muted)" }}>
            Category
          </label>
          <select
            value={category}
            onChange={(e) => setCategory(e.target.value)}
            className="mt-1 w-full rounded-md border px-2 py-1.5 text-sm"
            style={{ borderColor: "var(--border)" }}
          >
            {CATEGORIES.map((c) => (
              <option key={c} value={c}>
                {c}
              </option>
            ))}
          </select>
        </div>
        <div>
          <label className="text-xs font-medium" style={{ color: "var(--text-muted)" }}>
            Payment mode
          </label>
          <select
            value={payment}
            onChange={(e) => setPayment(e.target.value)}
            className="mt-1 w-full rounded-md border px-2 py-1.5 text-sm"
            style={{ borderColor: "var(--border)" }}
          >
            {PAYMENT_MODES.map((p) => (
              <option key={p} value={p}>
                {p}
              </option>
            ))}
          </select>
        </div>
        <div>
          <label className="text-xs font-medium" style={{ color: "var(--text-muted)" }}>
            Delivery tier
          </label>
          <select
            value={tier}
            onChange={(e) => setTier(e.target.value)}
            className="mt-1 w-full rounded-md border px-2 py-1.5 text-sm"
            style={{ borderColor: "var(--border)" }}
          >
            {TIERS.map((t) => (
              <option key={t} value={t}>
                {t}
              </option>
            ))}
          </select>
        </div>
      </div>
      <div className="mt-3 flex items-center gap-3">
        <button
          onClick={handleSubmit}
          disabled={loading}
          className="rounded-md px-3 py-1.5 text-sm font-semibold text-white disabled:opacity-60"
          style={{ background: "var(--series-1)" }}
        >
          {loading ? "Scoring…" : "Score order"}
        </button>
        <button onClick={() => setOpen(false)} className="text-sm" style={{ color: "var(--text-muted)" }}>
          Cancel
        </button>
        {error && <span style={{ color: "var(--status-critical)" }} className="text-sm">{error}</span>}
      </div>
    </div>
  );
}

function DecisionButtons({ order, onDecided }: { order: ScoredOrder; onDecided: () => void }) {
  const [recorded, setRecorded] = useState<{ decision: AnalystDecision; decided_at: string } | null>(null);
  const [pending, setPending] = useState<AnalystDecision | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setRecorded(null);
    setError(null);
  }, [order.order_id]);

  async function handleDecide(decision: AnalystDecision) {
    setPending(decision);
    setError(null);
    try {
      const result = await postDecision(order.order_id, decision);
      setRecorded(result);
      onDecided();
    } catch (e) {
      setError(String(e));
    } finally {
      setPending(null);
    }
  }

  return (
    <div className="mt-3">
      <div className="grid grid-cols-2 gap-2">
        <button
          onClick={() => handleDecide("confirmed_normal")}
          disabled={pending !== null}
          className="rounded-md px-3 py-1.5 text-xs font-semibold disabled:opacity-60"
          style={{ background: "var(--status-good-bg)", color: "var(--status-good)" }}
        >
          {pending === "confirmed_normal" ? "Recording…" : "Confirm normal"}
        </button>
        <button
          onClick={() => handleDecide("flagged_for_verification")}
          disabled={pending !== null}
          className="rounded-md px-3 py-1.5 text-xs font-semibold disabled:opacity-60"
          style={{ background: "var(--status-critical-bg)", color: "var(--status-critical)" }}
        >
          {pending === "flagged_for_verification" ? "Recording…" : "Flag for verification"}
        </button>
      </div>
      {recorded && (
        <p className="mt-1.5 text-xs" style={{ color: "var(--text-muted)" }}>
          Decision recorded: <strong style={{ color: "var(--text-secondary)" }}>{recorded.decision.replace(/_/g, " ")}</strong>{" "}
          at {new Date(recorded.decided_at + "Z").toLocaleTimeString()}. Logged below &mdash; clicking again records a new entry.
        </p>
      )}
      {error && <p className="mt-1.5 text-xs" style={{ color: "var(--status-critical)" }}>{error}</p>}
    </div>
  );
}

function OrderDetail({ order, onDecided }: { order: ScoredOrder; onDecided: () => void }) {
  const contributors = order.top_contributors ?? [];
  const maxAbs = Math.max(...contributors.map((c) => Math.abs(c.shap_value)), 0.0001);

  return (
    <div className="panel p-5">
      <div className="flex items-start justify-between gap-2">
        <div>
          <div className="text-sm font-semibold" style={{ color: "var(--text-primary)" }}>
            {order.order_id}
          </div>
          <div className="text-xs" style={{ color: "var(--text-muted)" }}>
            {order.customer_id}
          </div>
        </div>
        <RiskBadge band={order.risk_band} />
      </div>

      <div className="tabular mt-4 text-3xl font-semibold" style={{ color: "var(--text-primary)" }}>
        {(order.probability * 100).toFixed(1)}%
      </div>
      <div className="text-xs" style={{ color: "var(--text-muted)" }}>
        calibrated return probability &middot; threshold {order.optimal_threshold.toFixed(3)}
      </div>

      <div
        className="mt-2 inline-block rounded-md px-2.5 py-1 text-xs font-semibold"
        style={{
          background: order.recommendation === "flag_for_verification" ? "var(--status-critical-bg)" : "var(--status-good-bg)",
          color: order.recommendation === "flag_for_verification" ? "var(--status-critical)" : "var(--status-good)",
        }}
      >
        {order.recommendation === "flag_for_verification" ? "Flag for verification" : "Accept normally"}
      </div>

      <div className="mt-2 rounded-md p-2.5 text-xs" style={{ background: "var(--surface)", color: "var(--text-secondary)" }}>
        <span className="font-semibold" style={{ color: "var(--text-primary)" }}>
          Suggested action:
        </span>{" "}
        {order.recommended_action}
      </div>

      <DecisionButtons order={order} onDecided={onDecided} />

      <dl className="mt-5 grid grid-cols-2 gap-y-2 text-sm">
        <dt style={{ color: "var(--text-muted)" }}>Category</dt>
        <dd className="text-right" style={{ color: "var(--text-primary)" }}>{order.product_category}</dd>
        <dt style={{ color: "var(--text-muted)" }}>Payment</dt>
        <dd className="text-right" style={{ color: "var(--text-primary)" }}>{order.payment_mode}</dd>
        <dt style={{ color: "var(--text-muted)" }}>Value</dt>
        <dd className="tabular text-right" style={{ color: "var(--text-primary)" }}>{rupees(order.order_value)}</dd>
        <dt style={{ color: "var(--text-muted)" }}>Delivery tier</dt>
        <dd className="text-right" style={{ color: "var(--text-primary)" }}>{order.delivery_pincode_tier}</dd>
        <dt style={{ color: "var(--text-muted)" }}>Customer return history</dt>
        <dd className="tabular text-right" style={{ color: "var(--text-primary)" }}>
          {order.customer_features ? `${(order.customer_features.bayesian_return_rate * 100).toFixed(1)}%` : "—"}
        </dd>
      </dl>

      {order.customer_features && (
        <div className="mt-5">
          <h3 className="text-xs font-semibold uppercase tracking-wide" style={{ color: "var(--text-muted)" }}>
            Customer profile
          </h3>
          <dl className="mt-2 grid grid-cols-2 gap-y-2 text-sm">
            <dt style={{ color: "var(--text-muted)" }}>Return rate (bayesian)</dt>
            <dd className="tabular text-right" style={{ color: "var(--text-primary)" }}>
              {(order.customer_features.bayesian_return_rate * 100).toFixed(1)}%
            </dd>
            {!order.customer_features.has_return_history && (
              <p className="col-span-2 -mt-1 text-xs" style={{ color: "var(--text-muted)" }}>
                Population average &mdash; no resolved order history for this customer yet.
              </p>
            )}
            <dt style={{ color: "var(--text-muted)" }}>Returns (30d / 90d)</dt>
            <dd className="tabular text-right" style={{ color: "var(--text-primary)" }}>
              {order.customer_features.returns_last_30d} / {order.customer_features.returns_last_90d}
            </dd>
            <dt style={{ color: "var(--text-muted)" }}>Last order</dt>
            <dd className="tabular text-right" style={{ color: "var(--text-primary)" }}>
              {order.customer_features.days_since_last_order === -1
                ? "First order"
                : `${order.customer_features.days_since_last_order} days ago`}
            </dd>
            <dt style={{ color: "var(--text-muted)" }}>Value vs. customer avg</dt>
            <dd className="tabular text-right" style={{ color: "var(--text-primary)" }}>
              {order.customer_features.order_value_vs_customer_avg.toFixed(1)}&times;
            </dd>
            <dt style={{ color: "var(--text-muted)" }}>Purchase frequency</dt>
            <dd className="tabular text-right" style={{ color: "var(--text-primary)" }}>
              {order.customer_features.customer_purchase_frequency.toFixed(1)} orders/month
            </dd>
            <dt style={{ color: "var(--text-muted)" }}>Account age</dt>
            <dd className="tabular text-right" style={{ color: "var(--text-primary)" }}>
              {order.customer_features.account_age_days} days
            </dd>
          </dl>
        </div>
      )}

      <div className="mt-5">
        <h3 className="text-xs font-semibold uppercase tracking-wide" style={{ color: "var(--text-muted)" }}>
          Why this score (SHAP)
        </h3>
        {contributors.length === 0 ? (
          <p className="mt-1 text-xs" style={{ color: "var(--text-muted)" }}>
            Not available for orders loaded from history &mdash; SHAP explanations are only computed
            at scoring time.
          </p>
        ) : (
        <>
        <p className="mt-1 text-xs" style={{ color: "var(--text-muted)" }}>
          Relative influence on the raw risk score. SHAP explains the underlying model directly;
          the calibrated probability above passes that raw score through an additional isotonic
          correction, so these values won&apos;t sum to exactly that percentage &mdash; they
          explain the ranking, not the exact calibrated number.
        </p>
        <div className="mt-3 flex flex-col gap-2.5">
          {contributors.map((c, i) => {
            const width = (Math.abs(c.shap_value) / maxAbs) * 100;
            const color = c.direction === "increases_risk" ? "var(--status-critical)" : "var(--status-good)";
            return (
              <div key={i}>
                <div className="flex justify-between text-xs" style={{ color: "var(--text-secondary)" }}>
                  <span>{c.feature}</span>
                  <span className="tabular" style={{ color }}>
                    {c.shap_value > 0 ? "+" : ""}
                    {c.shap_value.toFixed(3)}
                  </span>
                </div>
                <div className="mt-1 h-1.5 w-full overflow-hidden rounded-full" style={{ background: "var(--gridline)" }}>
                  <div className="h-full rounded-full" style={{ width: `${width}%`, background: color }} />
                </div>
              </div>
            );
          })}
        </div>
        </>
        )}
      </div>
    </div>
  );
}

export default function DashboardPage() {
  const { orders, addOrders, selectedOrder, selectOrder } = useOrders();
  const [decisionsRefreshKey, setDecisionsRefreshKey] = useState(0);
  const hasFetchedOrders = useRef(false);
  const prevOrderIdsRef = useRef<Set<string> | null>(null);
  const [newOrderIds, setNewOrderIds] = useState<Set<string>>(new Set());

  useEffect(() => {
    if (hasFetchedOrders.current) return;
    hasFetchedOrders.current = true;
    getOrders(200)
      .then((res) => addOrders(res.orders))
      .catch(() => {
        // Cold-start seeding is a nice-to-have -- if it fails, the feed
        // just starts empty as before, same as if the API were briefly down.
      });
  }, [addOrders]);

  // Slide-in animation for rows that newly appear in the feed (cold-start
  // load, manual score submission, or a simulated batch arriving via SSE
  // while this page is mounted) -- but not for rows already present on
  // first mount (e.g. restored from sessionStorage or from a simulate run
  // on another page before navigating here).
  useEffect(() => {
    const currentIds = new Set(orders.map((o) => o.order_id));
    if (prevOrderIdsRef.current === null) {
      prevOrderIdsRef.current = currentIds;
      return;
    }
    const added = orders.filter((o) => !prevOrderIdsRef.current!.has(o.order_id)).map((o) => o.order_id);
    prevOrderIdsRef.current = currentIds;
    if (added.length === 0) return;
    setNewOrderIds(new Set(added));
    const t = setTimeout(() => setNewOrderIds(new Set()), 600);
    return () => clearTimeout(t);
  }, [orders]);

  return (
    <div className="flex flex-col gap-6">
      {/* Bounded to the viewport (minus NavBar + this <main>'s padding) only at
          lg: and up, matching the breakpoint where the grid below becomes two
          side-by-side columns. Below lg everything reverts to plain, unbounded
          stacked flow -- a fixed-height two-pane workspace only makes sense
          once there's room for two panes side by side. The header row is
          `shrink-0` (always its natural height, whether the score form is
          open or not) and the grid is `flex-1` so it gets exactly whatever
          height remains -- no need to separately estimate the header row's
          own (variable) height. */}
      <div className="flex flex-col gap-6 lg:h-[calc(100vh-130px)] lg:min-h-0">
        <div className="flex shrink-0 flex-wrap items-start justify-between gap-3">
          <div>
            <h1 className="text-xl font-semibold" style={{ color: "var(--text-primary)" }}>
              Risk Analyst Dashboard
            </h1>
            <p className="mt-1 text-sm" style={{ color: "var(--text-secondary)" }}>
              Live-scored order feed. Click any order for its SHAP-based explanation.
            </p>
          </div>
          <ManualScoreForm onScored={(o) => { addOrders([o]); selectOrder(o); }} />
        </div>

        {orders.length === 0 ? (
          <div className="panel p-10 text-center text-sm" style={{ color: "var(--text-muted)" }}>
            No orders yet.{" "}
            <Link href="/simulate" className="font-medium underline" style={{ color: "var(--series-1)" }}>
              Generate a batch in the Simulation Console
            </Link>{" "}
            to populate the live feed.
          </div>
        ) : (
        <div className="grid grid-cols-1 gap-6 lg:min-h-0 lg:flex-1 lg:grid-cols-[1fr_360px]">
          <div className="panel flex flex-col overflow-hidden lg:min-h-0">
            <div className="overflow-y-auto lg:min-h-0 lg:flex-1">
              <table className="w-full text-sm">
                <thead className="sticky top-0" style={{ background: "var(--surface-raised)" }}>
                  <tr className="border-b text-left text-xs" style={{ borderColor: "var(--border)", color: "var(--text-muted)" }}>
                    <th className="px-4 py-2.5 font-medium">Order</th>
                    <th className="px-4 py-2.5 font-medium">Category</th>
                    <th className="px-4 py-2.5 font-medium">Payment</th>
                    <th className="px-4 py-2.5 font-medium text-right">Value</th>
                    <th className="px-4 py-2.5 font-medium text-right">Probability</th>
                    <th className="px-4 py-2.5 font-medium">Risk</th>
                  </tr>
                </thead>
                <tbody>
                  {orders.map((order) => (
                    <tr
                      key={order.order_id}
                      onClick={() => selectOrder(order)}
                      className={`cursor-pointer border-b text-sm transition-colors last:border-0${
                        newOrderIds.has(order.order_id) ? " animate-in" : ""
                      }`}
                      style={{
                        borderColor: "var(--gridline)",
                        background: selectedOrder?.order_id === order.order_id ? "var(--series-1-soft)" : "transparent",
                      }}
                    >
                      <td className="px-4 py-2.5" style={{ color: "var(--text-primary)" }}>
                        <div className="font-medium">{order.order_id}</div>
                        <div className="text-xs" style={{ color: "var(--text-muted)" }}>{order.customer_id}</div>
                      </td>
                      <td className="px-4 py-2.5" style={{ color: "var(--text-secondary)" }}>{order.product_category}</td>
                      <td className="px-4 py-2.5" style={{ color: "var(--text-secondary)" }}>{order.payment_mode}</td>
                      <td className="tabular px-4 py-2.5 text-right" style={{ color: "var(--text-secondary)" }}>
                        {rupees(order.order_value)}
                      </td>
                      <td className="tabular px-4 py-2.5 text-right font-medium" style={{ color: "var(--text-primary)" }}>
                        {(order.probability * 100).toFixed(1)}%
                      </td>
                      <td className="px-4 py-2.5">
                        <RiskBadge band={order.risk_band} />
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          <div className="overflow-y-auto lg:h-full lg:min-h-0">
            {selectedOrder ? (
              <OrderDetail order={selectedOrder} onDecided={() => setDecisionsRefreshKey((k) => k + 1)} />
            ) : (
              <div className="panel p-6 text-center text-sm" style={{ color: "var(--text-muted)" }}>
                Select an order to see its explanation.
              </div>
            )}
          </div>
        </div>
        )}
      </div>

      <DecisionsLog refreshKey={decisionsRefreshKey} />
    </div>
  );
}

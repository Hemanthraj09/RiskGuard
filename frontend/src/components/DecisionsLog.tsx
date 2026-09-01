"use client";

import { useEffect, useState } from "react";
import { getDecisions } from "@/lib/api";
import type { DecisionLogEntry } from "@/lib/types";
import { RiskBadge } from "@/components/RiskBadge";
import { StatTile } from "@/components/StatTile";

const rupees = (v: number) => `Rs.${v.toLocaleString(undefined, { maximumFractionDigits: 0 })}`;

const DECISION_LABEL: Record<string, string> = {
  confirmed_normal: "Confirmed normal",
  flagged_for_verification: "Flagged for verification",
};

function agreementBadge(entry: DecisionLogEntry) {
  const modelSaysFlag = entry.risk_band === "high";
  const analystSaysFlag = entry.analyst_decision === "flagged_for_verification";
  const agrees = modelSaysFlag === analystSaysFlag;
  return (
    <span
      className="rounded-full px-2 py-0.5 text-xs font-medium"
      style={{
        background: agrees ? "var(--status-good-bg)" : "var(--status-warning-bg)",
        color: agrees ? "var(--status-good)" : "var(--status-warning)",
      }}
    >
      {agrees ? "Agrees with model" : "Overrides model"}
    </span>
  );
}

export function DecisionsLog({ refreshKey }: { refreshKey: number }) {
  const [decisions, setDecisions] = useState<DecisionLogEntry[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getDecisions(50)
      .then((res) => setDecisions(res.decisions))
      .catch((e) => setError(String(e)));
  }, [refreshKey]);

  if (error) {
    return (
      <section className="panel p-6 text-sm" style={{ color: "var(--status-critical)" }}>
        Failed to load decisions: {error}
      </section>
    );
  }

  if (decisions === null) {
    return null;
  }

  const totalDecisions = decisions.length;
  const agreements = decisions.filter(
    (d) => (d.risk_band === "high") === (d.analyst_decision === "flagged_for_verification")
  ).length;
  const overrides = totalDecisions - agreements;
  const agreementRate = totalDecisions > 0 ? agreements / totalDecisions : 0;
  const agreementRateColor =
    agreementRate > 0.7 ? "var(--status-good)" : agreementRate >= 0.5 ? "var(--status-warning)" : "var(--status-critical)";

  return (
    <section className="panel p-6">
      <h2 className="text-base font-semibold" style={{ color: "var(--text-primary)" }}>
        Decisions log
      </h2>
      <p className="mt-1 text-sm" style={{ color: "var(--text-secondary)" }}>
        Every human verify/decide action, timestamped &mdash; the model recommends, an analyst
        decides. Outcome vs. prediction: did the analyst agree with the model&apos;s risk band or
        override it?
      </p>
      {decisions.length === 0 ? (
        <div className="mt-4 text-sm" style={{ color: "var(--text-muted)" }}>
          No decisions logged yet. Click &quot;Confirm normal&quot; or &quot;Flag for
          verification&quot; on an order to record one.
        </div>
      ) : (
        <>
        <div className="mt-4 grid grid-cols-3 gap-3">
          <StatTile label="Total decisions" value={String(totalDecisions)} />
          <StatTile
            label="Agreement rate"
            value={`${(agreementRate * 100).toFixed(0)}%`}
            valueColor={agreementRateColor}
            sublabel={`${agreements} of ${totalDecisions} agree with model`}
          />
          <StatTile label="Overrides" value={String(overrides)} sublabel="Analyst disagreed with model" />
        </div>
        <div className="mt-4 max-h-[420px] overflow-y-auto overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="sticky top-0" style={{ background: "var(--surface-raised)" }}>
              <tr className="border-b text-left text-xs" style={{ borderColor: "var(--border)", color: "var(--text-muted)" }}>
                <th className="py-2 pr-4 font-medium">Order</th>
                <th className="py-2 pr-4 font-medium">Model risk</th>
                <th className="py-2 pr-4 font-medium">Analyst decision</th>
                <th className="py-2 pr-4 font-medium">Agreement</th>
                <th className="py-2 pr-4 font-medium">When</th>
              </tr>
            </thead>
            <tbody>
              {decisions.map((d) => (
                <tr key={d.id} className="border-b last:border-0" style={{ borderColor: "var(--gridline)" }}>
                  <td className="py-2 pr-4" style={{ color: "var(--text-primary)" }}>
                    <div className="font-medium">{d.order_id}</div>
                    <div className="text-xs" style={{ color: "var(--text-muted)" }}>
                      {d.product_category} &middot; {rupees(d.order_value)}
                    </div>
                  </td>
                  <td className="py-2 pr-4">
                    <RiskBadge band={d.risk_band} />
                    <div className="tabular mt-0.5 text-xs" style={{ color: "var(--text-muted)" }}>
                      {(d.predicted_probability * 100).toFixed(1)}%
                    </div>
                  </td>
                  <td className="py-2 pr-4" style={{ color: "var(--text-secondary)" }}>
                    {DECISION_LABEL[d.analyst_decision] ?? d.analyst_decision}
                  </td>
                  <td className="py-2 pr-4">{agreementBadge(d)}</td>
                  <td className="py-2 pr-4 text-xs" style={{ color: "var(--text-muted)" }}>
                    {new Date(d.decided_at + "Z").toLocaleString()}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        </>
      )}
    </section>
  );
}

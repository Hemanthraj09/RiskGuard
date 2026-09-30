"use client";

import { useState } from "react";
import Link from "next/link";
import type { RiskBand } from "@/lib/types";
import { useOrders } from "@/context/OrdersContext";

const BAND_COLOR: Record<RiskBand, string> = {
  low: "var(--status-good)",
  medium: "var(--status-warning)",
  high: "var(--status-critical)",
};

const DEFAULT_ORDERS = 100;
const MAX_ORDERS = 500;

function parseOrderCount(text: string): number {
  const value = Math.round(Number(text));
  return Number.isFinite(value) && value >= 1 ? Math.min(MAX_ORDERS, value) : DEFAULT_ORDERS;
}

export default function SimulatePage() {
  const { simulation, startSimulation } = useOrders();
  // Raw text, so the field can be cleared and retyped; parsed and clamped to
  // 1-500 on blur and on submit.
  const [countText, setCountText] = useState(String(DEFAULT_ORDERS));
  const [riskShift, setRiskShift] = useState(0);
  const n = parseOrderCount(countText);
  const running = simulation?.running ?? false;
  // Band counts arrive with the final "done" event, i.e. once the batch is saved.
  const lastCounts = simulation && !simulation.running ? simulation.bandCounts : null;
  const lastTotal = simulation?.received ?? 0;

  function handleGenerate() {
    setCountText(String(n));
    startSimulation(n, riskShift);
  }

  return (
    <div className="flex flex-col gap-8">
      <div>
        <h1 className="text-xl font-semibold" style={{ color: "var(--text-primary)" }}>
          Simulation Console
        </h1>
        <p className="mt-1 text-sm" style={{ color: "var(--text-secondary)" }}>
          Generate genuinely new orders &mdash; separate from both the train and test sets &mdash; and
          watch the trained model score them live. The risk-shift slider skews the batch toward
          higher-risk conditions (more COD, more footwear/apparel, more tier-3 delivery, mid-range
          order values).
        </p>
      </div>

      <section className="panel p-6">
        <div className="grid grid-cols-1 gap-6 sm:grid-cols-2">
          <div>
            <label className="text-sm font-medium" style={{ color: "var(--text-primary)" }}>
              Orders to generate
            </label>
            <input
              type="number"
              min={1}
              max={MAX_ORDERS}
              value={countText}
              onChange={(e) => setCountText(e.target.value)}
              onBlur={() => setCountText(String(n))}
              className="mt-2 w-full rounded-md border px-3 py-2 text-sm tabular"
              style={{ borderColor: "var(--border)", background: "var(--surface)" }}
            />
          </div>
          <div>
            <div className="flex items-center justify-between text-sm">
              <label className="font-medium" style={{ color: "var(--text-primary)" }}>
                Risk shift
              </label>
              <span className="tabular font-semibold" style={{ color: "var(--series-1)" }}>
                {(riskShift * 100).toFixed(0)}%
              </span>
            </div>
            <input
              type="range"
              min={0}
              max={1}
              step={0.05}
              value={riskShift}
              onChange={(e) => setRiskShift(Number(e.target.value))}
              className="mt-2 w-full accent-(--series-1)"
            />
            <div className="mt-1 flex justify-between text-xs" style={{ color: "var(--text-muted)" }}>
              <span>Population baseline</span>
              <span>Maximally skewed</span>
            </div>
          </div>
        </div>

        <button
          onClick={handleGenerate}
          disabled={running}
          className="mt-6 rounded-md px-4 py-2 text-sm font-semibold text-white transition-opacity disabled:opacity-60"
          style={{ background: "var(--series-1)" }}
        >
          {running && simulation
            ? `Generating… (${simulation.received}/${simulation.requested})`
            : `Generate ${n} orders`}
        </button>

        {running && (
          <p className="mt-3 text-sm" style={{ color: "var(--text-secondary)" }}>
            Scoring live &mdash; switch to the{" "}
            <Link href="/dashboard" className="font-medium underline" style={{ color: "var(--series-1)" }}>
              Risk Analyst Dashboard
            </Link>{" "}
            to watch the orders arrive; the batch keeps streaming.
          </p>
        )}

        {simulation?.error && (
          <p className="mt-3 text-sm" style={{ color: "var(--status-critical)" }}>
            {simulation.error}
          </p>
        )}
      </section>

      {lastCounts && lastTotal > 0 && (
        <section className="panel p-6">
          <div className="flex flex-wrap items-baseline justify-between gap-2">
            <h2 className="text-base font-semibold" style={{ color: "var(--text-primary)" }}>
              Last batch &middot; {lastTotal} orders
            </h2>
            <Link href="/dashboard" className="text-sm font-medium underline" style={{ color: "var(--series-1)" }}>
              View in Risk Analyst Dashboard &rarr;
            </Link>
          </div>

          <div className="mt-4 flex h-3 w-full overflow-hidden rounded-full">
            {(["low", "medium", "high"] as RiskBand[]).map((band) => {
              const width = (lastCounts[band] / lastTotal) * 100;
              return width > 0 ? (
                <div key={band} style={{ width: `${width}%`, background: BAND_COLOR[band] }} />
              ) : null;
            })}
          </div>

          <div className="mt-4 grid grid-cols-3 gap-4">
            {(["low", "medium", "high"] as RiskBand[]).map((band) => (
              <div key={band} className="text-center">
                <div className="tabular text-xl font-semibold" style={{ color: BAND_COLOR[band] }}>
                  {lastCounts[band]}
                </div>
                <div className="text-xs capitalize" style={{ color: "var(--text-muted)" }}>
                  {band} risk &middot; {((lastCounts[band] / lastTotal) * 100).toFixed(0)}%
                </div>
              </div>
            ))}
          </div>
        </section>
      )}
    </div>
  );
}

"use client";

import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import { simulateStreamUrl } from "@/lib/api";
import type { RiskBand, ScoredOrder } from "@/lib/types";
import { useOrders } from "@/context/OrdersContext";

const BAND_COLOR: Record<RiskBand, string> = {
  low: "var(--status-good)",
  medium: "var(--status-warning)",
  high: "var(--status-critical)",
};

export default function SimulatePage() {
  const { addOrders } = useOrders();
  const [n, setN] = useState(100);
  const [riskShift, setRiskShift] = useState(0);
  const [loading, setLoading] = useState(false);
  const [received, setReceived] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [lastBatch, setLastBatch] = useState<{ bandCounts: Record<RiskBand, number>; total: number } | null>(null);
  const sourceRef = useRef<EventSource | null>(null);

  useEffect(() => () => sourceRef.current?.close(), []);

  function handleGenerate() {
    sourceRef.current?.close();
    setLoading(true);
    setError(null);
    setReceived(0);
    setLastBatch(null);

    const source = new EventSource(simulateStreamUrl(n, riskShift));
    sourceRef.current = source;
    let count = 0;

    source.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.done) {
        setLastBatch({ bandCounts: data.band_counts, total: count });
        source.close();
        sourceRef.current = null;
        setLoading(false);
        return;
      }
      addOrders([data as ScoredOrder]);
      count += 1;
      setReceived(count);
    };

    source.onerror = () => {
      source.close();
      sourceRef.current = null;
      setLoading(false);
      setError("Streaming connection to the API failed.");
    };
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
              max={500}
              value={n}
              onChange={(e) => setN(Math.max(1, Math.min(500, Number(e.target.value))))}
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
              className="mt-2 w-full accent-[--series-1]"
            />
            <div className="mt-1 flex justify-between text-xs" style={{ color: "var(--text-muted)" }}>
              <span>Population baseline</span>
              <span>Maximally skewed</span>
            </div>
          </div>
        </div>

        <button
          onClick={handleGenerate}
          disabled={loading}
          className="mt-6 rounded-md px-4 py-2 text-sm font-semibold text-white transition-opacity disabled:opacity-60"
          style={{ background: "var(--series-1)" }}
        >
          {loading ? `Generating… (${received}/${n})` : `Generate ${n} orders`}
        </button>

        {error && (
          <p className="mt-3 text-sm" style={{ color: "var(--status-critical)" }}>
            {error}
          </p>
        )}
      </section>

      {lastBatch && (
        <section className="panel p-6">
          <div className="flex flex-wrap items-baseline justify-between gap-2">
            <h2 className="text-base font-semibold" style={{ color: "var(--text-primary)" }}>
              Last batch &middot; {lastBatch.total} orders
            </h2>
            <Link href="/dashboard" className="text-sm font-medium underline" style={{ color: "var(--series-1)" }}>
              View in Risk Analyst Dashboard &rarr;
            </Link>
          </div>

          <div className="mt-4 flex h-3 w-full overflow-hidden rounded-full">
            {(["low", "medium", "high"] as RiskBand[]).map((band) => {
              const width = (lastBatch.bandCounts[band] / lastBatch.total) * 100;
              return width > 0 ? (
                <div key={band} style={{ width: `${width}%`, background: BAND_COLOR[band] }} />
              ) : null;
            })}
          </div>

          <div className="mt-4 grid grid-cols-3 gap-4">
            {(["low", "medium", "high"] as RiskBand[]).map((band) => (
              <div key={band} className="text-center">
                <div className="tabular text-xl font-semibold" style={{ color: BAND_COLOR[band] }}>
                  {lastBatch.bandCounts[band]}
                </div>
                <div className="text-xs capitalize" style={{ color: "var(--text-muted)" }}>
                  {band} risk &middot; {((lastBatch.bandCounts[band] / lastBatch.total) * 100).toFixed(0)}%
                </div>
              </div>
            ))}
          </div>
        </section>
      )}
    </div>
  );
}

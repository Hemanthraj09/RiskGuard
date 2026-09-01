"use client";

import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

export function LiftChart({
  decilePct,
  captureRate,
  randomBaseline,
}: {
  decilePct: number[];
  captureRate: number[];
  randomBaseline: number[];
}) {
  const data = decilePct.map((d, i) => ({
    decile: d,
    capture: captureRate[i],
    baseline: randomBaseline[i],
  }));

  return (
    <ResponsiveContainer width="100%" height={260}>
      <LineChart data={data} margin={{ top: 8, right: 16, bottom: 4, left: 0 }}>
        <CartesianGrid stroke="var(--gridline)" vertical={false} />
        <XAxis
          type="number"
          dataKey="decile"
          domain={[0, 100]}
          tick={{ fill: "var(--text-muted)", fontSize: 11 }}
          stroke="var(--baseline)"
          tickFormatter={(v: number) => `${v}%`}
          label={{ value: "% of orders reviewed (riskiest first)", position: "insideBottom", offset: -2, fontSize: 11, fill: "var(--text-secondary)" }}
        />
        <YAxis
          type="number"
          domain={[0, 1]}
          tick={{ fill: "var(--text-muted)", fontSize: 11 }}
          stroke="var(--baseline)"
          tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`}
          label={{ value: "% of actual returns captured", angle: -90, position: "insideLeft", fontSize: 11, fill: "var(--text-secondary)" }}
        />
        <Tooltip
          formatter={(value) => `${(Number(value) * 100).toFixed(1)}%`}
          labelFormatter={(v) => `Top ${v}% by risk`}
          contentStyle={{ fontSize: 12, borderRadius: 8, border: "1px solid var(--border)" }}
        />
        <Legend wrapperStyle={{ fontSize: 12 }} />
        <Line
          dataKey="baseline"
          name="Random review order"
          stroke="var(--baseline)"
          strokeDasharray="4 4"
          strokeWidth={1.5}
          dot={false}
          isAnimationActive={false}
        />
        <Line
          dataKey="capture"
          name="Model-prioritized review"
          stroke="var(--series-1)"
          strokeWidth={2}
          dot={{ r: 3, fill: "var(--series-1)", strokeWidth: 0 }}
          isAnimationActive={false}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}

"use client";

import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceDot,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

export function CostCurveChart({
  data,
  optimalThreshold,
  optimalCost,
}: {
  data: { threshold: number; total_cost: number }[];
  optimalThreshold: number;
  optimalCost: number;
}) {
  return (
    <ResponsiveContainer width="100%" height={280}>
      <LineChart data={data} margin={{ top: 8, right: 24, bottom: 4, left: 8 }}>
        <CartesianGrid stroke="var(--gridline)" vertical={false} />
        <XAxis
          type="number"
          dataKey="threshold"
          domain={[0, 1]}
          tick={{ fill: "var(--text-muted)", fontSize: 11 }}
          stroke="var(--baseline)"
          label={{ value: "Decision threshold", position: "insideBottom", offset: -2, fontSize: 11, fill: "var(--text-secondary)" }}
        />
        <YAxis
          tick={{ fill: "var(--text-muted)", fontSize: 11 }}
          stroke="var(--baseline)"
          tickFormatter={(v: number) => `Rs.${(v / 1000).toFixed(0)}k`}
          label={{ value: "Expected total cost", angle: -90, position: "insideLeft", fontSize: 11, fill: "var(--text-secondary)" }}
        />
        <Tooltip
          formatter={(value) => [`Rs.${Number(value).toLocaleString(undefined, { maximumFractionDigits: 0 })}`, "Total cost"]}
          labelFormatter={(v) => `Threshold ${Number(v).toFixed(3)}`}
          contentStyle={{ fontSize: 12, borderRadius: 8, border: "1px solid var(--border)" }}
        />
        <ReferenceLine
          x={optimalThreshold}
          stroke="var(--status-good)"
          strokeDasharray="4 4"
          label={{ value: `Optimal: ${optimalThreshold.toFixed(3)}`, position: "top", fontSize: 11, fill: "var(--status-good)" }}
        />
        <Line
          type="monotone"
          dataKey="total_cost"
          stroke="var(--series-1)"
          strokeWidth={2}
          dot={false}
          isAnimationActive={false}
        />
        <ReferenceDot
          x={optimalThreshold}
          y={optimalCost}
          r={5}
          fill="var(--status-good)"
          stroke="var(--surface-raised)"
          strokeWidth={2}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}

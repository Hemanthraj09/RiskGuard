"use client";

import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

export function PrCurveChart({
  precision,
  recall,
  positiveRate,
}: {
  precision: number[];
  recall: number[];
  positiveRate: number;
}) {
  const data = recall.map((r, i) => ({ recall: r, precision: precision[i] }));

  return (
    <ResponsiveContainer width="100%" height={260}>
      <LineChart data={data} margin={{ top: 8, right: 16, bottom: 4, left: 0 }}>
        <CartesianGrid stroke="var(--gridline)" vertical={false} />
        <XAxis
          type="number"
          dataKey="recall"
          domain={[0, 1]}
          tick={{ fill: "var(--text-muted)", fontSize: 11 }}
          stroke="var(--baseline)"
          label={{ value: "Recall", position: "insideBottom", offset: -2, fontSize: 11, fill: "var(--text-secondary)" }}
        />
        <YAxis
          type="number"
          domain={[0, 1]}
          tick={{ fill: "var(--text-muted)", fontSize: 11 }}
          stroke="var(--baseline)"
          label={{ value: "Precision", angle: -90, position: "insideLeft", fontSize: 11, fill: "var(--text-secondary)" }}
        />
        <Tooltip
          formatter={(value) => Number(value).toFixed(3)}
          contentStyle={{ fontSize: 12, borderRadius: 8, border: "1px solid var(--border)" }}
        />
        <Legend wrapperStyle={{ fontSize: 12 }} />
        <ReferenceLine
          y={positiveRate}
          stroke="var(--baseline)"
          strokeDasharray="4 4"
          label={{ value: "Baseline (positive rate)", position: "insideTopRight", fontSize: 10, fill: "var(--text-muted)" }}
        />
        <Line
          type="monotone"
          dataKey="precision"
          name="Model"
          stroke="var(--series-1)"
          strokeWidth={2}
          dot={false}
          isAnimationActive={false}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}

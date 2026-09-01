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

export function CalibrationChart({
  predicted,
  actual,
}: {
  predicted: number[];
  actual: number[];
}) {
  const data = predicted.map((p, i) => ({ predicted: p, actual: actual[i] }));
  const diagonal = [
    { predicted: 0, diag: 0 },
    { predicted: 1, diag: 1 },
  ];

  return (
    <ResponsiveContainer width="100%" height={260}>
      <LineChart margin={{ top: 8, right: 16, bottom: 4, left: 0 }}>
        <CartesianGrid stroke="var(--gridline)" vertical={false} />
        <XAxis
          type="number"
          dataKey="predicted"
          domain={[0, 1]}
          tick={{ fill: "var(--text-muted)", fontSize: 11 }}
          stroke="var(--baseline)"
          label={{ value: "Predicted probability", position: "insideBottom", offset: -2, fontSize: 11, fill: "var(--text-secondary)" }}
        />
        <YAxis
          type="number"
          domain={[0, 1]}
          tick={{ fill: "var(--text-muted)", fontSize: 11 }}
          stroke="var(--baseline)"
          label={{ value: "Actual return rate", angle: -90, position: "insideLeft", fontSize: 11, fill: "var(--text-secondary)" }}
        />
        <Tooltip
          formatter={(value) => Number(value).toFixed(3)}
          contentStyle={{ fontSize: 12, borderRadius: 8, border: "1px solid var(--border)" }}
        />
        <Legend wrapperStyle={{ fontSize: 12 }} />
        <Line
          data={diagonal}
          dataKey="diag"
          name="Perfect calibration"
          stroke="var(--baseline)"
          strokeDasharray="4 4"
          strokeWidth={1.5}
          dot={false}
          isAnimationActive={false}
        />
        <Line
          data={data}
          dataKey="actual"
          name="Model (by decile bucket)"
          stroke="var(--series-1)"
          strokeWidth={2}
          dot={{ r: 4, fill: "var(--series-1)", strokeWidth: 0 }}
          isAnimationActive={false}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}

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

export function RocCurveChart({
  fpr,
  tpr,
  auc,
}: {
  fpr: number[];
  tpr: number[];
  auc: number;
}) {
  const rocData = fpr.map((f, i) => ({ fpr: f, tpr: tpr[i] }));
  const diagonal = [
    { fpr: 0, diag: 0 },
    { fpr: 1, diag: 1 },
  ];

  return (
    <div>
      <ResponsiveContainer width="100%" height={260}>
        <LineChart margin={{ top: 8, right: 16, bottom: 4, left: 0 }}>
          <CartesianGrid stroke="var(--gridline)" vertical={false} />
          <XAxis
            type="number"
            dataKey="fpr"
            domain={[0, 1]}
            tick={{ fill: "var(--text-muted)", fontSize: 11 }}
            stroke="var(--baseline)"
            label={{ value: "False positive rate", position: "insideBottom", offset: -2, fontSize: 11, fill: "var(--text-secondary)" }}
          />
          <YAxis
            type="number"
            domain={[0, 1]}
            tick={{ fill: "var(--text-muted)", fontSize: 11 }}
            stroke="var(--baseline)"
            label={{ value: "True positive rate", angle: -90, position: "insideLeft", fontSize: 11, fill: "var(--text-secondary)" }}
          />
          <Tooltip
            formatter={(value) => Number(value).toFixed(3)}
            contentStyle={{ fontSize: 12, borderRadius: 8, border: "1px solid var(--border)" }}
          />
          <Legend wrapperStyle={{ fontSize: 12 }} />
          <Line
            data={diagonal}
            dataKey="diag"
            name="Random classifier"
            stroke="var(--baseline)"
            strokeDasharray="4 4"
            strokeWidth={1.5}
            dot={false}
            isAnimationActive={false}
          />
          <Line
            data={rocData}
            dataKey="tpr"
            name={`Model (AUC = ${auc.toFixed(3)})`}
            stroke="var(--series-1)"
            strokeWidth={2}
            dot={false}
            isAnimationActive={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

export function ConfusionMatrix({ matrix }: { matrix: [[number, number], [number, number]] }) {
  const [[tn, fp], [fn, tp]] = matrix;
  const total = tn + fp + fn + tp;
  const pct = (v: number) => (total > 0 ? `${((v / total) * 100).toFixed(1)}%` : "-");

  const Cell = ({
    label,
    value,
    tone,
  }: {
    label: string;
    value: number;
    tone: "good" | "critical";
  }) => (
    <div
      className="flex flex-col items-center justify-center rounded-lg p-4"
      style={{
        background: tone === "good" ? "var(--status-good-bg)" : "var(--status-critical-bg)",
      }}
    >
      <div
        className="tabular text-xl font-semibold"
        style={{ color: tone === "good" ? "var(--status-good)" : "var(--status-critical)" }}
      >
        {value.toLocaleString()}
      </div>
      <div className="mt-0.5 text-xs" style={{ color: "var(--text-secondary)" }}>
        {label} &middot; {pct(value)}
      </div>
    </div>
  );

  return (
    <div>
      <div className="grid grid-cols-[auto_1fr_1fr] gap-2 text-xs">
        <div />
        <div className="text-center font-medium" style={{ color: "var(--text-muted)" }}>
          Predicted: No return
        </div>
        <div className="text-center font-medium" style={{ color: "var(--text-muted)" }}>
          Predicted: Returned
        </div>

        <div
          className="flex items-center justify-center px-2 text-center font-medium"
          style={{ color: "var(--text-muted)" }}
        >
          Actual:
          <br />
          No return
        </div>
        <Cell label="True Negative" value={tn} tone="good" />
        <Cell label="False Positive" value={fp} tone="critical" />

        <div
          className="flex items-center justify-center px-2 text-center font-medium"
          style={{ color: "var(--text-muted)" }}
        >
          Actual:
          <br />
          Returned
        </div>
        <Cell label="False Negative" value={fn} tone="critical" />
        <Cell label="True Positive" value={tp} tone="good" />
      </div>
    </div>
  );
}

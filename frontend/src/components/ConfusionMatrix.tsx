// Module-level, not defined inside ConfusionMatrix: a component created during
// render is a brand-new component type every render, so React would unmount
// and remount all four cells on each cost-slider move.
function Cell({
  label,
  value,
  total,
  tone,
}: {
  label: string;
  value: number;
  total: number;
  tone: "good" | "critical";
}) {
  const share = total > 0 ? `${((value / total) * 100).toFixed(1)}%` : "-";
  return (
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
        {label} &middot; {share}
      </div>
    </div>
  );
}

export function ConfusionMatrix({ matrix }: { matrix: [[number, number], [number, number]] }) {
  const [[tn, fp], [fn, tp]] = matrix;
  const total = tn + fp + fn + tp;

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
        <Cell label="True Negative" value={tn} total={total} tone="good" />
        <Cell label="False Positive" value={fp} total={total} tone="critical" />

        <div
          className="flex items-center justify-center px-2 text-center font-medium"
          style={{ color: "var(--text-muted)" }}
        >
          Actual:
          <br />
          Returned
        </div>
        <Cell label="False Negative" value={fn} total={total} tone="critical" />
        <Cell label="True Positive" value={tp} total={total} tone="good" />
      </div>
    </div>
  );
}

import type { SegmentMetric } from "@/lib/types";

// AUC/precision/recall are decimals throughout the Model Performance page --
// kept consistent with the headline KPI tiles and eval_results.json.
const dec = (v: number) => v.toFixed(3);

const SECTION_LABELS: Record<string, string> = {
  product_category: "By product category",
  payment_mode: "By payment mode",
  customer_tenure: "By customer tenure",
};

function SegmentTable({ title, data }: { title: string; data: Record<string, SegmentMetric> }) {
  const rows = Object.entries(data);
  return (
    <div>
      <h3 className="text-sm font-semibold" style={{ color: "var(--text-primary)" }}>
        {title}
      </h3>
      <div className="mt-2 overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b text-left text-xs" style={{ borderColor: "var(--border)", color: "var(--text-muted)" }}>
              <th className="py-2 pr-4 font-medium">Segment</th>
              <th className="py-2 pr-4 font-medium">n</th>
              <th className="py-2 pr-4 font-medium">AUC</th>
              <th className="py-2 pr-4 font-medium">Precision</th>
              <th className="py-2 pr-4 font-medium">Recall</th>
            </tr>
          </thead>
          <tbody>
            {rows.map(([segment, m]) => (
              <tr key={segment} className="border-b last:border-0" style={{ borderColor: "var(--gridline)" }}>
                <td className="py-2 pr-4 capitalize" style={{ color: "var(--text-primary)" }}>
                  {segment.replace(/_/g, " ")}
                  {m.insufficient_sample && (
                    <span
                      className="ml-2 rounded-full px-1.5 py-0.5 text-[10px] font-semibold"
                      style={{ background: "var(--status-warning-bg)", color: "var(--status-warning)" }}
                    >
                      low sample
                    </span>
                  )}
                </td>
                <td className="tabular py-2 pr-4" style={{ color: "var(--text-secondary)" }}>{m.n.toLocaleString()}</td>
                <td
                  className="tabular py-2 pr-4"
                  style={{ color: m.auc !== null && m.auc < 0.55 ? "var(--status-critical)" : "var(--text-primary)" }}
                >
                  {m.auc !== null ? m.auc.toFixed(3) : "n/a"}
                </td>
                <td className="tabular py-2 pr-4" style={{ color: "var(--text-secondary)" }}>{dec(m.precision)}</td>
                <td className="tabular py-2 pr-4" style={{ color: "var(--text-secondary)" }}>{dec(m.recall)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export function SegmentBreakdown({
  segments,
}: {
  segments: Record<string, Record<string, SegmentMetric>>;
}) {
  const weakSegments = Object.entries(segments).flatMap(([group, data]) =>
    Object.entries(data)
      .filter(([, m]) => m.auc !== null && m.auc < 0.55)
      .map(([seg]) => `${seg.replace(/_/g, " ")} (${SECTION_LABELS[group]?.toLowerCase() ?? group})`)
  );

  return (
    <section className="panel p-6">
      <h2 className="text-base font-semibold" style={{ color: "var(--text-primary)" }}>
        Segment-level performance
      </h2>
      <p className="mt-1 text-sm" style={{ color: "var(--text-secondary)" }}>
        The global metrics above average over segments the model treats very differently &mdash;
        this is what tells a merchant where NOT to trust the flag.
      </p>
      <div className="mt-5 flex flex-col gap-6">
        {Object.entries(segments).map(([group, data]) => (
          <SegmentTable key={group} title={SECTION_LABELS[group] ?? group} data={data} />
        ))}
      </div>
      {weakSegments.length > 0 && (
        <p className="mt-4 rounded-md p-3 text-xs" style={{ background: "var(--status-warning-bg)", color: "var(--status-warning)" }}>
          <strong>Known limitation:</strong> the model performs near or below random within{" "}
          {weakSegments.join(", ")} &mdash; these categories have low overall return rates and the
          model rarely has enough signal there to discriminate. Treat flags in these segments with
          lower confidence than the global metrics suggest.
        </p>
      )}
    </section>
  );
}

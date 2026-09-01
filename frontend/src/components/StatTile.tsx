export function StatTile({
  label,
  value,
  sublabel,
  valueColor,
}: {
  label: string;
  value: string;
  sublabel?: string;
  valueColor?: string;
}) {
  return (
    <div className="panel p-4">
      <div className="text-xs font-medium uppercase tracking-wide" style={{ color: "var(--text-muted)" }}>
        {label}
      </div>
      <div className="tabular mt-1.5 text-2xl font-semibold" style={{ color: valueColor ?? "var(--text-primary)" }}>
        {value}
      </div>
      {sublabel && (
        <div className="mt-0.5 text-xs" style={{ color: "var(--text-secondary)" }}>
          {sublabel}
        </div>
      )}
    </div>
  );
}

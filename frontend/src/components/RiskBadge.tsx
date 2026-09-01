import type { RiskBand } from "@/lib/types";

const STYLES: Record<RiskBand, { bg: string; fg: string; label: string }> = {
  low: { bg: "var(--status-good-bg)", fg: "var(--status-good)", label: "Low" },
  medium: { bg: "var(--status-warning-bg)", fg: "var(--status-warning)", label: "Medium" },
  high: { bg: "var(--status-critical-bg)", fg: "var(--status-critical)", label: "High" },
};

export function RiskBadge({ band }: { band: RiskBand }) {
  const s = STYLES[band];
  return (
    <span
      className="inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium"
      style={{ background: s.bg, color: s.fg }}
    >
      <span className="h-1.5 w-1.5 rounded-full" style={{ background: s.fg }} />
      {s.label}
    </span>
  );
}

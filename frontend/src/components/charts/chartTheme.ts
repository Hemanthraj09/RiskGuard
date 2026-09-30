import type { CSSProperties } from "react";

// Recharts' default tooltip is a white box whose label text inherits the
// page's near-white foreground -- unreadable on this dark theme. Every chart
// spreads these into its <Tooltip> instead. Item rows use one readable colour
// (each row is named) because the dashed baseline series' own stroke colour
// is too dark to read on the tooltip background.
export const TOOLTIP_STYLE: {
  contentStyle: CSSProperties;
  labelStyle: CSSProperties;
  itemStyle: CSSProperties;
} = {
  contentStyle: {
    fontSize: 12,
    borderRadius: 8,
    border: "1px solid var(--border)",
    background: "var(--surface)",
  },
  labelStyle: { color: "var(--text-primary)", fontWeight: 600 },
  itemStyle: { color: "var(--text-secondary)" },
};

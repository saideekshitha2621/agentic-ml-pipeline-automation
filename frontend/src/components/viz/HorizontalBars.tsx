import { Box, Tooltip, Typography } from "@mui/material";
import { fmt, viz } from "./vizTheme";

export interface BarRow {
  key: string;
  label: string;
  value: number | null;
  valueLabel?: string;
  highlight?: boolean;
  badge?: string;
  tooltip?: string;
}

/** Responsive HTML bar list: label | bar | direct value label. 4px-rounded data end, muted
 * bars with one highlighted entity, hover tooltip, and keyboard focus per row. */
export default function HorizontalBars({ rows, labelWidth = 170 }: { rows: BarRow[]; labelWidth?: number }) {
  const max = Math.max(...rows.map((r) => Math.abs(r.value ?? 0)), 1e-9);
  return (
    <Box role="list" sx={{ display: "flex", flexDirection: "column", gap: 1.25 }}>
      {rows.map((r) => {
        const pct = r.value === null ? 0 : (Math.abs(r.value) / max) * 100;
        return (
          <Tooltip key={r.key} title={r.tooltip ?? ""} placement="top" arrow disableHoverListener={!r.tooltip}>
            <Box
              role="listitem" tabIndex={0}
              sx={{
                display: "grid", gridTemplateColumns: { xs: `minmax(0,110px) 1fr 64px`, sm: `${labelWidth}px 1fr 72px` },
                alignItems: "center", gap: 1.5, py: 0.5, px: 0.75, mx: -0.75, borderRadius: 1.5,
                "&:hover, &:focus-visible": { bgcolor: viz.neutralTrack, outline: "none" },
              }}
            >
              <Box sx={{ minWidth: 0, display: "flex", alignItems: "center", gap: 0.75 }}>
                <Typography noWrap sx={{ fontSize: "0.875rem", fontWeight: r.highlight ? 600 : 400, color: viz.textPrimary }}>
                  {r.label}
                </Typography>
                {r.badge && (
                  <Box component="span" sx={{ flexShrink: 0, fontSize: "0.6875rem", fontWeight: 600, px: 0.75, py: 0.125, borderRadius: 1, bgcolor: "#e3eefb", color: "#184f95", letterSpacing: 0.2 }}>
                    {r.badge}
                  </Box>
                )}
              </Box>
              <Box sx={{ position: "relative", height: 20, bgcolor: viz.neutralTrack, borderRadius: "4px" }}>
                <Box
                  sx={{
                    position: "absolute", inset: 0, width: `${Math.max(pct, 1.5)}%`, borderRadius: "4px",
                    bgcolor: r.highlight ? viz.highlight : viz.muted, transition: "width 400ms ease",
                  }}
                />
              </Box>
              <Typography sx={{ fontSize: "0.875rem", fontVariantNumeric: "tabular-nums", textAlign: "right", color: viz.textPrimary, fontWeight: r.highlight ? 600 : 400 }}>
                {r.valueLabel ?? fmt(r.value)}
              </Typography>
            </Box>
          </Tooltip>
        );
      })}
    </Box>
  );
}

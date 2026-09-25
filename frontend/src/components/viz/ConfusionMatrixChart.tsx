import { Box, Typography } from "@mui/material";
import type { VisualizationData } from "../../types";
import ChartCard from "./ChartCard";
import { prettyAlgorithm, viz } from "./vizTheme";

/** Heatmap shaded by each row's share (recall per actual class) so a minority class is
 * still readable, with the raw count printed in every cell. */
export default function ConfusionMatrixChart({ data }: { data: NonNullable<VisualizationData["confusion_matrix"]> }) {
  const { labels, matrix } = data;
  const total = matrix.flat().reduce((a, b) => a + b, 0) || 1;
  const correct = matrix.reduce((acc, row, i) => acc + (row[i] ?? 0), 0);
  const cell = labels.length > 6 ? 56 : 84;
  const shade = (share: number) => viz.sequential[Math.min(viz.sequential.length - 1, Math.floor(share * viz.sequential.length))];

  return (
    <ChartCard
      title="Confusion matrix"
      subtitle={`${prettyAlgorithm(data.algorithm)} · ${correct} of ${total} test rows correct (${Math.round((correct / total) * 100)}%). Shade = share of each actual class.`}
    >
      <Box sx={{ overflowX: "auto", pb: 1 }}>
        <Box sx={{ display: "grid", gridTemplateColumns: `auto repeat(${labels.length}, minmax(${cell}px, 1fr))`, gap: "3px", minWidth: "fit-content" }}>
          <Box />
          {labels.map((l) => (
            <Typography key={`h-${l}`} noWrap title={`Predicted ${l}`} sx={{ fontSize: "0.8125rem", textAlign: "center", color: viz.textSecondary, px: 0.5 }}>
              {l}
            </Typography>
          ))}
          {matrix.map((row, i) => {
            const rowTotal = row.reduce((a, b) => a + b, 0) || 1;
            return [
              <Typography key={`r-${i}`} noWrap title={`Actual ${labels[i]}`} sx={{ fontSize: "0.8125rem", color: viz.textSecondary, pr: 1, alignSelf: "center", textAlign: "right" }}>
                {labels[i]}
              </Typography>,
              ...row.map((count, j) => {
                const share = count / rowTotal;
                const dark = share >= 0.45;
                return (
                  <Box
                    key={`c-${i}-${j}`} tabIndex={0}
                    title={`Actual ${labels[i]} → predicted ${labels[j]}: ${count} rows (${Math.round(share * 100)}% of actual ${labels[i]})`}
                    sx={{
                      height: cell * 0.8, borderRadius: 1.5, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center",
                      bgcolor: count === 0 ? viz.neutralTrack : shade(share), color: dark ? "#fff" : viz.textPrimary,
                      outline: i === j ? `2px solid ${viz.highlight}` : "none", outlineOffset: -2,
                      "&:hover, &:focus-visible": { boxShadow: "0 0 0 2px rgba(0,0,0,0.25)", outline: "none" },
                    }}
                  >
                    <Typography sx={{ fontSize: "1rem", fontWeight: 600, lineHeight: 1.1, color: "inherit" }}>{count}</Typography>
                    <Typography sx={{ fontSize: "0.6875rem", color: "inherit", opacity: 0.85 }}>{Math.round(share * 100)}%</Typography>
                  </Box>
                );
              }),
            ];
          })}
        </Box>
        <Typography sx={{ fontSize: "0.8125rem", color: viz.textMuted, mt: 1.5 }}>
          Rows: actual outcome · Columns: predicted · Outlined diagonal = correct predictions
        </Typography>
      </Box>
    </ChartCard>
  );
}

import { useState } from "react";
import { ToggleButton, ToggleButtonGroup, Typography } from "@mui/material";
import type { VisualizationData } from "../../types";
import ChartCard from "./ChartCard";
import HorizontalBars from "./HorizontalBars";
import { fmt, prettyAlgorithm, viz } from "./vizTheme";

export default function ModelComparisonChart({ data }: { data: NonNullable<VisualizationData["model_comparison"]> }) {
  const [metricKey, setMetricKey] = useState(data.metrics[0]?.key);
  const metric = data.metrics.find((m) => m.key === metricKey) ?? data.metrics[0];
  if (!metric) return null;

  // Best first for the chosen metric, so the bars always read as a ranking.
  const models = [...data.models].sort((a, b) => {
    const av = a.values[metric.key], bv = b.values[metric.key];
    if (av === null || av === undefined) return 1;
    if (bv === null || bv === undefined) return -1;
    return metric.higher_is_better ? bv - av : av - bv;
  });

  const rows = models.map((m) => ({
    key: m.run_id,
    label: prettyAlgorithm(m.algorithm),
    value: m.values[metric.key] ?? null,
    highlight: m.is_recommended,
    badge: m.is_recommended ? "Recommended" : undefined,
    tooltip: [
      `${prettyAlgorithm(m.algorithm)}${m.rank ? ` · rank #${m.rank}` : ""}`,
      ...data.metrics.map((x) => `${x.label}: ${fmt(m.values[x.key])}`),
      ...(m.n_clusters !== undefined ? [`Clusters: ${m.n_clusters}`] : []),
    ].join("\n"),
  }));

  return (
    <ChartCard
      title="Model comparison"
      subtitle={`${models.length} models evaluated · ${metric.higher_is_better ? "higher" : "lower"} is better`}
      span={2}
      action={
        data.metrics.length > 1 ? (
          <ToggleButtonGroup
            exclusive size="small" value={metric.key} aria-label="Comparison metric"
            onChange={(_, v) => v && setMetricKey(v)}
            sx={{ flexWrap: "wrap", "& .MuiToggleButton-root": { textTransform: "none", fontSize: "0.8125rem", px: 1.5, py: 0.5 } }}
          >
            {data.metrics.map((m) => (
              <ToggleButton key={m.key} value={m.key}>{m.label}</ToggleButton>
            ))}
          </ToggleButtonGroup>
        ) : undefined
      }
    >
      <HorizontalBars rows={rows} />
      <Typography sx={{ fontSize: "0.8125rem", color: viz.textMuted, mt: 2 }}>
        The recommended model is highlighted; hover a row for every metric.
      </Typography>
    </ChartCard>
  );
}

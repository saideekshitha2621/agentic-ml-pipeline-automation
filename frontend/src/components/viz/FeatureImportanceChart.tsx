import type { VisualizationData } from "../../types";
import ChartCard from "./ChartCard";
import HorizontalBars from "./HorizontalBars";
import { prettyAlgorithm } from "./vizTheme";

export default function FeatureImportanceChart({ data }: { data: NonNullable<VisualizationData["feature_importance"]> }) {
  const rows = data.features.map((f, i) => ({
    key: f.feature,
    label: f.feature,
    value: Math.abs(f.importance_pct),
    valueLabel: `${f.importance_pct}%`,
    highlight: i === 0,
    tooltip: `${f.feature}\nShare of total importance: ${f.importance_pct}%`,
  }));
  return (
    <ChartCard
      title="Top contributing features"
      subtitle={`${data.algorithm ? prettyAlgorithm(data.algorithm) + " · " : ""}top ${data.features.length} of ${data.total_features} features, as % of total importance`}
    >
      <HorizontalBars rows={rows} labelWidth={150} />
    </ChartCard>
  );
}

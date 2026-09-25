import { Alert, Box, Skeleton, Typography } from "@mui/material";
import type { VisualizationData } from "../../types";
import ClusterScatterChart from "./ClusterScatterChart";
import ConfusionMatrixChart from "./ConfusionMatrixChart";
import FeatureImportanceChart from "./FeatureImportanceChart";
import ModelComparisonChart from "./ModelComparisonChart";
import { prettyAlgorithm, viz } from "./vizTheme";

export default function InsightsSection({ data, loading }: { data?: VisualizationData; loading: boolean }) {
  if (loading) return <Skeleton variant="rounded" height={280} />;
  if (!data?.model_comparison) return null;

  const cluster = data.problem_type === "clustering";
  return (
    <Box component="section" aria-labelledby="insights-heading">
      <Box sx={{ mb: 2 }}>
        <Typography id="insights-heading" component="h2" sx={{ fontSize: "1.5rem", fontWeight: 600, color: viz.textPrimary }}>
          Model insights
        </Typography>
        <Typography sx={{ fontSize: "0.9375rem", color: viz.textSecondary, mt: 0.25 }}>
          {data.recommended_algorithm
            ? `How the ${data.model_comparison.models.length} candidate models compare, and why ${prettyAlgorithm(data.recommended_algorithm)} is recommended.`
            : "How the candidate models compare."}
        </Typography>
      </Box>
      <Box sx={{ display: "grid", gap: 3, gridTemplateColumns: { xs: "minmax(0,1fr)", lg: "repeat(2, minmax(0,1fr))" } }}>
        <ModelComparisonChart data={data.model_comparison} />
        {data.feature_importance && <FeatureImportanceChart data={data.feature_importance} />}
        {data.confusion_matrix && <ConfusionMatrixChart data={data.confusion_matrix} />}
        {cluster && data.cluster_plot && <ClusterScatterChart data={data.cluster_plot} />}
        {!data.feature_importance && !cluster && (
          <Alert severity="info" sx={{ borderRadius: 3 }}>Feature importance appears once the recommended model is approved and finalized.</Alert>
        )}
      </Box>
    </Box>
  );
}

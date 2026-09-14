import { useParams } from "react-router-dom";
import { Box, LinearProgress, Paper, Stack, Typography } from "@mui/material";
import Plot from "react-plotly.js";
import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { useInterpretation, useVisualizations } from "../api/visualizations";

export default function VisualizationPage() {
  const { jobId } = useParams();
  const { data: viz, isLoading: vizLoading } = useVisualizations(jobId);
  const { data: interpretation, isLoading: interpLoading } = useInterpretation(jobId);

  if (vizLoading || interpLoading || !viz) return <LinearProgress />;

  const clusters = Array.from(new Set(viz.scatter.map((p) => p.cluster)));
  const palette = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974", "#64B5CD", "#DD8452", "#937860"];

  const traces = clusters.map((c, i) => ({
    x: viz.scatter.filter((p) => p.cluster === c).map((p) => p.x),
    y: viz.scatter.filter((p) => p.cluster === c).map((p) => p.y),
    mode: "markers" as const,
    type: "scattergl" as const,
    name: c === "noise" ? "Noise" : `Cluster ${c}`,
    marker: { color: c === "noise" ? "#bbbbbb" : palette[i % palette.length], size: 6, opacity: 0.75 },
  }));

  return (
    <Stack spacing={3}>
      <Typography variant="h4">Visualizations</Typography>

      <Paper variant="outlined" sx={{ p: 2 }}>
        <Typography variant="h6" gutterBottom>
          Cluster Scatter (PCA-projected)
        </Typography>
        <Plot
          data={traces}
          layout={{ autosize: true, height: 480, margin: { t: 20 }, legend: { orientation: "h" } }}
          style={{ width: "100%" }}
          useResizeHandler
        />
      </Paper>

      <Stack direction="row" spacing={2} sx={{ flexWrap: "wrap" }}>
        <Paper variant="outlined" sx={{ p: 2, flex: 1, minWidth: 380 }}>
          <Typography variant="h6" gutterBottom>
            Cluster Size Distribution
          </Typography>
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={viz.cluster_sizes}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="cluster" />
              <YAxis />
              <Tooltip />
              <Bar dataKey="count" fill="#4C72B0" />
            </BarChart>
          </ResponsiveContainer>
        </Paper>

        <Paper variant="outlined" sx={{ p: 2, flex: 1, minWidth: 380 }}>
          <Typography variant="h6" gutterBottom>
            Metric Comparison Across Runs
          </Typography>
          <ResponsiveContainer width="100%" height={280}>
            <BarChart data={viz.metric_comparison}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="algorithm" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Bar dataKey="silhouette_score" name="Silhouette" fill="#55A868" />
              <Bar dataKey="davies_bouldin_score" name="Davies-Bouldin" fill="#C44E52" />
            </BarChart>
          </ResponsiveContainer>
        </Paper>
      </Stack>

      {interpretation && (
        <Paper variant="outlined" sx={{ p: 2 }}>
          <Typography variant="h6" gutterBottom>
            Cluster Summaries
          </Typography>
          <Stack spacing={1.5}>
            {Object.entries(interpretation.summaries).map(([clusterId, summary]) => (
              <Box key={clusterId}>
                <Typography variant="subtitle2">
                  Cluster {clusterId} — "{interpretation.suggested_names[clusterId]}"
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  {summary}
                </Typography>
              </Box>
            ))}
          </Stack>
        </Paper>
      )}
    </Stack>
  );
}

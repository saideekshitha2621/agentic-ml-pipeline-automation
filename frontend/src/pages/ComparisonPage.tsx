import { useSearchParams, useParams, useNavigate } from "react-router-dom";
import { Box, Button, LinearProgress, Paper, Stack, Typography } from "@mui/material";
import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { useCompareRuns } from "../api/leaderboard";

export default function ComparisonPage() {
  const { jobId } = useParams();
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const runIds = (searchParams.get("runs") ?? "").split(",").filter(Boolean);
  const { data: runs, isLoading } = useCompareRuns(jobId, runIds);

  if (runIds.length === 0) {
    return (
      <Stack spacing={2}>
        <Typography variant="h4">Model Comparison</Typography>
        <Typography color="text.secondary">
          No models selected. Go back to the Leaderboard and select at least two models to compare.
        </Typography>
        <Button onClick={() => navigate(`/jobs/${jobId}/leaderboard`)}>← Back to Leaderboard</Button>
      </Stack>
    );
  }

  if (isLoading || !runs) return <LinearProgress />;

  const chartData = runs.map((r) => ({
    name: `${r.algorithm}\n#${r.rank ?? "-"}`,
    Silhouette: r.silhouette_score ?? 0,
    "Davies-Bouldin": r.davies_bouldin_score ?? 0,
  }));

  return (
    <Stack spacing={3}>
      <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <Typography variant="h4">Model Comparison</Typography>
        <Button onClick={() => navigate(`/jobs/${jobId}/leaderboard`)}>← Back to Leaderboard</Button>
      </Box>

      <Paper variant="outlined" sx={{ p: 2 }}>
        <ResponsiveContainer width="100%" height={320}>
          <BarChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" />
            <YAxis />
            <Tooltip />
            <Legend />
            <Bar dataKey="Silhouette" fill="#4C72B0" />
            <Bar dataKey="Davies-Bouldin" fill="#C44E52" />
          </BarChart>
        </ResponsiveContainer>
      </Paper>

      <Stack direction="row" spacing={2} sx={{ flexWrap: "wrap" }}>
        {runs.map((r) => (
          <Paper key={r.id} variant="outlined" sx={{ p: 2, minWidth: 260, flex: 1 }}>
            <Typography variant="subtitle1">
              {r.algorithm} (rank #{r.rank ?? "—"})
            </Typography>
            <Typography variant="body2" sx={{ fontFamily: "monospace" }}>
              {JSON.stringify(r.params_json)}
            </Typography>
            <Typography variant="body2">Clusters: {r.n_clusters} | Noise: {r.n_noise}</Typography>
            <Typography variant="body2">Silhouette: {r.silhouette_score?.toFixed(3) ?? "—"}</Typography>
            <Typography variant="body2">Davies-Bouldin: {r.davies_bouldin_score?.toFixed(3) ?? "—"}</Typography>
            <Typography variant="body2">
              Calinski-Harabasz: {r.calinski_harabasz_score?.toFixed(1) ?? "—"}
            </Typography>
          </Paper>
        ))}
      </Stack>
    </Stack>
  );
}

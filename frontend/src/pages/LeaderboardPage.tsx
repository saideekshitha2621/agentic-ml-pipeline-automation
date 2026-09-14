import { useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import {
  Alert,
  Box,
  Button,
  Checkbox,
  LinearProgress,
  Paper,
  Stack,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableRow,
  Typography,
} from "@mui/material";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { useLeaderboard } from "../api/leaderboard";

export default function LeaderboardPage() {
  const { jobId } = useParams();
  const navigate = useNavigate();
  const { data: leaderboard, isLoading } = useLeaderboard(jobId);
  const [selected, setSelected] = useState<string[]>([]);

  if (isLoading || !leaderboard) return <LinearProgress />;

  const toggle = (id: string) =>
    setSelected((prev) => (prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id]));

  const kmeansByK = leaderboard
    .filter((r) => r.algorithm === "kmeans")
    .map((r) => ({
      k: (r.params_json.n_clusters as number) ?? 0,
      inertia: (r.extra_json.inertia as number) ?? null,
      silhouette: r.silhouette_score,
    }))
    .sort((a, b) => a.k - b.k);

  return (
    <Stack spacing={3}>
      <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <Typography variant="h4">Model Leaderboard</Typography>
        <Stack direction="row" spacing={2}>
          <Button
            disabled={selected.length < 2}
            onClick={() => navigate(`/jobs/${jobId}/comparison?runs=${selected.join(",")}`)}
          >
            Compare Selected ({selected.length})
          </Button>
          <Button variant="contained" onClick={() => navigate(`/jobs/${jobId}/approval`)}>
            Continue to HITL Approval →
          </Button>
        </Stack>
      </Box>

      <Alert severity="info">
        Scores are relative to this run's other models, not an absolute quality scale. Runs where
        more than half the data was classified as noise are excluded from ranking entirely — their
        metrics only reflect the minority of points that weren't discarded, which can look
        deceptively good.
      </Alert>

      {kmeansByK.length > 1 && (
        <Paper variant="outlined" sx={{ p: 2 }}>
          <Typography variant="h6" gutterBottom>
            Cluster Count (K) Selection — KMeans Elbow &amp; Silhouette
          </Typography>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
            The elbow curve (inertia — within-cluster sum of squares) helps spot where adding
            more clusters stops meaningfully improving compactness; silhouette confirms which K
            actually separates clusters best. Look for where the inertia curve "bends" and cross-
            check it against a silhouette peak.
          </Typography>
          <ResponsiveContainer width="100%" height={280}>
            <LineChart data={kmeansByK}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="k" label={{ value: "K (number of clusters)", position: "insideBottom", offset: -5 }} />
              <YAxis yAxisId="left" label={{ value: "Inertia", angle: -90, position: "insideLeft" }} />
              <YAxis
                yAxisId="right"
                orientation="right"
                domain={[0, 1]}
                label={{ value: "Silhouette", angle: 90, position: "insideRight" }}
              />
              <Tooltip />
              <Legend />
              <Line yAxisId="left" type="monotone" dataKey="inertia" name="Inertia (Elbow)" stroke="#4C72B0" />
              <Line
                yAxisId="right"
                type="monotone"
                dataKey="silhouette"
                name="Silhouette"
                stroke="#55A868"
              />
            </LineChart>
          </ResponsiveContainer>
        </Paper>
      )}

      <Paper variant="outlined">
        <Table size="small">
          <TableHead>
            <TableRow>
              <TableCell padding="checkbox" />
              <TableCell>Algorithm</TableCell>
              <TableCell>Parameters</TableCell>
              <TableCell align="right">Clusters</TableCell>
              <TableCell align="right">Noise</TableCell>
              <TableCell align="right">Silhouette</TableCell>
              <TableCell align="right">Davies-Bouldin</TableCell>
              <TableCell align="right">Calinski-Harabasz</TableCell>
              <TableCell align="right">Rank</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {leaderboard.map((run) => (
              <TableRow key={run.id} hover selected={selected.includes(run.id)}>
                <TableCell padding="checkbox">
                  <Checkbox checked={selected.includes(run.id)} onChange={() => toggle(run.id)} />
                </TableCell>
                <TableCell>{run.algorithm}</TableCell>
                <TableCell sx={{ fontFamily: "monospace", fontSize: 12 }}>
                  {JSON.stringify(run.params_json)}
                </TableCell>
                <TableCell align="right">{run.n_clusters}</TableCell>
                <TableCell
                  align="right"
                  sx={run.noise_pct >= 30 ? { color: "error.main", fontWeight: 600 } : undefined}
                >
                  {run.n_noise} {run.noise_pct > 0 ? `(${run.noise_pct.toFixed(0)}%)` : ""}
                </TableCell>
                <TableCell align="right">{run.silhouette_score?.toFixed(3) ?? "—"}</TableCell>
                <TableCell align="right">{run.davies_bouldin_score?.toFixed(3) ?? "—"}</TableCell>
                <TableCell align="right">{run.calinski_harabasz_score?.toFixed(1) ?? "—"}</TableCell>
                <TableCell align="right">{run.rank ?? (run.noise_pct >= 50 ? "excluded" : "—")}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </Paper>
    </Stack>
  );
}

import { useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import {
  Alert,
  Box,
  Button,
  Chip,
  LinearProgress,
  Paper,
  Stack,
  TextField,
  Typography,
} from "@mui/material";
import { CheckCircle, Warning } from "@mui/icons-material";
import { useApproveModel, useRecommendations } from "../api/approval";

export default function ApprovalPage() {
  const { jobId } = useParams();
  const navigate = useNavigate();
  const { data: recommendations, isLoading } = useRecommendations(jobId);
  const approve = useApproveModel(jobId);
  const [approvedBy, setApprovedBy] = useState("");
  const [notes, setNotes] = useState("");
  const [approvedRunId, setApprovedRunId] = useState<string | null>(null);

  if (isLoading) return <LinearProgress />;
  if (!recommendations || recommendations.length === 0) {
    return <Alert severity="warning">No recommendations available yet.</Alert>;
  }

  const handleApprove = (runId: string) => {
    approve.mutate(
      { cluster_run_id: runId, approved_by: approvedBy || "unspecified user", notes: notes || undefined },
      { onSuccess: () => setApprovedRunId(runId) },
    );
  };

  return (
    <Stack spacing={3}>
      <Typography variant="h4">Human-in-the-Loop Approval</Typography>
      <Typography color="text.secondary">
        The platform never picks the final model automatically. Review the top 3 recommendations below and
        approve the one you trust.
      </Typography>

      <Stack direction="row" spacing={2}>
        <TextField
          label="Approved by"
          size="small"
          value={approvedBy}
          onChange={(e) => setApprovedBy(e.target.value)}
          sx={{ minWidth: 260 }}
        />
        <TextField
          label="Notes (optional)"
          size="small"
          value={notes}
          onChange={(e) => setNotes(e.target.value)}
          sx={{ minWidth: 320 }}
        />
      </Stack>

      <Stack spacing={2}>
        {recommendations.map((rec) => (
          <Paper key={rec.cluster_run.id} variant="outlined" sx={{ p: 3 }}>
            <Stack direction="row" sx={{ justifyContent: "space-between", alignItems: "flex-start" }}>
              <Box>
                <Typography variant="h6">
                  #{rec.cluster_run.rank} — {rec.cluster_run.algorithm}
                </Typography>
                <Typography variant="body2" sx={{ fontFamily: "monospace" }} color="text.secondary">
                  {JSON.stringify(rec.cluster_run.params_json)}
                </Typography>
              </Box>
              <Button
                variant={approvedRunId === rec.cluster_run.id ? "outlined" : "contained"}
                color={approvedRunId === rec.cluster_run.id ? "success" : "primary"}
                disabled={approve.isPending || !!approvedRunId}
                onClick={() => handleApprove(rec.cluster_run.id)}
              >
                {approvedRunId === rec.cluster_run.id ? "Approved ✓" : "Approve this model"}
              </Button>
            </Stack>

            <Typography sx={{ mt: 2 }}>{rec.rationale}</Typography>

            <Stack direction="row" spacing={4} sx={{ mt: 2 }}>
              <Box>
                <Typography variant="subtitle2">Strengths</Typography>
                {rec.strengths.map((s, i) => (
                  <Stack direction="row" spacing={1} key={i} sx={{ alignItems: "center" }}>
                    <CheckCircle fontSize="small" color="success" />
                    <Typography variant="body2">{s}</Typography>
                  </Stack>
                ))}
              </Box>
              {rec.weaknesses.length > 0 && (
                <Box>
                  <Typography variant="subtitle2">Weaknesses</Typography>
                  {rec.weaknesses.map((w, i) => (
                    <Stack direction="row" spacing={1} key={i} sx={{ alignItems: "center" }}>
                      <Warning fontSize="small" color="warning" />
                      <Typography variant="body2">{w}</Typography>
                    </Stack>
                  ))}
                </Box>
              )}
            </Stack>

            <Stack direction="row" spacing={1} sx={{ mt: 2, flexWrap: "wrap" }}>
              {Object.entries(rec.cluster_size_breakdown).map(([cluster, count]) => (
                <Chip key={cluster} size="small" label={`cluster ${cluster}: ${count}`} />
              ))}
            </Stack>
          </Paper>
        ))}
      </Stack>

      {approve.isError && <Alert severity="error">Failed to approve model.</Alert>}
      {approvedRunId && (
        <Button variant="contained" onClick={() => navigate(`/jobs/${jobId}/visualization`)}>
          Continue to Visualizations →
        </Button>
      )}
    </Stack>
  );
}

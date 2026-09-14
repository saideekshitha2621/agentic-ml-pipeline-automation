import { useEffect, useRef } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { Alert, Box, Button, LinearProgress, Paper, Stack, Typography } from "@mui/material";
import { useCreateJob, useJob } from "../api/jobs";
import { usePreprocessingPlan } from "../api/preprocessing";
import { useWorkspace } from "../components/WorkspaceContext";

const ALGORITHMS = ["KMeans", "DBSCAN", "Hierarchical", "GMM", "Spectral", "Birch", "OPTICS"];

export default function ModelExecutionPage() {
  const { datasetId } = useParams();
  const navigate = useNavigate();
  const { jobId, setJobId } = useWorkspace();
  const { data: plan } = usePreprocessingPlan(datasetId);
  const createJob = useCreateJob();
  const { data: job } = useJob(jobId ?? undefined);
  const logRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    logRef.current?.scrollTo({ top: logRef.current.scrollHeight });
  }, [job?.log_lines]);

  const startJob = () => {
    if (!datasetId || !plan) return;
    createJob.mutate(
      { dataset_id: datasetId, preprocessing_plan_id: plan.id },
      { onSuccess: (newJob) => setJobId(newJob.id) },
    );
  };

  return (
    <Stack spacing={3} sx={{ maxWidth: 900 }}>
      <Typography variant="h4">Model Execution</Typography>

      <Paper variant="outlined" sx={{ p: 2 }}>
        <Typography variant="subtitle2" gutterBottom>
          Algorithms in this run
        </Typography>
        <Stack direction="row" spacing={1} sx={{ flexWrap: "wrap" }}>
          {ALGORITHMS.map((a) => (
            <Box
              key={a}
              sx={{ px: 1.5, py: 0.5, border: 1, borderColor: "divider", borderRadius: 4, fontSize: 13 }}
            >
              {a}
            </Box>
          ))}
        </Stack>
      </Paper>

      {!job && (
        <Button variant="contained" onClick={startJob} disabled={createJob.isPending || !plan}>
          {createJob.isPending ? "Starting..." : "Run Pipeline"}
        </Button>
      )}

      {job && (
        <Paper variant="outlined" sx={{ p: 2 }}>
          <Stack direction="row" sx={{ justifyContent: "space-between", alignItems: "center", mb: 1 }}>
            <Typography variant="subtitle1">Status: {job.status}</Typography>
            <Typography variant="body2" color="text.secondary">
              {job.progress_pct.toFixed(0)}%
            </Typography>
          </Stack>
          <LinearProgress
            variant="determinate"
            value={job.progress_pct}
            color={job.status === "failed" ? "error" : "primary"}
          />
          <Box
            ref={logRef}
            sx={{
              mt: 2,
              p: 1.5,
              bgcolor: "grey.900",
              color: "grey.100",
              fontFamily: "monospace",
              fontSize: 12,
              height: 220,
              overflowY: "auto",
              borderRadius: 1,
            }}
          >
            {job.log_lines.map((line, i) => (
              <div key={i}>{line}</div>
            ))}
          </Box>

          {job.status === "failed" && <Alert severity="error" sx={{ mt: 2 }}>{job.error_message}</Alert>}

          {job.status === "completed" && (
            <Button
              variant="contained"
              sx={{ mt: 2 }}
              onClick={() => navigate(`/jobs/${job.id}/leaderboard`)}
            >
              View Leaderboard →
            </Button>
          )}
        </Paper>
      )}
    </Stack>
  );
}

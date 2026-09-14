import { useState } from "react";
import { useParams } from "react-router-dom";
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
import { CheckCircle, Edit, HourglassEmpty, Cancel, SmartToy } from "@mui/icons-material";
import {
  pipelineReportExportUrl,
  usePipelineDecisions,
  usePipelineReport,
  usePipelineRun,
  useReviewDecision,
} from "../api/pipeline";
import type { AgentDecision } from "../types";

const STATUS_COLOR: Record<string, "default" | "success" | "warning" | "error" | "info"> = {
  proposed: "warning",
  approved: "success",
  edited: "info",
  rejected: "error",
};

function StatusIcon({ status }: { status: string }) {
  if (status === "approved") return <CheckCircle fontSize="small" color="success" />;
  if (status === "edited") return <Edit fontSize="small" color="info" />;
  if (status === "rejected") return <Cancel fontSize="small" color="error" />;
  return <HourglassEmpty fontSize="small" color="warning" />;
}

function DecisionReviewCard({ decision, pipelineRunId }: { decision: AgentDecision; pipelineRunId: string }) {
  const review = useReviewDecision(pipelineRunId);
  const [reviewedBy, setReviewedBy] = useState("");
  const [reason, setReason] = useState("");
  const [showReject, setShowReject] = useState(false);

  const isProblemDetection = decision.agent_name === "problem_detection";
  const proposedType = decision.decision_json.problem_type as string | undefined;
  const [overrideType, setOverrideType] = useState(proposedType ?? "clustering");

  return (
    <Paper variant="outlined" sx={{ p: 3, borderColor: "warning.main" }}>
      <Stack direction="row" spacing={1} sx={{ alignItems: "center", mb: 1 }}>
        <SmartToy fontSize="small" />
        <Typography variant="h6">
          {decision.agent_name.replace(/_/g, " ")} — awaiting your review
        </Typography>
        {decision.confidence !== null && (
          <Chip
            size="small"
            label={`confidence ${Math.round(decision.confidence * 100)}%`}
            color={decision.confidence >= 0.85 ? "success" : decision.confidence >= 0.6 ? "warning" : "error"}
          />
        )}
      </Stack>
      <Typography color="text.secondary" sx={{ mb: 2 }}>
        {decision.reasoning_text}
      </Typography>

      <TextField
        label="Your name"
        size="small"
        value={reviewedBy}
        onChange={(e) => setReviewedBy(e.target.value)}
        sx={{ mb: 2, minWidth: 260 }}
      />

      {isProblemDetection && (
        <Stack direction="row" spacing={1} sx={{ mb: 2, alignItems: "center" }}>
          <Typography variant="body2">Override problem type:</Typography>
          {["clustering", "classification", "regression"].map((t) => (
            <Chip
              key={t}
              label={t}
              size="small"
              color={overrideType === t ? "primary" : "default"}
              onClick={() => setOverrideType(t)}
            />
          ))}
        </Stack>
      )}

      <Stack direction="row" spacing={2}>
        <Button
          variant="contained"
          color="success"
          disabled={review.isPending || !reviewedBy}
          onClick={() =>
            review.mutate({ decisionId: decision.id, action: "approve", reviewed_by: reviewedBy })
          }
        >
          Approve as proposed
        </Button>
        {isProblemDetection && (
          <Button
            variant="outlined"
            disabled={review.isPending || !reviewedBy || overrideType === proposedType}
            onClick={() =>
              review.mutate({
                decisionId: decision.id,
                action: "edit",
                edits: { problem_type: overrideType, target_column: overrideType === "clustering" ? null : decision.decision_json.target_column },
                reviewed_by: reviewedBy,
              })
            }
          >
            Override to "{overrideType}"
          </Button>
        )}
        <Button color="error" disabled={review.isPending || !reviewedBy} onClick={() => setShowReject((s) => !s)}>
          Reject
        </Button>
      </Stack>

      {showReject && (
        <Stack direction="row" spacing={2} sx={{ mt: 2 }}>
          <TextField
            label="Reason for rejection"
            size="small"
            fullWidth
            value={reason}
            onChange={(e) => setReason(e.target.value)}
          />
          <Button
            color="error"
            variant="contained"
            disabled={review.isPending || !reviewedBy || !reason}
            onClick={() =>
              review.mutate({ decisionId: decision.id, action: "reject", reason, reviewed_by: reviewedBy })
            }
          >
            Confirm reject
          </Button>
        </Stack>
      )}

      {review.isError && <Alert severity="error" sx={{ mt: 2 }}>Failed to submit review.</Alert>}
    </Paper>
  );
}

export default function PipelineRunPage() {
  const { pipelineRunId } = useParams();
  const { data: run } = usePipelineRun(pipelineRunId);
  const { data: decisions, isLoading } = usePipelineDecisions(pipelineRunId);
  const { data: report } = usePipelineReport(pipelineRunId, run?.status === "completed");

  if (isLoading || !run) return <LinearProgress />;

  const pendingDecision = decisions?.find((d) => d.status === "proposed");

  return (
    <Stack spacing={3}>
      <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <Typography variant="h4">Agentic Pipeline Run</Typography>
        <Chip
          label={run.status.replace(/_/g, " ")}
          color={run.status === "completed" ? "success" : run.status === "failed" ? "error" : "default"}
        />
      </Box>

      {(run.status === "profiling" || run.status === "model_execution" || run.status === "reporting") && (
        <LinearProgress />
      )}

      {run.status === "failed" && <Alert severity="error">{run.error_message}</Alert>}

      {pendingDecision && <DecisionReviewCard decision={pendingDecision} pipelineRunId={run.id} />}

      <Typography variant="h6">Agent Activity Timeline</Typography>
      <Stack spacing={1.5}>
        {decisions?.map((d) => (
          <Paper key={d.id} variant="outlined" sx={{ p: 2 }}>
            <Stack direction="row" spacing={1.5} sx={{ alignItems: "flex-start" }}>
              <StatusIcon status={d.status} />
              <Box sx={{ flexGrow: 1 }}>
                <Stack direction="row" spacing={1} sx={{ alignItems: "center" }}>
                  <Typography variant="subtitle2">{d.agent_name.replace(/_/g, " ")}</Typography>
                  <Chip size="small" label={d.status} color={STATUS_COLOR[d.status]} />
                  {d.confidence !== null && (
                    <Typography variant="caption" color="text.secondary">
                      confidence {Math.round(d.confidence * 100)}%
                    </Typography>
                  )}
                  <Typography variant="caption" color="text.secondary" sx={{ ml: "auto" }}>
                    {new Date(d.created_at).toLocaleTimeString()}
                  </Typography>
                </Stack>
                <Typography variant="body2" sx={{ mt: 0.5 }}>
                  {d.reasoning_text}
                </Typography>
                {d.override_reason && (
                  <Typography variant="body2" color="error.main" sx={{ mt: 0.5 }}>
                    Override reason: {d.override_reason}
                  </Typography>
                )}
                {d.approved_by && (
                  <Typography variant="caption" color="text.secondary">
                    {d.status} by {d.approved_by}
                  </Typography>
                )}
              </Box>
            </Stack>
          </Paper>
        ))}
      </Stack>

      {run.status === "completed" && report && (
        <Paper variant="outlined" sx={{ p: 3 }}>
          <Stack direction="row" sx={{ justifyContent: "space-between", alignItems: "center" }}>
            <Typography variant="h6">Final Report</Typography>
            <Button variant="contained" href={pipelineReportExportUrl(run.id)} target="_blank" rel="noreferrer">
              Download PDF
            </Button>
          </Stack>
          <Typography sx={{ mt: 2 }}>
            <b>Problem type:</b> {report.analysis_summary.problem_type}
          </Typography>
          {report.recommendation_details && (
            <Typography sx={{ mt: 1 }}>
              <b>Recommended model:</b> {report.recommendation_details.top_choice.algorithm} —{" "}
              {report.recommendation_details.top_choice.rationale}
            </Typography>
          )}
        </Paper>
      )}
    </Stack>
  );
}

import { useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import {
  Alert,
  Box,
  Button,
  Checkbox,
  Chip,
  FormControlLabel,
  LinearProgress,
  MenuItem,
  Paper,
  Select,
  Slider,
  Stack,
  Step,
  StepLabel,
  Stepper,
  TextField,
  Typography,
} from "@mui/material";
import { CheckCircle, Edit, HourglassEmpty, Cancel, SmartToy, Science } from "@mui/icons-material";
import {
  pipelineReportExportUrl,
  usePipelineDecisions,
  usePipelineReport,
  usePipelineRun,
  useReviewDecision,
  useTrainingProgress,
} from "../api/pipeline";
import type { AgentDecision, PipelineRunStatus } from "../types";
import { PIPELINE_STAGES } from "../types";
import {
  AlgorithmShortlistContent,
  CleaningPlanContent,
  DatasetUnderstandingContent,
  EvaluationContent,
  HPOContent,
  SplitContent,
  TrainingProgressContent,
  TransformationContent,
  ValidationContent,
} from "../components/pipeline/StageContent";
import ChatPanel from "../components/pipeline/ChatPanel";

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

function StageSummary({ decision }: { decision: AgentDecision }) {
  switch (decision.agent_name) {
    case "data_profiling":
      return <DatasetUnderstandingContent decision={decision} />;
    case "data_validation":
      return <ValidationContent decision={decision} />;
    case "cleaning_plan":
      return <CleaningPlanContent decision={decision} />;
    case "transformation":
      return <TransformationContent decision={decision} />;
    case "train_test_split":
      return <SplitContent decision={decision} />;
    case "algorithm_recommendation":
      return <AlgorithmShortlistContent decision={decision} />;
    case "hyperparameter_optimization":
      return <HPOContent decision={decision} />;
    case "evaluation":
      return <EvaluationContent decision={decision} />;
    default:
      return <Typography variant="body2">{decision.reasoning_text}</Typography>;
  }
}

interface CleaningRecommendation {
  column: string;
  column_type: "numeric" | "categorical";
  issue: string;
  action: string;
  reason: string;
  missing_count: number;
  missing_pct: number;
  no_information?: boolean;
  options?: string[];
  custom_value?: string;
}

const NEEDS_CUSTOM_VALUE = new Set(["fill_custom", "business_rule"]);

function CleaningPlanEditForm({ decision, onChange }: { decision: AgentDecision; onChange: (edits: Record<string, unknown>) => void }) {
  const recs = (decision.decision_json.recommendations ?? []) as CleaningRecommendation[];
  const [rows, setRows] = useState<CleaningRecommendation[]>(recs);

  const update = (i: number, patch: Partial<CleaningRecommendation>) => {
    const next = rows.map((r, idx) => (idx === i ? { ...r, ...patch } : r));
    setRows(next);
    onChange({ ...decision.decision_json, recommendations: next });
  };

  return (
    <Stack spacing={1} sx={{ mb: 2 }}>
      {rows.map((r, i) => {
        const actionChoices = r.options ?? (r.column_type === "numeric" ? ["mean", "median", "drop_rows", "drop_column", "fill_zero", "fill_custom", "keep"] : ["mode", "drop_rows", "drop_column", "fill_zero", "fill_custom", "keep"]);
        return (
          <Stack key={r.column} direction="row" spacing={1} sx={{ alignItems: "center" }}>
            <Typography variant="body2" sx={{ minWidth: 140 }}>{r.column}</Typography>
            <Typography variant="caption" color="text.secondary" sx={{ minWidth: 90 }}>
              {r.missing_count} missing ({r.missing_pct}%)
            </Typography>
            <Select size="small" value={r.action} onChange={(e) => update(i, { action: e.target.value })}>
              {actionChoices.map((a) => (
                <MenuItem key={a} value={a}>{a.replace(/_/g, " ")}</MenuItem>
              ))}
            </Select>
            {NEEDS_CUSTOM_VALUE.has(r.action) && (
              <TextField
                size="small"
                label="Custom value"
                value={r.custom_value ?? ""}
                onChange={(e) => update(i, { custom_value: e.target.value })}
              />
            )}
            {r.no_information && <Chip size="small" color="error" label="no information available" />}
            <Typography variant="caption" color="text.secondary">{r.reason}</Typography>
          </Stack>
        );
      })}
    </Stack>
  );
}

function SplitEditForm({ decision, onChange }: { decision: AgentDecision; onChange: (edits: Record<string, unknown>) => void }) {
  const s = decision.decision_json as { test_size?: number; stratify?: boolean };
  const [testSize, setTestSize] = useState(Math.round((s.test_size ?? 0.2) * 100));

  return (
    <Box sx={{ mb: 2, maxWidth: 400 }}>
      <Typography variant="body2" gutterBottom>Test set size: {testSize}%</Typography>
      <Slider
        min={10} max={40} value={testSize}
        onChange={(_e, v) => {
          const pct = v as number;
          setTestSize(pct);
          onChange({ ...decision.decision_json, test_size: pct / 100, train_size: 1 - pct / 100 });
        }}
      />
    </Box>
  );
}

function AlgorithmShortlistEditForm({ decision, onChange }: { decision: AgentDecision; onChange: (edits: Record<string, unknown>) => void }) {
  const shortlist = (decision.decision_json.shortlist ?? []) as { algorithm: string; recommended: boolean; rationale: string }[];
  const [selected, setSelected] = useState<Set<string>>(new Set(shortlist.filter((s) => s.recommended).map((s) => s.algorithm)));

  const toggle = (name: string) => {
    const next = new Set(selected);
    next.has(name) ? next.delete(name) : next.add(name);
    setSelected(next);
    onChange({ ...decision.decision_json, selected_algorithms: Array.from(next) });
  };

  return (
    <Stack sx={{ mb: 2 }}>
      {shortlist.map((s) => (
        <FormControlLabel
          key={s.algorithm}
          control={<Checkbox checked={selected.has(s.algorithm)} onChange={() => toggle(s.algorithm)} />}
          label={`${s.algorithm.replace(/_/g, " ")} — ${s.rationale}`}
        />
      ))}
    </Stack>
  );
}

function DecisionReviewCard({ decision, pipelineRunId }: { decision: AgentDecision; pipelineRunId: string }) {
  const review = useReviewDecision(pipelineRunId);
  const [reviewedBy, setReviewedBy] = useState("");
  const [reason, setReason] = useState("");
  const [showReject, setShowReject] = useState(false);
  const [pendingEdits, setPendingEdits] = useState<Record<string, unknown> | null>(null);

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

      <Box sx={{ mb: 2 }}>
        <StageSummary decision={decision} />
      </Box>

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
      {decision.agent_name === "cleaning_plan" && <CleaningPlanEditForm decision={decision} onChange={setPendingEdits} />}
      {decision.agent_name === "train_test_split" && <SplitEditForm decision={decision} onChange={setPendingEdits} />}
      {decision.agent_name === "algorithm_recommendation" && (
        <AlgorithmShortlistEditForm decision={decision} onChange={setPendingEdits} />
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
        {pendingEdits && (
          <Button
            variant="outlined"
            disabled={review.isPending || !reviewedBy}
            onClick={() => review.mutate({ decisionId: decision.id, action: "edit", edits: pendingEdits, reviewed_by: reviewedBy })}
          >
            Save changes &amp; approve
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

function stageIndex(status: PipelineRunStatus): number {
  const idx = PIPELINE_STAGES.findIndex((s) => s.status === status || s.gate === status);
  return idx === -1 ? PIPELINE_STAGES.length : idx;
}

export default function PipelineRunPage() {
  const { pipelineRunId } = useParams();
  const navigate = useNavigate();
  const { data: run } = usePipelineRun(pipelineRunId);
  const { data: decisions, isLoading } = usePipelineDecisions(pipelineRunId);
  const { data: report } = usePipelineReport(pipelineRunId, run?.status === "completed");
  const { data: trainingProgress } = useTrainingProgress(pipelineRunId, run?.status === "training");

  if (isLoading || !run) return <LinearProgress />;

  const pendingDecision = decisions?.find((d) => d.status === "proposed");
  const activeIndex = stageIndex(run.status);

  const decisionForAgent = (agentName: string) =>
    decisions?.filter((d) => d.agent_name === agentName).slice(-1)[0];

  return (
    <Stack spacing={3}>
      <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <Typography variant="h4">Agentic Pipeline Run</Typography>
        <Stack direction="row" spacing={1}>
          {run.champion_model_path && (
            <Button startIcon={<Science />} variant="outlined" onClick={() => navigate(`/pipeline-runs/${run.id}/predict`)}>
              Prediction Playground
            </Button>
          )}
          <Chip
            label={run.status.replace(/_/g, " ")}
            color={run.status === "completed" ? "success" : run.status === "failed" ? "error" : "default"}
          />
        </Stack>
      </Box>

      {run.status === "failed" && <Alert severity="error">{run.error_message}</Alert>}

      <Stepper activeStep={activeIndex} orientation="vertical">
        {PIPELINE_STAGES.map((stage, i) => {
          const decision = decisionForAgent(stage.agentName);
          const isActive = i === activeIndex;
          return (
            <Step key={stage.status} completed={i < activeIndex || run.status === "completed"}>
              <StepLabel error={run.status === "failed" && isActive}>{stage.label}</StepLabel>
              <Box sx={{ pl: 2, pb: 2 }}>
                {stage.status === "training" && isActive && run.status === "training" && (
                  <TrainingProgressContent progress={trainingProgress} />
                )}
                {isActive && run.status !== "training" && !decision && <LinearProgress sx={{ maxWidth: 300 }} />}
                {decision && decision.status === "proposed" ? (
                  <DecisionReviewCard decision={decision} pipelineRunId={run.id} />
                ) : (
                  decision && <StageSummary decision={decision} />
                )}
              </Box>
            </Step>
          );
        })}
      </Stepper>

      {pendingDecision && !PIPELINE_STAGES.some((s) => s.agentName === pendingDecision.agent_name) && (
        <DecisionReviewCard decision={pendingDecision} pipelineRunId={run.id} />
      )}

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

      <ChatPanel pipelineRunId={run.id} />
    </Stack>
  );
}

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
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableRow,
  TextField,
  Tooltip,
  Typography,
} from "@mui/material";
import { SmartToy, Science, Person, Bolt } from "@mui/icons-material";
import {
  pipelineReportExportUrl,
  useExecutiveSummary,
  usePipelineDecisions,
  useVisualizations,
  usePipelineReport,
  usePipelineRun,
  useReviewDecision,
  useTrainingProgress,
} from "../api/pipeline";
import type { AgentDecision, PipelineRunStatus } from "../types";
import { PIPELINE_STAGES, modelLevelColor } from "../types";
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
  ConfusionMatrixTable,
  isConfusionMatrix,
} from "../components/pipeline/StageContent";
import ChatPanel from "../components/pipeline/ChatPanel";
import InsightsSection from "../components/viz/InsightsSection";
import ExecutiveSummaryCard from "../components/pipeline/ExecutiveSummaryCard";

/** Clearly distinguishes stages the agent auto-approved (approved_by === "system") from
 * ones a human actually reviewed and approved/edited — the core UI requirement for the
 * confidence-based auto-approval policy in services/approval_policy_service.py. */
function ApprovalBadge({ decision }: { decision: AgentDecision }) {
  if (decision.status === "proposed") return null;
  const reason = decision.decision_json.approval_reason as string | undefined;
  if (decision.status === "rejected") {
    return <Chip size="small" color="error" label={`Rejected by ${decision.approved_by}`} />;
  }
  if (decision.approved_by === "system") {
    return (
      <Tooltip title={reason ?? ""}>
        <Chip size="small" icon={<Bolt fontSize="small" />} color="info" variant="outlined" label="Auto Approved" />
      </Tooltip>
    );
  }
  if (decision.approved_by) {
    return (
      <Tooltip title={reason ?? ""}>
        <Chip
          size="small"
          icon={<Person fontSize="small" />}
          color="success"
          label={decision.status === "edited" ? `Human Edited by ${decision.approved_by}` : `Human Approved by ${decision.approved_by}`}
        />
      </Tooltip>
    );
  }
  return null;
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
    case "quality_check":
    case "critic": {
      const d = decision.decision_json as { verdict?: string; flags?: string[]; findings?: { severity: string; message: string }[] };
      return (
        <Stack spacing={0.5}>
          <Typography variant="body2">{decision.reasoning_text}</Typography>
          {d.findings?.map((f, i) => (
            <Typography key={i} variant="caption" color={f.severity === "high" ? "error.main" : "text.secondary"}>
              [{f.severity}] {f.message}
            </Typography>
          ))}
        </Stack>
      );
    }
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
  const proposedTargetColumn = decision.decision_json.target_column as string | null | undefined;
  const targetCandidates = (decision.decision_json.target_candidates ?? []) as {
    column: string;
    score: number;
    problem_type: string;
    reasoning: string[];
  }[];
  const requiresTargetSelection = !!decision.decision_json.requires_target_selection;
  const [overrideType, setOverrideType] = useState(proposedType ?? "clustering");
  const [overrideTargetColumn, setOverrideTargetColumn] = useState<string | null | undefined>(proposedTargetColumn);

  const selectCandidate = (candidate: { column: string; problem_type: string }) => {
    setOverrideTargetColumn(candidate.column);
    setOverrideType(candidate.problem_type);
  };

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
        <BusinessImpactNote decision={decision} />
      </Box>

      <TextField
        label="Your name"
        size="small"
        value={reviewedBy}
        onChange={(e) => setReviewedBy(e.target.value)}
        sx={{ mb: 2, minWidth: 260 }}
      />

      {isProblemDetection && targetCandidates.length > 0 && (
        <Box sx={{ mb: 2 }}>
          <Typography variant="body1" gutterBottom>
            {requiresTargetSelection
              ? "Select the column you want to predict (the highlighted one is only a suggestion):"
              : "Target column candidates (ranked by column name and value distribution):"}
          </Typography>
          <Stack spacing={0.5}>
            {targetCandidates.map((c) => (
              <Tooltip key={c.column} title={c.reasoning.join(" ")}>
                <Chip
                  label={`${c.column} — ${c.problem_type} (score ${c.score})`}
                  variant={overrideTargetColumn === c.column ? "filled" : "outlined"}
                  color={overrideTargetColumn === c.column ? "primary" : "default"}
                  onClick={() => selectCandidate(c)}
                  sx={{ justifyContent: "flex-start", maxWidth: 560, height: 36, fontSize: "1rem" }}
                />
              </Tooltip>
            ))}
          </Stack>
        </Box>
      )}
      {isProblemDetection && (
        <Stack direction="row" spacing={1} sx={{ mb: 2, alignItems: "center" }}>
          <Typography variant="body1">
            Problem type{overrideTargetColumn ? ` for "${overrideTargetColumn}"` : ""}:
          </Typography>
          {["clustering", "classification", "regression"].map((t) => (
            <Chip
              key={t}
              label={t}
              sx={{ height: 36, fontSize: "1rem" }}
              color={overrideType === t ? "primary" : "default"}
              onClick={() => {
                setOverrideType(t);
              }}
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
        {(() => {
          // For problem_detection, exactly one of these two buttons is ever active: "Approve
          // as proposed" while the reviewer hasn't touched the suggestion (the backend accepts
          // this even when requiresTargetSelection is true — see routers/pipeline.py), "Confirm
          // ..." once they've picked something different.
          const unchanged = !isProblemDetection ||
            (overrideType === proposedType && overrideTargetColumn === proposedTargetColumn);
          return (
            <Button
              variant="contained"
              color="success"
              disabled={review.isPending || !reviewedBy || !unchanged}
              onClick={() =>
                review.mutate({ decisionId: decision.id, action: "approve", reviewed_by: reviewedBy })
              }
            >
              Approve as proposed
            </Button>
          );
        })()}
        {isProblemDetection && (
          <Button
            variant="outlined"
            disabled={
              review.isPending ||
              !reviewedBy ||
              // A target must be set unless clustering is chosen.
              (overrideType !== "clustering" && !overrideTargetColumn) ||
              // Nothing to confirm that "Approve as proposed" doesn't already cover.
              (overrideType === proposedType && overrideTargetColumn === proposedTargetColumn)
            }
            onClick={() =>
              review.mutate({
                decisionId: decision.id,
                action: "edit",
                edits: {
                  problem_type: overrideType,
                  target_column: overrideType === "clustering" ? null : overrideTargetColumn,
                },
                reviewed_by: reviewedBy,
              })
            }
          >
            Confirm "{overrideType}"{overrideType !== "clustering" && overrideTargetColumn ? ` on "${overrideTargetColumn}"` : ""}
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
          Reject &amp; revise
        </Button>
      </Stack>

      {showReject && (
        <Stack direction="row" spacing={2} sx={{ mt: 2 }}>
          <TextField
            label="What should change? (the agent re-proposes using this)"
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
            Send back to agent
          </Button>
        </Stack>
      )}

      {review.isError && <Alert severity="error" sx={{ mt: 2 }}>Failed to submit review.</Alert>}
    </Paper>
  );
}

function BusinessImpactNote({ decision }: { decision: AgentDecision }) {
  const impact = decision.decision_json.business_impact as string | undefined;
  if (!impact) return null;
  return (
    <Typography variant="body2" color="text.secondary" sx={{ mt: 1, fontStyle: "italic" }}>
      Why this matters: {impact}
    </Typography>
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
  const { data: executiveSummary } = useExecutiveSummary(pipelineRunId);
  const { data: vizData, isLoading: vizLoading } = useVisualizations(pipelineRunId, run?.status, !!run?.job_id);

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

      <ExecutiveSummaryCard summary={executiveSummary} />

      {run.status === "failed" && <Alert severity="error">{run.error_message}</Alert>}

      <Stepper activeStep={activeIndex} orientation="vertical">
        {PIPELINE_STAGES.map((stage, i) => {
          const isActive = i === activeIndex;
          // Decisions are append-only, so a revision or self-correction retry leaves the previous
          // pass's later-stage decisions behind. Anything at/after the active stage that is still
          // (re)running is stale — don't show it as if it belonged to the current pass.
          const isStale = i > activeIndex || (isActive && run.status === "training");
          const decision = isStale ? undefined : decisionForAgent(stage.agentName);
          return (
            <Step key={stage.status} completed={i < activeIndex || run.status === "completed"}>
              <StepLabel
                error={run.status === "failed" && isActive}
                optional={decision && decision.status !== "proposed" ? <ApprovalBadge decision={decision} /> : undefined}
              >
                {stage.label}
              </StepLabel>
              <Box sx={{ pl: 2, pb: 2 }}>
                {stage.status === "training" && isActive && run.status === "training" && (
                  <TrainingProgressContent progress={trainingProgress} />
                )}
                {isActive && run.status !== "training" && !decision && <LinearProgress sx={{ maxWidth: 300 }} />}
                {decision && decision.status === "proposed" ? (
                  <DecisionReviewCard decision={decision} pipelineRunId={run.id} />
                ) : (
                  decision && (
                    <>
                      <StageSummary decision={decision} />
                      <BusinessImpactNote decision={decision} />
                    </>
                  )
                )}
              </Box>
            </Step>
          );
        })}
      </Stepper>

      {pendingDecision && !PIPELINE_STAGES.some((s) => s.agentName === pendingDecision.agent_name) && (
        <DecisionReviewCard decision={pendingDecision} pipelineRunId={run.id} />
      )}

      <InsightsSection data={vizData} loading={vizLoading} />

      {run.status === "completed" && report && (
        <Paper variant="outlined" sx={{ p: 3 }}>
          <Stack direction="row" sx={{ justifyContent: "space-between", alignItems: "center", mb: 2 }}>
            <Typography variant="h6">{report.title}</Typography>
            <Button variant="contained" href={pipelineReportExportUrl(run.id)} target="_blank" rel="noreferrer">
              Download PDF
            </Button>
          </Stack>
          <Stack spacing={2.5}>
            <Box>
              <Typography variant="subtitle1">Executive Summary</Typography>
              <Typography variant="body2" sx={{ mt: 0.5 }}>{report.executive_summary.problem_statement}</Typography>
              <Table size="small" sx={{ mt: 1 }}>
                <TableBody>
                  <TableRow><TableCell>Dataset Name</TableCell><TableCell>{report.executive_summary.dataset_overview.dataset_name}</TableCell></TableRow>
                  <TableRow><TableCell>Total Records</TableCell><TableCell>{report.executive_summary.dataset_overview.total_records.toLocaleString()}</TableCell></TableRow>
                  <TableRow><TableCell>Features Analyzed</TableCell><TableCell>{report.executive_summary.dataset_overview.features_analyzed.join(", ") || "—"}</TableCell></TableRow>
                  <TableRow><TableCell>Target Variable</TableCell><TableCell>{report.executive_summary.dataset_overview.target_variable ?? "None (unsupervised)"}</TableCell></TableRow>
                  <TableRow><TableCell>Business Domain</TableCell><TableCell>{report.executive_summary.dataset_overview.business_domain}</TableCell></TableRow>
                </TableBody>
              </Table>
            </Box>

            <Box>
              <Typography variant="subtitle1">Data Quality Summary</Typography>
              <Table size="small" sx={{ mt: 1 }}>
                <TableHead><TableRow><TableCell>Issue</TableCell><TableCell align="right">Count</TableCell></TableRow></TableHead>
                <TableBody>
                  <TableRow><TableCell>Missing Values</TableCell><TableCell align="right">{report.data_quality_summary.issues_detected.missing_values}</TableCell></TableRow>
                  <TableRow><TableCell>Duplicate Records</TableCell><TableCell align="right">{report.data_quality_summary.issues_detected.duplicate_records}</TableCell></TableRow>
                  <TableRow><TableCell>Invalid Data</TableCell><TableCell align="right">{report.data_quality_summary.issues_detected.invalid_data}</TableCell></TableRow>
                </TableBody>
              </Table>
              {report.data_quality_summary.actions_taken.length > 0 && (
                <Table size="small" sx={{ mt: 1 }}>
                  <TableHead><TableRow><TableCell>Column</TableCell><TableCell align="right">Missing</TableCell><TableCell>Action</TableCell><TableCell>Reason</TableCell></TableRow></TableHead>
                  <TableBody>
                    {report.data_quality_summary.actions_taken.map((a, i) => (
                      <TableRow key={i}>
                        <TableCell>{a.column}</TableCell><TableCell align="right">{a.missing_values}</TableCell>
                        <TableCell>{a.action}</TableCell><TableCell>{a.reason}</TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              )}
              <Typography variant="body2" sx={{ mt: 1 }}>{report.data_quality_summary.duplicate_handling}</Typography>
            </Box>

            <Box>
              <Typography variant="subtitle1">Data Preparation Summary</Typography>
              <Stack component="ul" sx={{ m: 0, pl: 2.5, mt: 0.5 }}>
                {report.data_preparation_summary.steps.map((s, i) => <Typography key={i} component="li" variant="body2">{s}</Typography>)}
              </Stack>
              {report.data_preparation_summary.train_pct !== null && (
                <Typography variant="body2" sx={{ mt: 0.5 }}>
                  Training Data: {report.data_preparation_summary.train_pct}% &nbsp;·&nbsp; Testing Data: {report.data_preparation_summary.test_pct}%
                </Typography>
              )}
            </Box>

            <Box>
              <Typography variant="subtitle1">Model Selection Summary</Typography>
              <Typography variant="body2" sx={{ mt: 0.5 }}>
                <b>Algorithms evaluated:</b> {report.model_selection_summary.algorithms_evaluated.join(", ") || "—"}
              </Typography>
              <Typography variant="body2"><b>Selected model:</b> {report.model_selection_summary.selected_model ?? "Pending"}</Typography>
              <Typography variant="body2"><b>Why selected:</b> {report.model_selection_summary.why_selected}</Typography>
              <Typography variant="body2"><b>Hyperparameter optimization:</b> {report.model_selection_summary.hyperparameter_optimization}</Typography>
            </Box>

            <Box>
              <Typography variant="subtitle1">Model Performance</Typography>
              <Typography variant="body2" sx={{ mt: 0.5 }}>{report.model_performance.reliability_sentence}</Typography>
              <Stack direction="row" spacing={1} sx={{ alignItems: "center", mt: 0.5 }}>
                <Chip size="small" label={report.model_performance.confidence_level} color={modelLevelColor(report.model_performance.confidence_level)} />
                <Typography variant="body2">{report.model_performance.confidence_explanation}</Typography>
              </Stack>
            </Box>

            <Box>
              <Typography variant="subtitle1">Key Insights</Typography>
              <Stack component="ul" sx={{ m: 0, pl: 2.5, mt: 0.5 }}>
                {report.key_insights.map((s, i) => <Typography key={i} component="li" variant="body2">{s}</Typography>)}
              </Stack>
            </Box>

            <Box>
              <Typography variant="subtitle1">Prediction Capability</Typography>
              <Typography variant="body2" sx={{ mt: 0.5 }}>{report.prediction_capability.description}</Typography>
              {report.prediction_capability.example_predictions.length > 0 && (
                <Stack component="ul" sx={{ m: 0, pl: 2.5, mt: 0.5 }}>
                  {report.prediction_capability.example_predictions.map((p, i) => (
                    <Typography key={i} component="li" variant="body2">
                      {JSON.stringify(p.input)} → <b>{String(p.prediction)}</b> ({p.confidence})
                      {p.suggested_business_action ? ` — ${p.suggested_business_action}` : ""}
                    </Typography>
                  ))}
                </Stack>
              )}
            </Box>

            <Box>
              <Typography variant="subtitle1">Business Recommendations</Typography>
              <Stack component="ul" sx={{ m: 0, pl: 2.5, mt: 0.5 }}>
                {report.business_recommendations.map((s, i) => <Typography key={i} component="li" variant="body2">{s}</Typography>)}
              </Stack>
            </Box>

            <Box>
              <Typography variant="subtitle1">Conclusion</Typography>
              <Typography variant="body2" sx={{ mt: 0.5 }}>{report.conclusion}</Typography>
            </Box>

            <Box>
              <Typography variant="subtitle1">Technical Appendix</Typography>
              {Object.keys(report.technical_appendix.model_details).length > 0 && (
                <Table size="small" sx={{ mt: 1 }}>
                  <TableHead><TableRow><TableCell>Metric</TableCell><TableCell>Value</TableCell></TableRow></TableHead>
                  <TableBody>
                    {Object.entries(report.technical_appendix.model_details)
                      .filter(([, v]) => !isConfusionMatrix(v))
                      .map(([k, v]) => (
                        <TableRow key={k}><TableCell>{k}</TableCell><TableCell>{typeof v === "object" ? JSON.stringify(v) : String(v)}</TableCell></TableRow>
                      ))}
                  </TableBody>
                </Table>
              )}
              {Object.values(report.technical_appendix.model_details).filter(isConfusionMatrix).map((cm, i) => (
                <Box key={i} sx={{ mt: 1.5 }}><ConfusionMatrixTable cm={cm} /></Box>
              ))}
              {report.technical_appendix.train_test_split && (
                <Typography variant="body2" sx={{ mt: 1 }}>
                  Training: {report.technical_appendix.train_test_split.training} records · Testing: {report.technical_appendix.train_test_split.testing} records
                </Typography>
              )}
            </Box>
          </Stack>
        </Paper>
      )}

      <ChatPanel pipelineRunId={run.id} />
    </Stack>
  );
}

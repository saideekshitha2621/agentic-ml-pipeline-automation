import { Box, Chip, LinearProgress, List, ListItem, ListItemIcon, ListItemText, Stack, Table, TableBody, TableCell, TableHead, TableRow, Tooltip, Typography } from "@mui/material";
import { CheckCircle, Error, HourglassEmpty, Warning } from "@mui/icons-material";
import type { AgentDecision, TrainingProgress } from "../../types";

const SEVERITY_COLOR: Record<string, "success" | "warning" | "error"> = { ok: "success", warning: "warning", critical: "error" };

function effectivePayload(d: AgentDecision): Record<string, unknown> {
  return (d.status === "edited" && d.human_edits_json) || d.decision_json;
}

export function DatasetUnderstandingContent({ decision }: { decision: AgentDecision }) {
  const du = (decision.decision_json.dataset_understanding ?? {}) as Record<string, unknown>;
  const missingByColumn = (du.missing_by_column ?? {}) as Record<string, number>;
  return (
    <Stack spacing={1.5}>
      <Stack direction="row" spacing={2} sx={{ flexWrap: "wrap" }}>
        {[
          ["Rows", du.n_rows], ["Columns", du.n_columns],
          ["Numerical", (du.numerical_columns as string[] | undefined)?.length ?? 0],
          ["Categorical", (du.categorical_columns as string[] | undefined)?.length ?? 0],
          ["Missing values", du.n_missing_values], ["Duplicate rows", du.n_duplicate_rows],
        ].map(([label, value]) => (
          <Box key={label as string} sx={{ minWidth: 110 }}>
            <Typography variant="caption" color="text.secondary">{label as string}</Typography>
            <Typography variant="h6">{String(value ?? 0)}</Typography>
          </Box>
        ))}
      </Stack>
      <Typography variant="body2">{du.data_quality_summary as string}</Typography>
      {!!(du.potential_target_columns as string[])?.length && (
        <Typography variant="body2">
          <b>Potential target columns:</b> {(du.potential_target_columns as string[]).join(", ")}
        </Typography>
      )}
      {Object.keys(missingByColumn).length > 0 && (
        <Table size="small">
          <TableHead><TableRow><TableCell>Column</TableCell><TableCell align="right">Missing values</TableCell></TableRow></TableHead>
          <TableBody>
            {Object.entries(missingByColumn).map(([col, n]) => (
              <TableRow key={col}><TableCell>{col}</TableCell><TableCell align="right">{n}</TableCell></TableRow>
            ))}
          </TableBody>
        </Table>
      )}
    </Stack>
  );
}

export function ValidationContent({ decision }: { decision: AgentDecision }) {
  const checks = (effectivePayload(decision).checks ?? []) as { name: string; status: string; detail: string; affected_columns: string[] }[];
  return (
    <List dense>
      {checks.map((c) => (
        <ListItem key={c.name}>
          <ListItemIcon>
            {c.status === "ok" ? <CheckCircle color="success" /> : c.status === "critical" ? <Error color="error" /> : <Warning color="warning" />}
          </ListItemIcon>
          <ListItemText
            primary={<Stack direction="row" spacing={1} sx={{ alignItems: "center" }}><span>{c.name}</span><Chip size="small" label={c.status} color={SEVERITY_COLOR[c.status]} /></Stack>}
            secondary={c.detail + (c.affected_columns.length ? ` (${c.affected_columns.join(", ")})` : "")}
          />
        </ListItem>
      ))}
    </List>
  );
}

export function CleaningPlanContent({ decision }: { decision: AgentDecision }) {
  const payload = effectivePayload(decision);
  const recs = (payload.recommendations ?? []) as {
    column: string; issue: string; action: string; reason: string; missing_count: number; missing_pct: number; no_information?: boolean;
  }[];
  const targetMissing = decision.decision_json.target_missing as { column: string; missing_count: number; missing_pct: number; reason: string } | null;
  return (
    <Stack spacing={1.5}>
      {targetMissing && (
        <Typography variant="body2" color="warning.main">
          <b>Target column ({targetMissing.column}):</b> {targetMissing.missing_count} row(s) ({targetMissing.missing_pct}%)
          have no target value — {targetMissing.reason}
        </Typography>
      )}
      <Table size="small">
        <TableHead>
          <TableRow>
            <TableCell>Column</TableCell><TableCell align="right">Missing count</TableCell><TableCell align="right">Missing %</TableCell>
            <TableCell>Recommended strategy</TableCell><TableCell>Reason</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {recs.map((r, i) => (
            <TableRow key={i}>
              <TableCell>{r.column}</TableCell>
              <TableCell align="right">{r.missing_count}</TableCell>
              <TableCell align="right">{r.missing_pct}%</TableCell>
              <TableCell>
                <Chip size="small" label={r.action.replace(/_/g, " ")} color={r.no_information ? "error" : undefined} />
              </TableCell>
              <TableCell>{r.reason}</TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </Stack>
  );
}

export function TransformationContent({ decision }: { decision: AgentDecision }) {
  const t = effectivePayload(decision) as { scaling_method?: string; scaling_reason?: string; encoding_method?: string; encoding_reason?: string; engineered_features?: { column: string; new_features: string[]; reason: string }[] };
  return (
    <Stack spacing={1}>
      <Typography variant="body2"><b>Scaling:</b> {t.scaling_method} — {t.scaling_reason}</Typography>
      <Typography variant="body2"><b>Encoding:</b> {t.encoding_method} — {t.encoding_reason}</Typography>
      {!!t.engineered_features?.length && (
        <Typography variant="body2"><b>Engineered features:</b> {t.engineered_features.map((f) => f.new_features.join(", ")).join("; ")}</Typography>
      )}
    </Stack>
  );
}

export function SplitContent({ decision }: { decision: AgentDecision }) {
  const s = effectivePayload(decision) as { applicable?: boolean; test_size?: number; train_size?: number; stratify?: boolean; reasoning?: string };
  if (!s.applicable) return <Typography variant="body2">{s.reasoning}</Typography>;
  return (
    <Typography variant="body2">
      <b>{Math.round((s.train_size ?? 0) * 100)}% train / {Math.round((s.test_size ?? 0) * 100)}% test</b>
      {s.stratify ? " (stratified)" : ""} — {s.reasoning}
    </Typography>
  );
}

export function AlgorithmShortlistContent({ decision }: { decision: AgentDecision }) {
  const shortlist = (effectivePayload(decision).shortlist ?? []) as { algorithm: string; recommended: boolean; rationale: string }[];
  return (
    <List dense>
      {shortlist.map((a) => (
        <ListItem key={a.algorithm}>
          <ListItemIcon>{a.recommended ? <CheckCircle color="success" fontSize="small" /> : <Box sx={{ width: 24 }} />}</ListItemIcon>
          <ListItemText primary={a.algorithm.replace(/_/g, " ")} secondary={a.rationale} />
        </ListItem>
      ))}
    </List>
  );
}

export function TrainingProgressContent({ progress }: { progress: TrainingProgress | undefined }) {
  if (!progress || progress.algorithms.length === 0) return <LinearProgress />;
  return (
    <List dense>
      {progress.algorithms.map((a) => (
        <ListItem key={a.algorithm}>
          <ListItemIcon>
            {a.status === "completed" ? <CheckCircle color="success" /> : a.status === "failed" ? <Error color="error" /> : <HourglassEmpty color="warning" />}
          </ListItemIcon>
          <ListItemText primary={a.algorithm.replace(/_/g, " ")} secondary={a.status} />
        </ListItem>
      ))}
    </List>
  );
}

// Classification reports f1_macro, regression reports r2 (see hpo_service.py's
// _PROBLEM_TYPE_CONFIG) — this component isn't handed problem_type, so the key metric is
// picked from whichever of these keys is actually present in the results.
const _HPO_METRIC_PRIORITY: { key: string; label: string }[] = [
  { key: "f1_macro", label: "F1" },
  { key: "r2", label: "R²" },
];

export function HPOContent({ decision }: { decision: AgentDecision }) {
  const results = (decision.decision_json.results ?? []) as {
    algorithm: string; skipped?: boolean; baseline_metrics?: Record<string, number>; optimized_metrics?: Record<string, number>; best_params?: Record<string, unknown>; n_trials?: number;
    status?: "improved" | "no_improvement" | "early_stopped" | "failed" | "skipped"; error?: string | null;
  }[];
  const metric = _HPO_METRIC_PRIORITY.find((m) =>
    results.some((r) => !r.skipped && r.baseline_metrics && m.key in r.baseline_metrics)
  ) ?? _HPO_METRIC_PRIORITY[0];
  return (
    <Table size="small">
      <TableHead>
        <TableRow>
          <TableCell>Algorithm</TableCell><TableCell>Baseline {metric.label}</TableCell><TableCell>Optimized {metric.label}</TableCell>
          <TableCell>Best params</TableCell><TableCell align="right">Trials</TableCell>
        </TableRow>
      </TableHead>
      <TableBody>
        {results.map((r) => (
          <TableRow key={r.algorithm}>
            <TableCell>{r.algorithm.replace(/_/g, " ")}</TableCell>
            <TableCell>{r.skipped ? "—" : r.baseline_metrics?.[metric.key]?.toFixed(3)}</TableCell>
            <TableCell>{r.skipped ? "—" : r.optimized_metrics?.[metric.key]?.toFixed(3)}</TableCell>
            <TableCell>
              {r.skipped ? "n/a" : JSON.stringify(r.best_params)}
              {r.status === "failed" && (
                <Typography variant="caption" color="error" display="block">
                  HPO failed — baseline kept{r.error ? `: ${r.error}` : ""}
                </Typography>
              )}
              {r.status === "no_improvement" && (
                <Typography variant="caption" color="text.secondary" display="block">
                  Search ran — baseline params were already the best
                </Typography>
              )}
              {r.status === "early_stopped" && (
                <Typography variant="caption" color="text.secondary" display="block">
                  Skipped search — baseline already near-perfect
                </Typography>
              )}
            </TableCell>
            <TableCell align="right">{r.n_trials ?? "—"}</TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  );
}

interface ConfusionMatrix {
  labels: string[];
  matrix: number[][];
}

export function isConfusionMatrix(v: unknown): v is ConfusionMatrix {
  const m = v as ConfusionMatrix | null;
  return !!m && Array.isArray(m.labels) && Array.isArray(m.matrix);
}

/** Actual (rows) vs predicted (columns); the diagonal is what the model got right. */
export function ConfusionMatrixTable({ cm }: { cm: ConfusionMatrix }) {
  return (
    <Box>
      <Typography variant="caption" color="text.secondary">
        Confusion matrix — rows are the actual outcome, columns are what the model predicted (diagonal = correct)
      </Typography>
      <Table size="small" sx={{ width: "auto", mt: 0.5 }}>
        <TableHead>
          <TableRow>
            <TableCell sx={{ fontWeight: 600 }}>Actual \ Predicted</TableCell>
            {cm.labels.map((l) => (
              <TableCell key={l} align="right" sx={{ fontWeight: 600 }}>{l}</TableCell>
            ))}
          </TableRow>
        </TableHead>
        <TableBody>
          {cm.matrix.map((row, i) => (
            <TableRow key={cm.labels[i]}>
              <TableCell sx={{ fontWeight: 600 }}>{cm.labels[i]}</TableCell>
              {row.map((count, j) => (
                <TableCell
                  key={j}
                  align="right"
                  sx={{ bgcolor: i === j ? "success.light" : count > 0 ? "error.light" : undefined, color: i === j || count > 0 ? "common.black" : undefined }}
                >
                  {count}
                </TableCell>
              ))}
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </Box>
  );
}

export function EvaluationContent({ decision }: { decision: AgentDecision }) {
  const e = decision.decision_json as { champion_algorithm?: string; metrics?: Record<string, unknown>; glossary?: Record<string, string> };
  const metrics = e.metrics ?? {};
  const confusion = isConfusionMatrix(metrics.confusion_matrix) ? metrics.confusion_matrix : null;
  return (
    <Stack spacing={1}>
      <Typography variant="body2"><b>Top model so far:</b> {e.champion_algorithm?.replace(/_/g, " ")}</Typography>
      <Stack direction="row" spacing={1} sx={{ flexWrap: "wrap" }}>
        {Object.entries(metrics).map(([k, v]) =>
          typeof v === "number" || typeof v === "string" ? (
            <Tooltip key={k} title={e.glossary?.[k] ?? ""}>
              <Chip label={`${k}: ${typeof v === "number" ? v.toFixed(3) : v}`} />
            </Tooltip>
          ) : null
        )}
      </Stack>
      {confusion && <ConfusionMatrixTable cm={confusion} />}
    </Stack>
  );
}

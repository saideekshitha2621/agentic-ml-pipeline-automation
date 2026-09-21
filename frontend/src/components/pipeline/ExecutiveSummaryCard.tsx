import { Box, Chip, Paper, Skeleton, Stack, Tooltip, Typography } from "@mui/material";
import type { ExecutiveSummary } from "../../types";
import { modelLevelColor } from "../../types";

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <Box sx={{ flex: 1, minWidth: 220 }}>
      <Typography variant="overline" color="text.secondary">{label}</Typography>
      <Box>{children}</Box>
    </Box>
  );
}

export default function ExecutiveSummaryCard({ summary }: { summary: ExecutiveSummary | undefined }) {
  if (!summary) return <Skeleton variant="rounded" height={140} />;

  return (
    <Paper variant="outlined" sx={{ p: 3, bgcolor: "action.hover" }}>
      <Typography variant="h6" sx={{ mb: 2 }}>Executive Summary</Typography>
      <Stack direction="row" spacing={3} useFlexGap sx={{ flexWrap: "wrap" }}>
        <Field label="Business problem">
          {summary.business_problem ? (
            <>
              <Typography variant="body2">{summary.business_problem.statement}</Typography>
              {summary.business_problem.key_features.length > 0 && (
                <Typography variant="caption" color="text.secondary" sx={{ display: "block", mt: 0.5 }}>
                  Key features: {summary.business_problem.key_features.join(", ")}
                </Typography>
              )}
            </>
          ) : (
            <Typography variant="body2" color="text.secondary">Analyzing the dataset…</Typography>
          )}
        </Field>
        <Field label="Dataset overview">
          {summary.dataset_overview ? (
            <Typography variant="body2">
              {summary.dataset_overview.n_rows.toLocaleString()} records, {summary.dataset_overview.n_columns} fields.{" "}
              {summary.dataset_overview.quality_summary}
            </Typography>
          ) : (
            <Typography variant="body2" color="text.secondary">Pending…</Typography>
          )}
        </Field>
        <Field label="ML type">
          {summary.ml_type ? (
            <Chip size="small" label={summary.ml_type} />
          ) : (
            <Typography variant="body2" color="text.secondary">Pending…</Typography>
          )}
        </Field>
        <Field label="Target variable">
          <Typography variant="body2">{summary.target_variable ?? "None (unsupervised)"}</Typography>
        </Field>
        <Field label="Recommended model">
          {summary.recommended_model ? (
            <Typography variant="body2">
              <b>{summary.recommended_model.algorithm.replace(/_/g, " ")}</b> — {summary.recommended_model.rationale}
            </Typography>
          ) : (
            <Typography variant="body2" color="text.secondary">Not yet available.</Typography>
          )}
        </Field>
        <Field label="Performance summary">
          {summary.performance_summary?.headline_metric_sentence ? (
            <>
              <Typography variant="body2">{summary.performance_summary.headline_metric_sentence}</Typography>
              {summary.performance_summary.level && summary.performance_summary.level !== "Unknown" && (
                <Tooltip title={summary.performance_summary.explanation ?? ""}>
                  <Chip
                    size="small"
                    sx={{ mt: 0.5 }}
                    color={modelLevelColor(summary.performance_summary.level)}
                    label={summary.performance_summary.level === "Verify" ? "Verify result" : `${summary.performance_summary.level} performance`}
                  />
                </Tooltip>
              )}
              {summary.performance_summary.tie_note && (
                <Typography variant="caption" color="text.secondary" sx={{ display: "block", mt: 0.5 }}>
                  {summary.performance_summary.tie_note}
                </Typography>
              )}
            </>
          ) : (
            <Typography variant="body2" color="text.secondary">Not yet available.</Typography>
          )}
        </Field>
      </Stack>
    </Paper>
  );
}

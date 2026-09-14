import { useEffect, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { Alert, Box, Button, LinearProgress, Paper, Slider, Stack, Typography } from "@mui/material";
import { Bar, CartesianGrid, ComposedChart, Line, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { usePCAPreview, useApplyPCA } from "../api/pca";
import { usePreprocessingPlan, useSavePreprocessingPlan } from "../api/preprocessing";

export default function PCAPage() {
  const { datasetId } = useParams();
  const navigate = useNavigate();
  const { data: preview, isLoading } = usePCAPreview(datasetId, true);
  const { data: plan } = usePreprocessingPlan(datasetId);
  const previewApply = useApplyPCA(datasetId);
  const savePlan = useSavePreprocessingPlan(datasetId);
  const [varianceTarget, setVarianceTarget] = useState(0.95);

  useEffect(() => {
    if (plan) setVarianceTarget(plan.pca_variance_target);
  }, [plan]);

  const chartData =
    preview?.components.map((c, i) => ({
      component: c,
      explained: preview.explained_variance_ratio[i],
      cumulative: preview.cumulative_variance[i],
    })) ?? [];

  const goNext = (pcaEnabled: boolean) => {
    if (!plan) return;
    savePlan.mutate(
      {
        numerical_columns: plan.numerical_columns,
        categorical_columns: plan.categorical_columns,
        dropped_columns: plan.dropped_columns,
        numerical_impute_strategy: plan.numerical_impute_strategy,
        categorical_impute_strategy: plan.categorical_impute_strategy,
        scaling_method: plan.scaling_method,
        drop_duplicates: plan.drop_duplicates,
        pca_enabled: pcaEnabled,
        pca_variance_target: varianceTarget,
      },
      { onSuccess: () => navigate(`/datasets/${datasetId}/execution`) },
    );
  };

  return (
    <Stack spacing={3} sx={{ maxWidth: 900 }}>
      <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <Typography variant="h4">Dimensionality Reduction (PCA)</Typography>
        <Stack direction="row" spacing={2}>
          <Button onClick={() => goNext(false)} disabled={savePlan.isPending}>
            Skip PCA →
          </Button>
          <Button variant="contained" disabled={savePlan.isPending} onClick={() => goNext(true)}>
            Apply PCA &amp; Continue →
          </Button>
        </Stack>
      </Box>

      {isLoading && <LinearProgress />}

      {preview && (
        <Paper variant="outlined" sx={{ p: 2 }}>
          <Typography variant="h6" gutterBottom>
            Explained Variance / Cumulative Variance
          </Typography>
          <ResponsiveContainer width="100%" height={320}>
            <ComposedChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="component" />
              <YAxis />
              <Tooltip />
              <Bar dataKey="explained" fill="#4C72B0" name="Explained Variance Ratio" />
              <Line type="monotone" dataKey="cumulative" stroke="#C44E52" name="Cumulative Variance" />
            </ComposedChart>
          </ResponsiveContainer>
        </Paper>
      )}

      <Paper variant="outlined" sx={{ p: 3 }}>
        <Typography gutterBottom>Variance to retain if PCA is applied: {(varianceTarget * 100).toFixed(0)}%</Typography>
        <Slider
          value={varianceTarget}
          min={0.5}
          max={0.99}
          step={0.01}
          onChange={(_e, v) => setVarianceTarget(v as number)}
          valueLabelDisplay="auto"
          valueLabelFormat={(v) => `${(v * 100).toFixed(0)}%`}
        />
        <Button
          sx={{ mt: 1 }}
          size="small"
          disabled={previewApply.isPending}
          onClick={() => previewApply.mutate(varianceTarget)}
        >
          Preview component count at this target
        </Button>
        {previewApply.data && (
          <Typography variant="body2" sx={{ mt: 1 }}>
            → {previewApply.data.n_components} components would retain{" "}
            {(previewApply.data.total_variance_retained * 100).toFixed(1)}% variance.
          </Typography>
        )}
      </Paper>

      {savePlan.isError && <Alert severity="error">Failed to save PCA choice.</Alert>}
    </Stack>
  );
}

import { useEffect, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import {
  Alert,
  Autocomplete,
  Box,
  Button,
  Checkbox,
  FormControlLabel,
  LinearProgress,
  MenuItem,
  Paper,
  Stack,
  TextField,
  Typography,
} from "@mui/material";
import { useDatasetProfile } from "../api/datasets";
import { useApplyPreprocessing, usePreprocessingPlan, useSavePreprocessingPlan } from "../api/preprocessing";
import type { PreprocessingPlanUpdate } from "../types";

export default function PreprocessingReviewPage() {
  const { datasetId } = useParams();
  const navigate = useNavigate();
  const { data: profile } = useDatasetProfile(datasetId);
  const { data: plan, isLoading } = usePreprocessingPlan(datasetId);
  const savePlan = useSavePreprocessingPlan(datasetId);
  const applyPlan = useApplyPreprocessing(datasetId);

  const [form, setForm] = useState<PreprocessingPlanUpdate | null>(null);

  useEffect(() => {
    if (plan && !form) {
      setForm({
        numerical_columns: plan.numerical_columns,
        categorical_columns: plan.categorical_columns,
        dropped_columns: plan.dropped_columns,
        numerical_impute_strategy: plan.numerical_impute_strategy,
        categorical_impute_strategy: plan.categorical_impute_strategy,
        scaling_method: plan.scaling_method,
        drop_duplicates: plan.drop_duplicates,
        pca_enabled: plan.pca_enabled,
        pca_variance_target: plan.pca_variance_target,
      });
    }
  }, [plan, form]);

  if (isLoading || !form) return <LinearProgress />;

  const allColumns = profile?.missing_values.map((m) => m.column) ?? [];

  const handleSaveAndContinue = () => {
    savePlan.mutate(form, {
      onSuccess: (savedPlan) => {
        applyPlan.mutate(savedPlan.id, {
          onSuccess: () => navigate(`/datasets/${datasetId}/pca`),
        });
      },
    });
  };

  return (
    <Stack spacing={3} sx={{ maxWidth: 900 }}>
      <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <Typography variant="h4">Preprocessing Review</Typography>
        <Button
          variant="contained"
          disabled={savePlan.isPending || applyPlan.isPending}
          onClick={handleSaveAndContinue}
        >
          Save &amp; Continue →
        </Button>
      </Box>

      <Paper variant="outlined" sx={{ p: 3 }}>
        <Stack spacing={3}>
          <Autocomplete
            multiple
            options={allColumns}
            value={form.numerical_columns}
            onChange={(_e, value) => setForm({ ...form, numerical_columns: value })}
            renderInput={(params) => <TextField {...params} label="Numerical Columns" />}
          />
          <Autocomplete
            multiple
            options={allColumns.filter((c) => !form.numerical_columns.includes(c))}
            value={form.categorical_columns}
            onChange={(_e, value) => setForm({ ...form, categorical_columns: value })}
            renderInput={(params) => <TextField {...params} label="Categorical Columns" />}
          />
          <Autocomplete
            multiple
            options={allColumns}
            value={form.dropped_columns}
            onChange={(_e, value) => setForm({ ...form, dropped_columns: value })}
            renderInput={(params) => <TextField {...params} label="Excluded / ID Columns" />}
          />

          <Stack direction="row" spacing={2}>
            <TextField
              select
              label="Numerical Imputation Strategy"
              value={form.numerical_impute_strategy}
              onChange={(e) => setForm({ ...form, numerical_impute_strategy: e.target.value })}
              sx={{ minWidth: 220 }}
            >
              <MenuItem value="median">Median</MenuItem>
              <MenuItem value="mean">Mean</MenuItem>
            </TextField>
            <TextField
              select
              label="Categorical Imputation Strategy"
              value={form.categorical_impute_strategy}
              disabled
              sx={{ minWidth: 220 }}
            >
              <MenuItem value="mode">Mode</MenuItem>
            </TextField>
            <TextField
              select
              label="Scaling Method"
              value={form.scaling_method}
              onChange={(e) => setForm({ ...form, scaling_method: e.target.value })}
              sx={{ minWidth: 220 }}
            >
              <MenuItem value="standard">StandardScaler</MenuItem>
              <MenuItem value="minmax">MinMaxScaler</MenuItem>
              <MenuItem value="robust">RobustScaler</MenuItem>
              <MenuItem value="none">None</MenuItem>
            </TextField>
          </Stack>

          <Stack direction="row" spacing={2}>
            <FormControlLabel
              control={
                <Checkbox
                  checked={form.drop_duplicates}
                  onChange={(e) => setForm({ ...form, drop_duplicates: e.target.checked })}
                />
              }
              label="Remove duplicate records"
            />
            <FormControlLabel
              control={
                <Checkbox
                  checked={form.pca_enabled}
                  onChange={(e) => setForm({ ...form, pca_enabled: e.target.checked })}
                />
              }
              label="Enable PCA (configure on next screen)"
            />
          </Stack>
        </Stack>
      </Paper>

      {(savePlan.isError || applyPlan.isError) && (
        <Alert severity="error">Failed to save/apply preprocessing plan.</Alert>
      )}
    </Stack>
  );
}

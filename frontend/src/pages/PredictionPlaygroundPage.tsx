import { useState } from "react";
import { useParams } from "react-router-dom";
import {
  Alert,
  Box,
  Button,
  Chip,
  LinearProgress,
  MenuItem,
  Paper,
  Select,
  Stack,
  TextField,
  Typography,
} from "@mui/material";
import { usePredict, usePredictionSchema } from "../api/prediction";

export default function PredictionPlaygroundPage() {
  const { pipelineRunId } = useParams();
  const { data: schema, isLoading } = usePredictionSchema(pipelineRunId);
  const predict = usePredict(pipelineRunId);
  const [values, setValues] = useState<Record<string, string>>({});

  if (isLoading) return <LinearProgress />;
  if (!schema) return <Alert severity="error">No prediction schema available for this run.</Alert>;

  const setValue = (feature: string, v: string) => setValues((prev) => ({ ...prev, [feature]: v }));

  const submit = () => {
    const features: Record<string, unknown> = {};
    for (const [feature, meta] of Object.entries(schema.feature_schema)) {
      const raw = values[feature] ?? meta.default;
      features[feature] = meta.type === "numeric" ? Number(raw) : raw;
    }
    predict.mutate(features);
  };

  return (
    <Stack spacing={3} sx={{ maxWidth: 700 }}>
      <Typography variant="h4">Prediction Playground</Typography>
      <Typography color="text.secondary">
        Enter feature values to get a live prediction from the approved champion model, with an explanation of the
        top contributing factors.
      </Typography>

      <Paper variant="outlined" sx={{ p: 3 }}>
        <Stack spacing={2}>
          {Object.entries(schema.feature_schema).map(([feature, meta]) =>
            meta.type === "categorical" ? (
              <Select
                key={feature}
                size="small"
                value={values[feature] ?? String(meta.default ?? "")}
                onChange={(e) => setValue(feature, e.target.value)}
                displayEmpty
              >
                {(meta.options ?? []).map((opt) => (
                  <MenuItem key={opt} value={opt}>
                    {feature}: {opt}
                  </MenuItem>
                ))}
              </Select>
            ) : (
              <TextField
                key={feature}
                label={`${feature} (${meta.min ?? "-"} to ${meta.max ?? "-"})`}
                size="small"
                type="number"
                value={values[feature] ?? meta.default ?? ""}
                onChange={(e) => setValue(feature, e.target.value)}
              />
            )
          )}
          <Button variant="contained" onClick={submit} disabled={predict.isPending}>
            Predict
          </Button>
        </Stack>
      </Paper>

      {predict.isError && <Alert severity="error">Prediction failed.</Alert>}

      {predict.data && (
        <Paper variant="outlined" sx={{ p: 3 }}>
          <Stack spacing={1.5}>
            <Stack direction="row" spacing={1} sx={{ alignItems: "center" }}>
              <Typography variant="h6">Prediction: {String(predict.data.prediction)}</Typography>
              {predict.data.confidence !== undefined && (
                <Chip label={`confidence ${Math.round(predict.data.confidence * 100)}%`} color="primary" />
              )}
            </Stack>
            {predict.data.probabilities && (
              <Stack direction="row" spacing={1} sx={{ flexWrap: "wrap" }}>
                {Object.entries(predict.data.probabilities).map(([cls, p]) => (
                  <Chip key={cls} label={`${cls}: ${(p * 100).toFixed(1)}%`} variant="outlined" />
                ))}
              </Stack>
            )}
            <Box>
              <Typography variant="subtitle2">Why this prediction?</Typography>
              <Typography variant="body2" sx={{ mt: 0.5 }}>{predict.data.explanation.narrative}</Typography>
            </Box>
          </Stack>
        </Paper>
      )}
    </Stack>
  );
}

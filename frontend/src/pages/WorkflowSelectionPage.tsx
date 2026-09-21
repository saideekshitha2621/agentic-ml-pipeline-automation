import { Box, Button, Card, CardActions, CardContent, Stack, ToggleButton, ToggleButtonGroup, Typography } from "@mui/material";
import { SmartToy, Tune } from "@mui/icons-material";
import { useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { useCreatePipelineRun } from "../api/pipeline";
import { useWorkspace } from "../components/WorkspaceContext";

export default function WorkflowSelectionPage() {
  const { datasetId } = useParams<{ datasetId: string }>();
  const { setPipelineRunId } = useWorkspace();
  const createPipelineRun = useCreatePipelineRun();
  const navigate = useNavigate();
  const [learningType, setLearningType] = useState<"auto" | "supervised" | "unsupervised">("auto");

  const handleAgenticWorkflow = () => {
    if (!datasetId) return;
    createPipelineRun.mutate(
      { dataset_id: datasetId, learning_type: learningType },
      {
        onSuccess: (run) => {
          setPipelineRunId(run.id);
          navigate(`/pipeline-runs/${run.id}`);
        },
      },
    );
  };

  return (
    <Stack spacing={3} sx={{ maxWidth: 800 }}>
      <Typography variant="h4">Choose Your Workflow</Typography>
      <Typography color="text.secondary">
        Your dataset has been uploaded. How would you like to proceed?
      </Typography>

      <Stack direction={{ xs: "column", sm: "row" }} spacing={2}>
        <Card variant="outlined" sx={{ flex: 1 }}>
          <CardContent>
            <Tune sx={{ fontSize: 40, color: "text.secondary" }} />
            <Typography variant="h6" sx={{ mt: 1 }}>
              Manual Workflow
            </Typography>
            <Typography color="text.secondary">
              Step through data quality, preprocessing, PCA, and model execution yourself, reviewing
              and adjusting each stage.
            </Typography>
          </CardContent>
          <CardActions>
            <Button
              variant="contained"
              onClick={() => datasetId && navigate(`/datasets/${datasetId}/quality`)}
            >
              Start Manual Workflow
            </Button>
          </CardActions>
        </Card>

        <Card variant="outlined" sx={{ flex: 1 }}>
          <CardContent>
            <SmartToy sx={{ fontSize: 40, color: "text.secondary" }} />
            <Typography variant="h6" sx={{ mt: 1 }}>
              Agentic Workflow
            </Typography>
            <Typography color="text.secondary">
              Let the agent automatically profile, preprocess, and run models on your dataset, with
              decisions you can review and approve.
            </Typography>
            <Typography variant="body2" sx={{ mt: 2, mb: 1 }}>
              What do you want to do with this data?
            </Typography>
            <ToggleButtonGroup
              exclusive
              size="small"
              orientation="vertical"
              fullWidth
              value={learningType}
              onChange={(_, v) => v && setLearningType(v)}
            >
              <ToggleButton value="auto">Not sure — suggest for me</ToggleButton>
              <ToggleButton value="supervised">Predict a column (classification / regression)</ToggleButton>
              <ToggleButton value="unsupervised">Find groups, no target (clustering)</ToggleButton>
            </ToggleButtonGroup>
          </CardContent>
          <CardActions>
            <Button
              variant="outlined"
              startIcon={<SmartToy />}
              disabled={createPipelineRun.isPending}
              onClick={handleAgenticWorkflow}
            >
              Start Agentic Workflow
            </Button>
          </CardActions>
        </Card>
      </Stack>

      <Box>
        <Button onClick={() => navigate("/")}>← Back to Dashboard</Button>
      </Box>
    </Stack>
  );
}

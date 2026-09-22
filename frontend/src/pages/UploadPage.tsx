import { useState, type DragEvent } from "react";
import { Alert, Box, Button, Paper, Stack, Typography } from "@mui/material";
import { UploadFile } from "@mui/icons-material";
import { useNavigate } from "react-router-dom";
import { useUploadDataset } from "../api/datasets";
import { useCreatePipelineRun } from "../api/pipeline";
import { useWorkspace } from "../components/WorkspaceContext";

export default function UploadPage() {
  const [dragOver, setDragOver] = useState(false);
  const upload = useUploadDataset();
  const createPipelineRun = useCreatePipelineRun();
  const { setDatasetId, setPipelineRunId } = useWorkspace();
  const navigate = useNavigate();

  const handleFile = (file: File) => {
    upload.mutate(file, {
      onSuccess: (dataset) => {
        setDatasetId(dataset.id);
        // Go straight into the agentic flow — no extra "choose a workflow" click. Problem
        // detection still runs in "auto" mode and pauses for review as usual; the manual
        // step-by-step workflow is still reachable from the dataset's own pages if wanted.
        createPipelineRun.mutate(
          { dataset_id: dataset.id, learning_type: "auto" },
          {
            onSuccess: (run) => {
              setPipelineRunId(run.id);
              navigate(`/pipeline-runs/${run.id}`);
            },
          },
        );
      },
    });
  };

  const onDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files?.[0];
    if (file) handleFile(file);
  };

  return (
    <Stack spacing={3} sx={{ maxWidth: 640 }}>
      <Typography variant="h4">Upload Dataset</Typography>

      <Paper
        variant="outlined"
        onDragOver={(e) => {
          e.preventDefault();
          setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={onDrop}
        sx={{
          p: 6,
          textAlign: "center",
          borderStyle: "dashed",
          borderWidth: 2,
          borderColor: dragOver ? "primary.main" : "divider",
          bgcolor: dragOver ? "action.hover" : "background.paper",
        }}
      >
        <UploadFile sx={{ fontSize: 48, color: "text.secondary" }} />
        <Typography sx={{ my: 2 }}>Drag and drop a CSV file here, or</Typography>
        <Button variant="contained" component="label" disabled={upload.isPending || createPipelineRun.isPending}>
          {upload.isPending ? "Uploading..." : createPipelineRun.isPending ? "Starting pipeline..." : "Choose File"}
          <input
            type="file"
            accept=".csv"
            hidden
            onChange={(e) => {
              const file = e.target.files?.[0];
              if (file) handleFile(file);
            }}
          />
        </Button>
      </Paper>

      {(upload.isError || createPipelineRun.isError) && (
        <Alert severity="error">
          {((upload.error ?? createPipelineRun.error) as { response?: { data?: { detail?: string } } })?.response
            ?.data?.detail ?? "Upload failed."}
        </Alert>
      )}

      <Box>
        <Button onClick={() => navigate("/")}>← Back to Dashboard</Button>
      </Box>
    </Stack>
  );
}

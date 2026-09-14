import { useState, type DragEvent } from "react";
import { Alert, Box, Button, Paper, Stack, Typography } from "@mui/material";
import { UploadFile } from "@mui/icons-material";
import { useNavigate } from "react-router-dom";
import { useUploadDataset } from "../api/datasets";
import { useWorkspace } from "../components/WorkspaceContext";

export default function UploadPage() {
  const [dragOver, setDragOver] = useState(false);
  const upload = useUploadDataset();
  const { setDatasetId } = useWorkspace();
  const navigate = useNavigate();

  const handleFile = (file: File) => {
    upload.mutate(file, {
      onSuccess: (dataset) => {
        setDatasetId(dataset.id);
        navigate(`/datasets/${dataset.id}/quality`);
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
        <Button variant="contained" component="label" disabled={upload.isPending}>
          {upload.isPending ? "Uploading..." : "Choose File"}
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

      {upload.isError && (
        <Alert severity="error">
          {(upload.error as { response?: { data?: { detail?: string } } })?.response?.data?.detail ??
            "Upload failed."}
        </Alert>
      )}

      <Box>
        <Button onClick={() => navigate("/")}>← Back to Dashboard</Button>
      </Box>
    </Stack>
  );
}

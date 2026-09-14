import { useParams } from "react-router-dom";
import { Button, Paper, Stack, Typography } from "@mui/material";
import { Download } from "@mui/icons-material";
import { exportUrl } from "../api/reports";

export default function ReportsPage() {
  const { jobId } = useParams();

  if (!jobId) return null;

  const formats: { format: "csv" | "xlsx" | "pdf"; label: string }[] = [
    { format: "csv", label: "Clustered Data (CSV)" },
    { format: "xlsx", label: "Full Report (Excel)" },
    { format: "pdf", label: "Evaluation Report (PDF)" },
  ];

  return (
    <Stack spacing={3} sx={{ maxWidth: 640 }}>
      <Typography variant="h4">Reports</Typography>
      <Typography color="text.secondary">
        Export the approved model's results. Excel includes the leaderboard, clustered data, cluster
        profiles, and business summaries on separate sheets; PDF is a condensed evaluation report.
      </Typography>

      <Paper variant="outlined" sx={{ p: 3 }}>
        <Stack spacing={2}>
          {formats.map(({ format, label }) => (
            <Button
              key={format}
              variant="outlined"
              startIcon={<Download />}
              href={exportUrl(jobId, format)}
              target="_blank"
              rel="noopener"
            >
              {label}
            </Button>
          ))}
        </Stack>
      </Paper>
    </Stack>
  );
}

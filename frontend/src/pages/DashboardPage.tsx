import { Box, Button, Paper, Stack, Table, TableBody, TableCell, TableHead, TableRow, Typography } from "@mui/material";
import { useNavigate } from "react-router-dom";
import { useDatasets } from "../api/datasets";
import MetricCard from "../components/MetricCard";
import { useWorkspace } from "../components/WorkspaceContext";

export default function DashboardPage() {
  const { data: datasets, isLoading } = useDatasets();
  const { setDatasetId } = useWorkspace();
  const navigate = useNavigate();

  const latest = datasets?.[0];

  return (
    <Stack spacing={3}>
      <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <Typography variant="h4">Dashboard</Typography>
        <Button variant="contained" onClick={() => navigate("/upload")}>
          Upload Dataset
        </Button>
      </Box>

      {latest && (
        <Stack direction="row" spacing={2} sx={{ flexWrap: "wrap" }}>
          <MetricCard label="Rows (latest dataset)" value={latest.n_rows.toLocaleString()} />
          <MetricCard label="Columns" value={latest.n_columns} />
          <MetricCard label="Data Quality Score" value={`${latest.data_quality_score}/100`} />
          <MetricCard label="Datasets Uploaded" value={datasets?.length ?? 0} />
        </Stack>
      )}

      <Paper variant="outlined" sx={{ p: 2 }}>
        <Typography variant="h6" gutterBottom>
          Upload History
        </Typography>
        {isLoading && <Typography color="text.secondary">Loading...</Typography>}
        {!isLoading && (!datasets || datasets.length === 0) && (
          <Typography color="text.secondary">No datasets uploaded yet.</Typography>
        )}
        {datasets && datasets.length > 0 && (
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell>Filename</TableCell>
                <TableCell>Uploaded</TableCell>
                <TableCell align="right">Rows</TableCell>
                <TableCell align="right">Columns</TableCell>
                <TableCell align="right">DQ Score</TableCell>
                <TableCell />
              </TableRow>
            </TableHead>
            <TableBody>
              {datasets.map((d) => (
                <TableRow key={d.id} hover>
                  <TableCell>{d.filename}</TableCell>
                  <TableCell>{new Date(d.uploaded_at).toLocaleString()}</TableCell>
                  <TableCell align="right">{d.n_rows.toLocaleString()}</TableCell>
                  <TableCell align="right">{d.n_columns}</TableCell>
                  <TableCell align="right">{d.data_quality_score}</TableCell>
                  <TableCell align="right">
                    <Button
                      size="small"
                      onClick={() => {
                        setDatasetId(d.id);
                        navigate(`/datasets/${d.id}/quality`);
                      }}
                    >
                      Open →
                    </Button>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        )}
      </Paper>
    </Stack>
  );
}

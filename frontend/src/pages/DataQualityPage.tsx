import { useParams, useNavigate } from "react-router-dom";
import {
  Alert,
  Box,
  Button,
  Chip,
  LinearProgress,
  Paper,
  Stack,
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableRow,
  Typography,
} from "@mui/material";
import { useDatasetProfile } from "../api/datasets";

export default function DataQualityPage() {
  const { datasetId } = useParams();
  const navigate = useNavigate();
  const { data: profile, isLoading } = useDatasetProfile(datasetId);

  if (isLoading || !profile) return <LinearProgress />;

  return (
    <Stack spacing={3}>
      <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <Typography variant="h4">Data Quality</Typography>
        <Button variant="contained" onClick={() => navigate(`/datasets/${datasetId}/preprocessing`)}>
          Continue to Preprocessing Review →
        </Button>
      </Box>

      <Stack direction="row" spacing={2}>
        <Chip label={`${profile.n_rows.toLocaleString()} rows`} />
        <Chip label={`${profile.n_columns} columns`} />
        <Chip label={`${profile.n_duplicates} duplicates`} color={profile.n_duplicates > 0 ? "warning" : "default"} />
        <Chip label={`DQ score: ${profile.data_quality_score}/100`} color="primary" />
      </Stack>

      <Paper variant="outlined" sx={{ p: 2 }}>
        <Typography variant="h6" gutterBottom>
          Missing Values
        </Typography>
        <Table size="small">
          <TableHead>
            <TableRow>
              <TableCell>Column</TableCell>
              <TableCell>Type</TableCell>
              <TableCell align="right">Missing</TableCell>
              <TableCell align="right">Missing %</TableCell>
              <TableCell align="right">Unique</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {profile.missing_values.map((row) => (
              <TableRow key={row.column}>
                <TableCell>{row.column}</TableCell>
                <TableCell>{row.dtype}</TableCell>
                <TableCell align="right">{row.missing_count}</TableCell>
                <TableCell align="right">{row.missing_pct}%</TableCell>
                <TableCell align="right">{row.n_unique}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </Paper>

      <Paper variant="outlined" sx={{ p: 2 }}>
        <Typography variant="h6" gutterBottom>
          Outlier Detection (IQR method)
        </Typography>
        {profile.outliers.length === 0 ? (
          <Typography color="text.secondary">No significant outliers detected.</Typography>
        ) : (
          <Table size="small">
            <TableHead>
              <TableRow>
                <TableCell>Column</TableCell>
                <TableCell align="right">Outliers</TableCell>
                <TableCell align="right">% of column</TableCell>
                <TableCell align="right">Bounds</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {profile.outliers.map((row) => (
                <TableRow key={row.column}>
                  <TableCell>{row.column}</TableCell>
                  <TableCell align="right">{row.n_outliers}</TableCell>
                  <TableCell align="right">{row.pct_outliers}%</TableCell>
                  <TableCell align="right">
                    [{row.lower_bound}, {row.upper_bound}]
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        )}
      </Paper>

      <Paper variant="outlined" sx={{ p: 2 }}>
        <Typography variant="h6" gutterBottom>
          Category Standardization Suggestions
        </Typography>
        {profile.category_suggestions.length === 0 ? (
          <Typography color="text.secondary">No category issues detected.</Typography>
        ) : (
          <Stack spacing={1}>
            {profile.category_suggestions.map((s, i) => (
              <Alert severity="info" key={i}>
                <b>{s.column}</b>: {s.issue}. {s.suggestion}
                <br />
                Examples: {s.examples.join(", ")}
              </Alert>
            ))}
          </Stack>
        )}
      </Paper>
    </Stack>
  );
}

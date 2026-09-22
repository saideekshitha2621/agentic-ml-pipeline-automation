import { CssBaseline, ThemeProvider } from "@mui/material";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Route, Routes } from "react-router-dom";
import { theme } from "./theme";
import NavShell from "./components/NavShell";
import { WorkspaceProvider } from "./components/WorkspaceContext";
import DashboardPage from "./pages/DashboardPage";
import UploadPage from "./pages/UploadPage";
import DataQualityPage from "./pages/DataQualityPage";
import PipelineRunPage from "./pages/PipelineRunPage";
import PredictionPlaygroundPage from "./pages/PredictionPlaygroundPage";

const queryClient = new QueryClient();

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <ThemeProvider theme={theme}>
        <CssBaseline />
        <WorkspaceProvider>
          <BrowserRouter>
            <NavShell>
              <Routes>
                <Route path="/" element={<DashboardPage />} />
                <Route path="/upload" element={<UploadPage />} />
                <Route path="/datasets/:datasetId/quality" element={<DataQualityPage />} />
                <Route path="/pipeline-runs/:pipelineRunId" element={<PipelineRunPage />} />
                <Route path="/pipeline-runs/:pipelineRunId/predict" element={<PredictionPlaygroundPage />} />
              </Routes>
            </NavShell>
          </BrowserRouter>
        </WorkspaceProvider>
      </ThemeProvider>
    </QueryClientProvider>
  );
}

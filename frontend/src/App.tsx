import { CssBaseline, ThemeProvider } from "@mui/material";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Route, Routes } from "react-router-dom";
import { theme } from "./theme";
import NavShell from "./components/NavShell";
import { WorkspaceProvider } from "./components/WorkspaceContext";
import DashboardPage from "./pages/DashboardPage";
import UploadPage from "./pages/UploadPage";
import DataQualityPage from "./pages/DataQualityPage";
import PreprocessingReviewPage from "./pages/PreprocessingReviewPage";
import PCAPage from "./pages/PCAPage";
import ModelExecutionPage from "./pages/ModelExecutionPage";
import LeaderboardPage from "./pages/LeaderboardPage";
import ComparisonPage from "./pages/ComparisonPage";
import ApprovalPage from "./pages/ApprovalPage";
import VisualizationPage from "./pages/VisualizationPage";
import ReportsPage from "./pages/ReportsPage";
import PipelineRunPage from "./pages/PipelineRunPage";

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
                <Route path="/datasets/:datasetId/preprocessing" element={<PreprocessingReviewPage />} />
                <Route path="/datasets/:datasetId/pca" element={<PCAPage />} />
                <Route path="/datasets/:datasetId/execution" element={<ModelExecutionPage />} />
                <Route path="/jobs/:jobId/leaderboard" element={<LeaderboardPage />} />
                <Route path="/jobs/:jobId/comparison" element={<ComparisonPage />} />
                <Route path="/jobs/:jobId/approval" element={<ApprovalPage />} />
                <Route path="/jobs/:jobId/visualization" element={<VisualizationPage />} />
                <Route path="/jobs/:jobId/reports" element={<ReportsPage />} />
                <Route path="/pipeline-runs/:pipelineRunId" element={<PipelineRunPage />} />
              </Routes>
            </NavShell>
          </BrowserRouter>
        </WorkspaceProvider>
      </ThemeProvider>
    </QueryClientProvider>
  );
}

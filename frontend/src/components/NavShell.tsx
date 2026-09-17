import { type ReactNode } from "react";
import {
  AppBar,
  Box,
  Drawer,
  List,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Toolbar,
  Typography,
  Chip,
} from "@mui/material";
import {
  Dashboard,
  UploadFile,
  FactCheck,
  Tune,
  ScatterPlot,
  PlayCircle,
  Leaderboard,
  CompareArrows,
  HowToReg,
  BarChart,
  Description,
  SmartToy,
  Science,
} from "@mui/icons-material";
import { useLocation, useNavigate } from "react-router-dom";
import { useWorkspace } from "./WorkspaceContext";

const DRAWER_WIDTH = 260;

export default function NavShell({ children }: { children: ReactNode }) {
  const navigate = useNavigate();
  const location = useLocation();
  const { datasetId, jobId, pipelineRunId } = useWorkspace();

  const navItems = [
    { label: "Dashboard", icon: <Dashboard />, path: "/" },
    { label: "Upload Dataset", icon: <UploadFile />, path: "/upload" },
    { label: "Data Quality", icon: <FactCheck />, path: datasetId ? `/datasets/${datasetId}/quality` : null },
    { label: "Preprocessing Review", icon: <Tune />, path: datasetId ? `/datasets/${datasetId}/preprocessing` : null },
    { label: "Dimensionality Reduction", icon: <ScatterPlot />, path: datasetId ? `/datasets/${datasetId}/pca` : null },
    { label: "Model Execution", icon: <PlayCircle />, path: datasetId ? `/datasets/${datasetId}/execution` : null },
    { label: "Leaderboard", icon: <Leaderboard />, path: jobId ? `/jobs/${jobId}/leaderboard` : null },
    { label: "Model Comparison", icon: <CompareArrows />, path: jobId ? `/jobs/${jobId}/comparison` : null },
    { label: "HITL Approval", icon: <HowToReg />, path: jobId ? `/jobs/${jobId}/approval` : null },
    { label: "Visualizations", icon: <BarChart />, path: jobId ? `/jobs/${jobId}/visualization` : null },
    { label: "Reports", icon: <Description />, path: jobId ? `/jobs/${jobId}/reports` : null },
    {
      label: "Agent Pipeline Run",
      icon: <SmartToy />,
      path: pipelineRunId ? `/pipeline-runs/${pipelineRunId}` : null,
    },
    {
      label: "Prediction Playground",
      icon: <Science />,
      path: pipelineRunId ? `/pipeline-runs/${pipelineRunId}/predict` : null,
    },
  ];

  return (
    <Box sx={{ display: "flex" }}>
      <AppBar position="fixed" sx={{ zIndex: (t) => t.zIndex.drawer + 1 }}>
        <Toolbar>
          <Typography variant="h6" noWrap component="div">
            Unsupervised AutoML Platform
          </Typography>
        </Toolbar>
      </AppBar>
      <Drawer
        variant="permanent"
        sx={{
          width: DRAWER_WIDTH,
          flexShrink: 0,
          [`& .MuiDrawer-paper`]: { width: DRAWER_WIDTH, boxSizing: "border-box" },
        }}
      >
        <Toolbar />
        <List>
          {navItems.map((item) => (
            <ListItemButton
              key={item.label}
              disabled={!item.path}
              selected={!!item.path && location.pathname === item.path}
              onClick={() => item.path && navigate(item.path)}
            >
              <ListItemIcon>{item.icon}</ListItemIcon>
              <ListItemText primary={item.label} />
            </ListItemButton>
          ))}
        </List>
        <Box sx={{ p: 2, mt: "auto" }}>
          {datasetId && <Chip size="small" label={`dataset: ${datasetId.slice(0, 8)}`} sx={{ mb: 1, display: "block" }} />}
          {jobId && <Chip size="small" label={`job: ${jobId.slice(0, 8)}`} color="secondary" />}
        </Box>
      </Drawer>
      <Box component="main" sx={{ flexGrow: 1, p: 3, width: `calc(100% - ${DRAWER_WIDTH}px)` }}>
        <Toolbar />
        {children}
      </Box>
    </Box>
  );
}

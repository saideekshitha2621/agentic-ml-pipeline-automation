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
import { Dashboard, UploadFile, FactCheck, SmartToy, Science } from "@mui/icons-material";
import { useLocation, useNavigate } from "react-router-dom";
import { useWorkspace } from "./WorkspaceContext";

const DRAWER_WIDTH = 260;

export default function NavShell({ children }: { children: ReactNode }) {
  const navigate = useNavigate();
  const location = useLocation();
  const { datasetId, pipelineRunId } = useWorkspace();

  const navItems = [
    { label: "Dashboard", icon: <Dashboard />, path: "/" },
    { label: "Upload Dataset", icon: <UploadFile />, path: "/upload" },
    { label: "Data Quality", icon: <FactCheck />, path: datasetId ? `/datasets/${datasetId}/quality` : null },
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
            AI Decision Engine
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
        </Box>
      </Drawer>
      <Box component="main" sx={{ flexGrow: 1, p: 3, width: `calc(100% - ${DRAWER_WIDTH}px)` }}>
        <Toolbar />
        {children}
      </Box>
    </Box>
  );
}

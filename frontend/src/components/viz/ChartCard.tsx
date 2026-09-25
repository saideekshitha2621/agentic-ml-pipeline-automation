import type { ReactNode } from "react";
import { Box, Paper, Typography } from "@mui/material";
import { viz } from "./vizTheme";

export default function ChartCard({
  title, subtitle, action, children, span = 1,
}: { title: string; subtitle?: string; action?: ReactNode; children: ReactNode; span?: 1 | 2 }) {
  return (
    <Paper
      variant="outlined"
      sx={{
        p: { xs: 2, sm: 3 }, borderRadius: 3, borderColor: viz.border, bgcolor: viz.surface,
        boxShadow: "0 1px 2px rgba(16,24,40,0.04)", minWidth: 0,
        gridColumn: { xs: "auto", lg: span === 2 ? "1 / -1" : "auto" },
      }}
    >
      <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: 2, flexWrap: "wrap", mb: 2.5 }}>
        <Box sx={{ minWidth: 0 }}>
          <Typography component="h3" sx={{ fontSize: "1.125rem", fontWeight: 600, color: viz.textPrimary, lineHeight: 1.3 }}>
            {title}
          </Typography>
          {subtitle && (
            <Typography sx={{ fontSize: "0.875rem", color: viz.textSecondary, mt: 0.5, lineHeight: 1.5 }}>{subtitle}</Typography>
          )}
        </Box>
        {action}
      </Box>
      {children}
    </Paper>
  );
}

import { createTheme } from "@mui/material/styles";

export const theme = createTheme({
  palette: {
    mode: "light",
    primary: { main: "#4C72B0" },
    secondary: { main: "#55A868" },
    background: { default: "#f7f8fa" },
  },
  shape: { borderRadius: 8 },
  typography: {
    body1: { fontSize: "1rem" },
    body2: { fontSize: "1rem" },
    caption: { fontSize: "1rem" },
    subtitle2: { fontSize: "1rem" },
    button: { fontSize: "1rem" },
  },
  components: {
    MuiChip: { styleOverrides: { root: { fontSize: "1rem", height: 32 } } },
    MuiTableCell: { styleOverrides: { root: { fontSize: "1rem" } } },
    MuiInputBase: { styleOverrides: { root: { fontSize: "1rem" } } },
    MuiInputLabel: { styleOverrides: { root: { fontSize: "1rem" } } },
    MuiMenuItem: { styleOverrides: { root: { fontSize: "1rem" } } },
    MuiFormControlLabel: { styleOverrides: { label: { fontSize: "1rem" } } },
    MuiStepLabel: { styleOverrides: { label: { fontSize: "1rem" } } },
    MuiTooltip: { styleOverrides: { tooltip: { fontSize: "0.9rem" } } },
  },
});

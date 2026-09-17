import { useState } from "react";
import {
  Alert,
  Box,
  Button,
  Chip,
  LinearProgress,
  MenuItem,
  Paper,
  Select,
  Stack,
  TextField,
  Typography,
} from "@mui/material";
import { CheckCircle, Science } from "@mui/icons-material";
import {
  useClearLLMSettings,
  useLLMSettings,
  useSaveLLMSettings,
  useTestLLMSettings,
  type LLMProvider,
} from "../api/settings";

const PROVIDER_LABELS: Record<LLMProvider, string> = {
  anthropic: "Claude (Anthropic)",
  gemini: "Gemini (Google)",
  openai: "OpenAI",
};

export default function SettingsPage() {
  const { data: status, isLoading } = useLLMSettings();
  const save = useSaveLLMSettings();
  const test = useTestLLMSettings();
  const clear = useClearLLMSettings();

  const [provider, setProvider] = useState<LLMProvider>("anthropic");
  const [apiKey, setApiKey] = useState("");
  const [model, setModel] = useState("");

  if (isLoading) return <LinearProgress />;

  const body = { provider, api_key: apiKey, model: model || undefined };

  return (
    <Stack spacing={3} sx={{ maxWidth: 640 }}>
      <Typography variant="h4">Settings</Typography>
      <Typography color="text.secondary">
        Connect an LLM provider to get richer, natural-language phrasing for business explanations and the
        Conversational Q&A panel. Without one configured, the platform stays fully functional using template-based
        problem statements, reports, and explanations.
      </Typography>

      <Paper variant="outlined" sx={{ p: 3 }}>
        <Stack direction="row" spacing={1} sx={{ mb: 2, alignItems: "center" }}>
          <Typography variant="h6">Current status</Typography>
          {status?.configured ? (
            <Chip
              icon={<CheckCircle />}
              color="success"
              label={`${PROVIDER_LABELS[status.provider as LLMProvider]} (${status.source})`}
            />
          ) : (
            <Chip label="Not configured — template-based mode" />
          )}
        </Stack>
        {status?.configured && (
          <>
            <Typography variant="body2">Model: {status.model}</Typography>
            <Typography variant="body2">API key: {status.masked_api_key}</Typography>
            {status.source === "database" && (
              <Button size="small" color="error" sx={{ mt: 1 }} onClick={() => clear.mutate()} disabled={clear.isPending}>
                Clear saved configuration
              </Button>
            )}
          </>
        )}
      </Paper>

      <Paper variant="outlined" sx={{ p: 3 }}>
        <Typography variant="h6" sx={{ mb: 2 }}>LLM Configuration</Typography>
        <Stack spacing={2}>
          <Select size="small" value={provider} onChange={(e) => setProvider(e.target.value as LLMProvider)}>
            {(Object.keys(PROVIDER_LABELS) as LLMProvider[]).map((p) => (
              <MenuItem key={p} value={p}>{PROVIDER_LABELS[p]}</MenuItem>
            ))}
          </Select>
          <TextField
            label="API key"
            type="password"
            size="small"
            value={apiKey}
            onChange={(e) => setApiKey(e.target.value)}
            placeholder="Paste your API key"
          />
          <TextField
            label="Model override (optional)"
            size="small"
            value={model}
            onChange={(e) => setModel(e.target.value)}
            placeholder="Leave blank to use the default model for this provider"
          />
          <Stack direction="row" spacing={2}>
            <Button
              startIcon={<Science />}
              variant="outlined"
              disabled={!apiKey || test.isPending}
              onClick={() => test.mutate(body)}
            >
              Test Connection
            </Button>
            <Button
              variant="contained"
              disabled={!apiKey || save.isPending}
              onClick={() => save.mutate(body)}
            >
              Save Configuration
            </Button>
          </Stack>

          {test.data && (
            <Alert severity={test.data.success ? "success" : "error"}>{test.data.message}</Alert>
          )}
          {save.isSuccess && <Alert severity="success">Configuration saved — it will be used for every new explanation and chat message.</Alert>}
        </Stack>
      </Paper>

      <Box>
        <Typography variant="subtitle2">Where these keys live (for reference)</Typography>
        <Typography variant="body2" color="text.secondary">
          Saved configuration is stored in the backend's local database (<code>app_settings</code> table) and always
          takes precedence. Without a saved configuration, the backend falls back to the environment variables
          <code> ANTHROPIC_API_KEY</code>, <code>GEMINI_API_KEY</code>, or <code>OPENAI_API_KEY</code>.
        </Typography>
      </Box>
    </Stack>
  );
}

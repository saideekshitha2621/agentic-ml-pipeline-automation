import { useState } from "react";
import { Alert, Box, Button, IconButton, Paper, Stack, TextField, Typography, Drawer, Fab, CircularProgress } from "@mui/material";
import { Chat, Close, Send, SmartToy, Person } from "@mui/icons-material";
import { useApplyChatAction, useChatHistory, useSendChatMessage } from "../../api/chat";
import type { ChatSuggestedAction } from "../../types";

export default function ChatPanel({ pipelineRunId }: { pipelineRunId: string }) {
  const [open, setOpen] = useState(false);
  const [question, setQuestion] = useState("");
  const { data: messages } = useChatHistory(open ? pipelineRunId : undefined);
  const send = useSendChatMessage(pipelineRunId);
  const apply = useApplyChatAction(pipelineRunId);
  const [action, setAction] = useState<ChatSuggestedAction | null>(null);

  const submit = () => {
    if (!question.trim()) return;
    setAction(null);
    send.mutate(question.trim(), { onSuccess: (msg) => setAction(msg.suggested_action ?? null) });
    setQuestion("");
  };

  return (
    <>
      <Fab
        variant="extended"
        color="primary"
        onClick={() => setOpen(true)}
        aria-label="Ask about this pipeline"
        sx={{ position: "fixed", bottom: 24, right: 24, zIndex: (theme) => theme.zIndex.drawer + 2 }}
      >
        <Chat sx={{ mr: 1 }} />
        Ask AI
      </Fab>
      <Drawer anchor="right" open={open} onClose={() => setOpen(false)}>
        <Box sx={{ width: { xs: "100vw", sm: 640, lg: 760 }, display: "flex", flexDirection: "column", height: "100%" }}>
          <Stack direction="row" sx={{ p: 2, alignItems: "center", justifyContent: "space-between", borderBottom: 1, borderColor: "divider" }}>
            <Typography variant="h6">Ask about this pipeline</Typography>
            <IconButton onClick={() => setOpen(false)}>
              <Close />
            </IconButton>
          </Stack>
          <Box sx={{ flexGrow: 1, overflowY: "auto", p: 2 }}>
            <Stack spacing={1.5}>
              {(!messages || messages.length === 0) && (
                <Typography color="text.secondary" variant="body2">
                  Ask about the dataset, preprocessing decisions, models, or predictions from this run.
                </Typography>
              )}
              {messages?.map((m) => (
                <Paper
                  key={m.id}
                  variant="outlined"
                  sx={{
                    p: 1.5,
                    alignSelf: m.role === "user" ? "flex-end" : "flex-start",
                    bgcolor: m.role === "user" ? "primary.main" : "background.paper",
                    color: m.role === "user" ? "primary.contrastText" : "text.primary",
                    maxWidth: "85%",
                  }}
                >
                  <Stack direction="row" spacing={0.5} sx={{ alignItems: "center", mb: 0.5 }}>
                    {m.role === "user" ? <Person fontSize="small" /> : <SmartToy fontSize="small" />}
                    <Typography variant="caption">{m.role === "user" ? "You" : "Assistant"}</Typography>
                  </Stack>
                  <Typography variant="body2" sx={{ whiteSpace: "pre-wrap" }}>
                    {m.content}
                  </Typography>
                </Paper>
              ))}
              {send.isPending && <CircularProgress size={20} />}
              {action && (
                <Alert
                  severity="info"
                  action={
                    <Button
                      color="inherit"
                      size="small"
                      disabled={apply.isPending}
                      onClick={() => apply.mutate(action, { onSuccess: () => setAction(null) })}
                    >
                      Confirm
                    </Button>
                  }
                >
                  Proposed action: send the pending {action.agent_name.replace(/_/g, " ")} back to its agent.
                </Alert>
              )}
              {apply.isError && <Alert severity="error">Could not apply that action.</Alert>}
            </Stack>
          </Box>
          <Stack direction="row" spacing={1} sx={{ p: 2, borderTop: 1, borderColor: "divider" }}>
            <TextField
              fullWidth
              size="small"
              placeholder="Ask a question..."
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && submit()}
            />
            <IconButton color="primary" onClick={submit} disabled={send.isPending || !question.trim()}>
              <Send />
            </IconButton>
          </Stack>
        </Box>
      </Drawer>
    </>
  );
}

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { apiClient } from "./client";
import type { ChatMessage } from "../types";

export function useChatHistory(pipelineRunId: string | undefined) {
  return useQuery({
    queryKey: ["pipeline-run", pipelineRunId, "chat"],
    queryFn: async () => (await apiClient.get<ChatMessage[]>(`/api/v1/pipeline-runs/${pipelineRunId}/chat`)).data,
    enabled: !!pipelineRunId,
  });
}

export function useSendChatMessage(pipelineRunId: string | undefined) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (question: string) =>
      (await apiClient.post<ChatMessage>(`/api/v1/pipeline-runs/${pipelineRunId}/chat`, { question })).data,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["pipeline-run", pipelineRunId, "chat"] });
    },
  });
}

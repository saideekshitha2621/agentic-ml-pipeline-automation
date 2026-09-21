import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { apiClient } from "./client";
import type { ChatMessage, ChatSuggestedAction } from "../types";

export function useChatHistory(pipelineRunId: string | undefined) {
  return useQuery({
    queryKey: ["pipeline-run", pipelineRunId, "chat"],
    queryFn: async () => (await apiClient.get<ChatMessage[]>(`/api/v1/pipeline-runs/${pipelineRunId}/chat`)).data,
    enabled: !!pipelineRunId,
  });
}

/** Applies a chat-proposed action. It goes through the normal review endpoint, so the same
 * audit trail, revision limits and human gates apply as for a manual rejection. */
export function useApplyChatAction(pipelineRunId: string | undefined) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (action: ChatSuggestedAction) =>
      (
        await apiClient.post(`/api/v1/pipeline-runs/${pipelineRunId}/decisions/${action.decision_id}/review`, {
          ...action.review_request,
          reviewed_by: "chat request (confirmed by user)",
        })
      ).data,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["pipeline-run", pipelineRunId] });
    },
  });
}

export function useSendChatMessage(pipelineRunId: string | undefined) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (question: string) =>
      (await apiClient.post<ChatMessage & { suggested_action?: ChatSuggestedAction | null }>(`/api/v1/pipeline-runs/${pipelineRunId}/chat`, { question })).data,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["pipeline-run", pipelineRunId, "chat"] });
    },
  });
}

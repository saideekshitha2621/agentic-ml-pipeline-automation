import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { apiClient } from "./client";

export type LLMProvider = "anthropic" | "gemini" | "openai";

export interface LLMSettingsStatus {
  provider: LLMProvider | null;
  model: string | null;
  masked_api_key: string | null;
  configured: boolean;
  source: "database" | "environment" | "none";
}

export interface LLMSettingsRequest {
  provider: LLMProvider;
  api_key: string;
  model?: string;
}

export interface LLMTestResult {
  success: boolean;
  message: string;
}

export function useLLMSettings() {
  return useQuery({
    queryKey: ["settings", "llm"],
    queryFn: async () => (await apiClient.get<LLMSettingsStatus>("/api/v1/settings/llm")).data,
  });
}

export function useSaveLLMSettings() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (body: LLMSettingsRequest) =>
      (await apiClient.post<LLMSettingsStatus>("/api/v1/settings/llm", body)).data,
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["settings", "llm"] }),
  });
}

export function useTestLLMSettings() {
  return useMutation({
    mutationFn: async (body: LLMSettingsRequest) =>
      (await apiClient.post<LLMTestResult>("/api/v1/settings/llm/test", body)).data,
  });
}

export function useClearLLMSettings() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async () => (await apiClient.delete<LLMSettingsStatus>("/api/v1/settings/llm")).data,
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["settings", "llm"] }),
  });
}

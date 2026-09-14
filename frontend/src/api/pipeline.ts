import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { apiClient } from "./client";
import type { AgentDecision, AgentRecommendation, PipelineReport, PipelineRun } from "../types";

const RUNNING_STATUSES = new Set([
  "profiling",
  "model_execution",
  "reporting",
]);

export function useCreatePipelineRun() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (body: { dataset_id: string; target_column?: string }) =>
      (await apiClient.post<PipelineRun>("/api/v1/pipeline-runs", body)).data,
    onSuccess: (run) => {
      queryClient.invalidateQueries({ queryKey: ["pipeline-run", run.id] });
    },
  });
}

export function usePipelineRun(pipelineRunId: string | undefined) {
  return useQuery({
    queryKey: ["pipeline-run", pipelineRunId],
    queryFn: async () => (await apiClient.get<PipelineRun>(`/api/v1/pipeline-runs/${pipelineRunId}`)).data,
    enabled: !!pipelineRunId,
    refetchInterval: (query) => (RUNNING_STATUSES.has(query.state.data?.status ?? "") ? 1500 : false),
  });
}

export function usePipelineDecisions(pipelineRunId: string | undefined) {
  return useQuery({
    queryKey: ["pipeline-run", pipelineRunId, "decisions"],
    queryFn: async () =>
      (await apiClient.get<AgentDecision[]>(`/api/v1/pipeline-runs/${pipelineRunId}/decisions`)).data,
    enabled: !!pipelineRunId,
    refetchInterval: 2000,
  });
}

export function useReviewDecision(pipelineRunId: string | undefined) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (body: {
      decisionId: string;
      action: "approve" | "edit" | "reject";
      edits?: Record<string, unknown>;
      reason?: string;
      reviewed_by: string;
    }) =>
      (
        await apiClient.post<AgentDecision>(
          `/api/v1/pipeline-runs/${pipelineRunId}/decisions/${body.decisionId}/review`,
          { action: body.action, edits: body.edits, reason: body.reason, reviewed_by: body.reviewed_by },
        )
      ).data,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["pipeline-run", pipelineRunId] });
      queryClient.invalidateQueries({ queryKey: ["pipeline-run", pipelineRunId, "decisions"] });
    },
  });
}

export function usePipelineRecommendation(pipelineRunId: string | undefined, enabled: boolean) {
  return useQuery({
    queryKey: ["pipeline-run", pipelineRunId, "recommendation"],
    queryFn: async () =>
      (await apiClient.get<AgentRecommendation>(`/api/v1/pipeline-runs/${pipelineRunId}/recommendation`)).data,
    enabled: !!pipelineRunId && enabled,
  });
}

export function usePipelineReport(pipelineRunId: string | undefined, enabled: boolean) {
  return useQuery({
    queryKey: ["pipeline-run", pipelineRunId, "report"],
    queryFn: async () => (await apiClient.get<PipelineReport>(`/api/v1/pipeline-runs/${pipelineRunId}/report`)).data,
    enabled: !!pipelineRunId && enabled,
  });
}

export function pipelineReportExportUrl(pipelineRunId: string): string {
  return `${apiClient.defaults.baseURL}/api/v1/pipeline-runs/${pipelineRunId}/report/export?format=pdf`;
}

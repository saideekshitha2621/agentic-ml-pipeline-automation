import { useQuery } from "@tanstack/react-query";
import { apiClient } from "./client";
import type { ClusterInterpretation, VisualizationBundle } from "../types";

export function useVisualizations(jobId: string | undefined, runId?: string) {
  return useQuery({
    queryKey: ["visualizations", jobId, runId],
    queryFn: async () =>
      (
        await apiClient.get<VisualizationBundle>(`/api/v1/jobs/${jobId}/visualizations`, {
          params: runId ? { run_id: runId } : undefined,
        })
      ).data,
    enabled: !!jobId,
  });
}

export function useInterpretation(jobId: string | undefined, runId?: string) {
  return useQuery({
    queryKey: ["interpretation", jobId, runId],
    queryFn: async () =>
      (
        await apiClient.get<ClusterInterpretation>(`/api/v1/jobs/${jobId}/interpretation`, {
          params: runId ? { run_id: runId } : undefined,
        })
      ).data,
    enabled: !!jobId,
  });
}

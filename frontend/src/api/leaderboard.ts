import { useQuery } from "@tanstack/react-query";
import { apiClient } from "./client";
import type { ClusterRun } from "../types";

export function useLeaderboard(jobId: string | undefined) {
  return useQuery({
    queryKey: ["leaderboard", jobId],
    queryFn: async () => (await apiClient.get<ClusterRun[]>(`/api/v1/jobs/${jobId}/leaderboard`)).data,
    enabled: !!jobId,
  });
}

export function useCompareRuns(jobId: string | undefined, runIds: string[]) {
  return useQuery({
    queryKey: ["compare", jobId, runIds],
    queryFn: async () =>
      (
        await apiClient.get<ClusterRun[]>(`/api/v1/jobs/${jobId}/compare`, {
          params: { run_ids: runIds.join(",") },
        })
      ).data,
    enabled: !!jobId && runIds.length > 0,
  });
}

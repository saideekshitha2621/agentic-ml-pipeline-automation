import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { apiClient } from "./client";
import type { Approval, Recommendation } from "../types";

export function useRecommendations(jobId: string | undefined) {
  return useQuery({
    queryKey: ["recommendations", jobId],
    queryFn: async () => (await apiClient.get<Recommendation[]>(`/api/v1/jobs/${jobId}/recommendations`)).data,
    enabled: !!jobId,
  });
}

export function useApproveModel(jobId: string | undefined) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (body: { cluster_run_id: string; approved_by: string; notes?: string }) =>
      (await apiClient.post<Approval>(`/api/v1/jobs/${jobId}/approve`, body)).data,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["visualizations", jobId] });
      queryClient.invalidateQueries({ queryKey: ["interpretation", jobId] });
    },
  });
}

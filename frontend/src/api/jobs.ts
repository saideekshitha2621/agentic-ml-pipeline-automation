import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { apiClient } from "./client";
import type { Job } from "../types";

export function useCreateJob() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (body: { dataset_id: string; preprocessing_plan_id: string; config?: Record<string, unknown> }) =>
      (await apiClient.post<Job>("/api/v1/jobs", body)).data,
    onSuccess: (job) => {
      queryClient.invalidateQueries({ queryKey: ["job", job.id] });
    },
  });
}

export function useJob(jobId: string | undefined) {
  return useQuery({
    queryKey: ["job", jobId],
    queryFn: async () => (await apiClient.get<Job>(`/api/v1/jobs/${jobId}`)).data,
    enabled: !!jobId,
    refetchInterval: (query) => {
      const status = query.state.data?.status;
      return status === "running" || status === "queued" ? 1500 : false;
    },
  });
}

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { apiClient } from "./client";
import type { PreprocessingPlan, PreprocessingPlanUpdate, PreprocessingReport } from "../types";

export function usePreprocessingPlan(datasetId: string | undefined) {
  return useQuery({
    queryKey: ["preprocessing-plan", datasetId],
    queryFn: async () =>
      (await apiClient.get<PreprocessingPlan>(`/api/v1/datasets/${datasetId}/preprocessing-plan`)).data,
    enabled: !!datasetId,
  });
}

export function useSavePreprocessingPlan(datasetId: string | undefined) {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (body: PreprocessingPlanUpdate) =>
      (await apiClient.put<PreprocessingPlan>(`/api/v1/datasets/${datasetId}/preprocessing-plan`, body)).data,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["preprocessing-plan", datasetId] });
    },
  });
}

export function useApplyPreprocessing(datasetId: string | undefined) {
  return useMutation({
    mutationFn: async (planId: string) =>
      (
        await apiClient.post<PreprocessingReport>(
          `/api/v1/datasets/${datasetId}/preprocess`,
          null,
          { params: { plan_id: planId } },
        )
      ).data,
  });
}

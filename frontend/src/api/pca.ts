import { useMutation, useQuery } from "@tanstack/react-query";
import { apiClient } from "./client";
import type { PCAPreview, PCAReport } from "../types";

export function usePCAPreview(datasetId: string | undefined, enabled: boolean) {
  return useQuery({
    queryKey: ["pca-preview", datasetId],
    queryFn: async () => (await apiClient.get<PCAPreview>(`/api/v1/datasets/${datasetId}/pca-preview`)).data,
    enabled: !!datasetId && enabled,
  });
}

export function useApplyPCA(datasetId: string | undefined) {
  return useMutation({
    mutationFn: async (nComponents: number) =>
      (await apiClient.post<PCAReport>(`/api/v1/datasets/${datasetId}/pca`, { n_components: nComponents })).data,
  });
}

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { apiClient } from "./client";
import type { Dataset, DataProfile } from "../types";

export const datasetKeys = {
  all: ["datasets"] as const,
  detail: (id: string) => ["datasets", id] as const,
  profile: (id: string) => ["datasets", id, "profile"] as const,
};

export function useDatasets() {
  return useQuery({
    queryKey: datasetKeys.all,
    queryFn: async () => (await apiClient.get<Dataset[]>("/api/v1/datasets")).data,
  });
}

export function useDataset(id: string | undefined) {
  return useQuery({
    queryKey: datasetKeys.detail(id ?? ""),
    queryFn: async () => (await apiClient.get<Dataset>(`/api/v1/datasets/${id}`)).data,
    enabled: !!id,
  });
}

export function useDatasetProfile(id: string | undefined) {
  return useQuery({
    queryKey: datasetKeys.profile(id ?? ""),
    queryFn: async () => (await apiClient.get<DataProfile>(`/api/v1/datasets/${id}/profile`)).data,
    enabled: !!id,
  });
}

export function useUploadDataset() {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn: async (file: File) => {
      const formData = new FormData();
      formData.append("file", file);
      const res = await apiClient.post<Dataset>("/api/v1/datasets", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      return res.data;
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: datasetKeys.all });
    },
  });
}

import { useMutation, useQuery } from "@tanstack/react-query";
import { apiClient } from "./client";
import type { PipelineRun } from "../types";

export interface PredictionSchemaResponse {
  feature_schema: PipelineRun["feature_schema_json"];
  target_classes: string[];
}

export interface PredictionResponse {
  prediction: string | number;
  probabilities?: Record<string, number>;
  confidence?: number;
  explanation: {
    top_features: { feature: string; contribution: number }[];
    narrative: string;
  };
}

export function usePredictionSchema(pipelineRunId: string | undefined) {
  return useQuery({
    queryKey: ["pipeline-run", pipelineRunId, "prediction-schema"],
    queryFn: async () =>
      (await apiClient.get<PredictionSchemaResponse>(`/api/v1/pipeline-runs/${pipelineRunId}/prediction-schema`)).data,
    enabled: !!pipelineRunId,
  });
}

export function usePredict(pipelineRunId: string | undefined) {
  return useMutation({
    mutationFn: async (features: Record<string, unknown>) =>
      (await apiClient.post<PredictionResponse>(`/api/v1/pipeline-runs/${pipelineRunId}/predict`, { features })).data,
  });
}

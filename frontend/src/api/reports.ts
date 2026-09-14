import { API_BASE } from "./client";

export function exportUrl(jobId: string, format: "csv" | "xlsx" | "pdf", runId?: string) {
  const params = new URLSearchParams({ format });
  if (runId) params.set("run_id", runId);
  return `${API_BASE}/api/v1/jobs/${jobId}/export?${params.toString()}`;
}

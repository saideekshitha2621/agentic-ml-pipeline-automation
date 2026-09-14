from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class ClusterRun(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    job_id: str
    algorithm: str
    params_json: dict
    extra_json: dict
    n_clusters: int
    n_noise: int
    noise_pct: float
    silhouette_score: float | None
    davies_bouldin_score: float | None
    calinski_harabasz_score: float | None
    composite_score: float | None
    rank: int | None


class Recommendation(BaseModel):
    cluster_run: ClusterRun
    rationale: str
    strengths: list[str]
    weaknesses: list[str]
    cluster_size_breakdown: dict[str, int]

from __future__ import annotations

from pydantic import BaseModel


class ScatterPoint(BaseModel):
    x: float
    y: float
    cluster: str


class ClusterSizeRow(BaseModel):
    cluster: str
    count: int
    pct_of_total: float


class MetricComparisonRow(BaseModel):
    run_id: str
    algorithm: str
    silhouette_score: float | None
    davies_bouldin_score: float | None
    calinski_harabasz_score: float | None


class VisualizationBundle(BaseModel):
    scatter: list[ScatterPoint]
    cluster_sizes: list[ClusterSizeRow]
    metric_comparison: list[MetricComparisonRow]


class ClusterInterpretation(BaseModel):
    profiles: list[dict]
    feature_importance: list[dict]
    summaries: dict[str, str]
    suggested_names: dict[str, str]

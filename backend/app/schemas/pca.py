from __future__ import annotations

from pydantic import BaseModel


class PCAPreview(BaseModel):
    explained_variance_ratio: list[float]
    cumulative_variance: list[float]
    components: list[str]


class PCAApplyRequest(BaseModel):
    n_components: float = 0.95  # int (# components) or float in (0,1] (variance target)


class PCAReport(BaseModel):
    n_components: int
    explained_variance_ratio: list[float]
    cumulative_explained_variance: list[float]
    total_variance_retained: float
    original_n_features: int

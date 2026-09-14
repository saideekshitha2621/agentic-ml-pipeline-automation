from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class JobCreateRequest(BaseModel):
    dataset_id: str
    preprocessing_plan_id: str
    config: dict = {}  # per-algorithm hyperparameter grid overrides; falls back to defaults


class Job(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    dataset_id: str
    preprocessing_plan_id: str
    status: str
    progress_pct: float
    log_lines: list[str]
    pca_report_json: dict
    started_at: datetime
    completed_at: datetime | None
    error_message: str | None

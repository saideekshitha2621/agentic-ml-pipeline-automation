from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict


class PipelineRunCreateRequest(BaseModel):
    dataset_id: str
    target_column: str | None = None  # optional explicit intent; skips target-detection heuristics


class PipelineRun(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    dataset_id: str
    declared_target: str | None
    status: str
    problem_type: str | None
    job_id: str | None
    created_at: datetime
    updated_at: datetime
    error_message: str | None


class AgentDecision(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    pipeline_run_id: str
    agent_name: str
    stage: str
    decision_json: dict
    confidence: float | None
    reasoning_text: str
    status: str
    human_edits_json: dict | None
    override_reason: str | None
    approved_by: str | None
    approved_at: datetime | None
    created_at: datetime


class DecisionReviewRequest(BaseModel):
    action: Literal["approve", "edit", "reject"]
    edits: dict | None = None  # required when action == "edit"
    reason: str | None = None  # required when action == "reject"
    reviewed_by: str = "unspecified user"

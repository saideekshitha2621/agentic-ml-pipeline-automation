from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class ApprovalRequest(BaseModel):
    cluster_run_id: str
    approved_by: str
    notes: str | None = None


class Approval(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    job_id: str
    cluster_run_id: str
    approved_by: str
    approved_at: datetime
    notes: str | None

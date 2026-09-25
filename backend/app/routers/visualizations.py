"""Chart-ready data for the run page's Insights section (see services/visualization_service.py)."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.models import PipelineRun as PipelineRunORM
from app.services import visualization_service

router = APIRouter(prefix="/api/v1/pipeline-runs", tags=["visualizations"])


@router.get("/{pipeline_run_id}/visualizations")
def get_visualizations(pipeline_run_id: str, db: Session = Depends(get_db)):
    run = db.get(PipelineRunORM, pipeline_run_id)
    if not run:
        raise HTTPException(404, "Pipeline run not found.")
    return visualization_service.build(db, run)

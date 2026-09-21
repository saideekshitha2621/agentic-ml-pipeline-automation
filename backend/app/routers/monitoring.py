"""Operational visibility (Phase 5): drift for a deployed model, and quality metrics for the agents."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.agents import monitoring_agent
from app.db.database import get_db
from app.db.models import AgentDecision as AgentDecisionORM
from app.db.models import PipelineRun as PipelineRunORM
from app.db.models import PredictionLog as PredictionLogORM
from app.services import agent_metrics_service, llm_service, task_queue_service

router = APIRouter(prefix="/api/v1", tags=["monitoring"])


@router.get("/pipeline-runs/{pipeline_run_id}/monitoring")
def run_monitoring(pipeline_run_id: str, limit: int = Query(500, ge=20, le=5000), db: Session = Depends(get_db)):
    """Drift report for a deployed champion, from its most recent `limit` logged predictions."""
    run = db.get(PipelineRunORM, pipeline_run_id)
    if not run:
        raise HTTPException(404, "Pipeline run not found.")
    if not run.champion_model_path:
        raise HTTPException(409, "This run has no deployed model to monitor (clustering runs and unfinished runs have none).")
    rows = (
        db.query(PredictionLogORM)
        .filter_by(pipeline_run_id=pipeline_run_id)
        .order_by(PredictionLogORM.created_at.desc())
        .limit(limit)
        .all()
    )
    logs = [{"input": r.input_json, "output": r.output_json} for r in reversed(rows)]  # oldest first
    return {"pipeline_run_id": pipeline_run_id, **monitoring_agent.assess(run.feature_schema_json or {}, logs)}


@router.get("/agent-metrics")
def agent_metrics(db: Session = Depends(get_db)):
    """How well are the agents doing? Approval/edit/rejection rates, confidence, LLM-vs-fallback
    share, and the cost of first-attempt mistakes (human revisions, automatic self-corrections)."""
    rows = db.query(AgentDecisionORM).all()
    decisions = [
        {
            "pipeline_run_id": r.pipeline_run_id, "agent_name": r.agent_name, "status": r.status,
            "approved_by": r.approved_by, "confidence": r.confidence, "decision_json": r.decision_json,
        }
        for r in rows
    ]
    return {
        **agent_metrics_service.summarize(decisions),
        "llm": llm_service.usage_stats(),
        "task_queue": task_queue_service.stats(),
    }

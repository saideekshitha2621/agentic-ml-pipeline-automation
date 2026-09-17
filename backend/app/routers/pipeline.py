"""Agent-driven pipeline endpoints — additive and parallel to the existing manual
datasets -> preprocessing -> jobs -> approval flow, which is untouched. A PipelineRun
that confirms 'clustering' delegates to the same Job/ModelRun machinery those routers
already use, so GET /jobs/{id}, /leaderboard, /visualizations, /export all keep working
unmodified on the job a pipeline run creates.
"""
from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.agents import orchestrator, reporting_agent
from app.core.config import EXPORTS_DIR
from app.db.database import get_db
from app.db.models import AgentDecision as AgentDecisionORM
from app.db.models import Dataset as DatasetORM
from app.db.models import Job as JobORM
from app.db.models import PipelineRun as PipelineRunORM
from app.schemas.pipeline import AgentDecision, DecisionReviewRequest, PipelineRun, PipelineRunCreateRequest

router = APIRouter(prefix="/api/v1/pipeline-runs", tags=["pipeline"])


def _get_run_or_404(pipeline_run_id: str, db: Session) -> PipelineRunORM:
    run = db.get(PipelineRunORM, pipeline_run_id)
    if not run:
        raise HTTPException(404, "Pipeline run not found.")
    return run


@router.post("", response_model=PipelineRun)
def create_pipeline_run(
    body: PipelineRunCreateRequest, background_tasks: BackgroundTasks, db: Session = Depends(get_db)
):
    dataset = db.get(DatasetORM, body.dataset_id)
    if not dataset:
        raise HTTPException(404, "Dataset not found.")

    run = PipelineRunORM(dataset_id=body.dataset_id, declared_target=body.target_column, status="profiling")
    db.add(run)
    db.commit()
    db.refresh(run)

    background_tasks.add_task(orchestrator.start, run.id)
    return run


@router.get("/{pipeline_run_id}", response_model=PipelineRun)
def get_pipeline_run(pipeline_run_id: str, db: Session = Depends(get_db)):
    return _get_run_or_404(pipeline_run_id, db)


@router.get("/{pipeline_run_id}/decisions", response_model=list[AgentDecision])
def list_decisions(pipeline_run_id: str, db: Session = Depends(get_db)):
    _get_run_or_404(pipeline_run_id, db)
    return (
        db.query(AgentDecisionORM)
        .filter_by(pipeline_run_id=pipeline_run_id)
        .order_by(AgentDecisionORM.created_at)
        .all()
    )


@router.post("/{pipeline_run_id}/decisions/{decision_id}/review", response_model=AgentDecision)
def review_decision(
    pipeline_run_id: str,
    decision_id: str,
    body: DecisionReviewRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    run = _get_run_or_404(pipeline_run_id, db)
    decision = db.get(AgentDecisionORM, decision_id)
    if not decision or decision.pipeline_run_id != pipeline_run_id:
        raise HTTPException(404, "Decision not found for this pipeline run.")
    if decision.status != "proposed":
        raise HTTPException(409, "This decision has already been reviewed.")

    if body.action == "approve":
        decision.status = "approved"
    elif body.action == "edit":
        if not body.edits:
            raise HTTPException(400, "`edits` is required when action is 'edit'.")
        decision.status = "edited"
        decision.human_edits_json = body.edits
    elif body.action == "reject":
        if not body.reason:
            raise HTTPException(400, "`reason` is required when action is 'reject'.")
        decision.status = "rejected"
        decision.override_reason = body.reason

    from datetime import datetime, timezone

    decision.approved_by = body.reviewed_by
    decision.approved_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(decision)

    if body.action == "reject":
        run.status = "failed"
        run.error_message = f"Rejected at stage '{decision.stage}': {body.reason}"
        db.commit()
        return decision

    _DISPATCH = {
        "problem_detection": orchestrator.advance_after_problem_approval,
        "data_validation": orchestrator.advance_after_validation_approval,
        "cleaning_plan": orchestrator.advance_after_cleaning_approval,
        "transformation": orchestrator.advance_after_transformation_approval,
        "train_test_split": orchestrator.advance_after_split_approval,
        "algorithm_recommendation": orchestrator.advance_after_algorithm_approval,
        "recommendation": orchestrator.finalize_after_recommendation_approval,
    }
    next_step = _DISPATCH.get(decision.agent_name)
    if next_step:
        background_tasks.add_task(next_step, pipeline_run_id)

    return decision


@router.get("/{pipeline_run_id}/training-progress")
def training_progress(pipeline_run_id: str, db: Session = Depends(get_db)):
    run = _get_run_or_404(pipeline_run_id, db)
    if not run.job_id:
        return {"algorithms": [], "job_status": None}
    job = db.get(JobORM, run.job_id)
    return {"algorithms": [{"algorithm": k, "status": v} for k, v in (job.training_status_json or {}).items()],
            "job_status": job.status, "progress_pct": job.progress_pct}


@router.get("/{pipeline_run_id}/recommendation")
def get_recommendation(pipeline_run_id: str, db: Session = Depends(get_db)):
    _get_run_or_404(pipeline_run_id, db)
    decision = (
        db.query(AgentDecisionORM)
        .filter_by(pipeline_run_id=pipeline_run_id, agent_name="recommendation")
        .order_by(AgentDecisionORM.created_at.desc())
        .first()
    )
    if not decision:
        raise HTTPException(409, "No recommendation yet — the run may still be executing.")
    return decision.decision_json


@router.get("/{pipeline_run_id}/report")
def get_report(pipeline_run_id: str, db: Session = Depends(get_db)):
    _get_run_or_404(pipeline_run_id, db)
    decision = (
        db.query(AgentDecisionORM)
        .filter_by(pipeline_run_id=pipeline_run_id, agent_name="reporting")
        .order_by(AgentDecisionORM.created_at.desc())
        .first()
    )
    if not decision:
        raise HTTPException(409, "Report not generated yet — the run may still be executing.")
    return decision.decision_json["report"]


@router.get("/{pipeline_run_id}/report/export")
def export_report(pipeline_run_id: str, format: str = Query(..., pattern="^(pdf)$"), db: Session = Depends(get_db)):
    report = get_report(pipeline_run_id, db)  # reuses the 409/404 handling above

    out_dir = EXPORTS_DIR / pipeline_run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path: Path = out_dir / f"{uuid.uuid4()}.pdf"
    reporting_agent.export_pdf(report, out_path)

    return FileResponse(out_path, media_type="application/pdf", filename=f"pipeline-run-{pipeline_run_id}.pdf")

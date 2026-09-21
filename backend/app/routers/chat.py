"""Conversational Q&A (req 14) — grounds every answer in this run's own stored artifacts:
dataset profile, the full AgentDecision trail, and champion model metrics/importances."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.models import AgentDecision as AgentDecisionORM
from app.db.models import ChatMessage as ChatMessageORM
from app.db.models import Dataset as DatasetORM
from app.db.models import PipelineRun as PipelineRunORM
from app.schemas.pipeline import ChatRequest
from app.services import chat_action_service, llm_service

router = APIRouter(prefix="/api/v1/pipeline-runs", tags=["chat"])

MAX_HISTORY_MESSAGES = 20


def _build_grounding_context(run: PipelineRunORM, dataset: DatasetORM, decisions: list[AgentDecisionORM]) -> str:
    lines = [
        f"Dataset: {dataset.filename} — {dataset.n_rows} rows, {dataset.n_columns} columns, "
        f"quality score {dataset.data_quality_score}/100.",
        f"Pipeline status: {run.status}. Problem type: {run.problem_type}. Target column: {run.declared_target}.",
    ]
    for d in decisions:
        lines.append(f"[{d.stage}/{d.agent_name}] ({d.status}) {d.reasoning_text} -- data: {d.decision_json}")
    if run.champion_model_path:
        lines.append(f"Champion model persisted. Feature importance: {run.feature_importance_json}")
    return "\n".join(lines)


@router.get("/{pipeline_run_id}/chat")
def list_chat(pipeline_run_id: str, db: Session = Depends(get_db)):
    if not db.get(PipelineRunORM, pipeline_run_id):
        raise HTTPException(404, "Pipeline run not found.")
    return (
        db.query(ChatMessageORM)
        .filter_by(pipeline_run_id=pipeline_run_id)
        .order_by(ChatMessageORM.created_at)
        .all()
    )


@router.post("/{pipeline_run_id}/chat")
def ask_chat(pipeline_run_id: str, body: ChatRequest, db: Session = Depends(get_db)):
    run = db.get(PipelineRunORM, pipeline_run_id)
    if not run:
        raise HTTPException(404, "Pipeline run not found.")
    dataset = db.get(DatasetORM, run.dataset_id)
    decisions = list(run.decisions)

    history_rows = (
        db.query(ChatMessageORM)
        .filter_by(pipeline_run_id=pipeline_run_id)
        .order_by(ChatMessageORM.created_at)
        .all()
    )[-MAX_HISTORY_MESSAGES:]
    history = [{"role": m.role, "content": m.content} for m in history_rows]

    user_message = ChatMessageORM(pipeline_run_id=pipeline_run_id, role="user", content=body.question)
    db.add(user_message)
    db.commit()

    # An instruction ("use random forest instead") becomes a *proposed* action the user must
    # confirm; it short-circuits the LLM call entirely (no quota used, no hallucinated claims).
    suggested_action = chat_action_service.detect_action(db, pipeline_run_id, body.question)
    if suggested_action:
        answer = suggested_action["description"] + " Confirm below to apply it."
    else:
        context = _build_grounding_context(run, dataset, decisions)
        answer = llm_service.chat(body.question, context, history)

    assistant_message = ChatMessageORM(pipeline_run_id=pipeline_run_id, role="assistant", content=answer)
    db.add(assistant_message)
    db.commit()
    db.refresh(assistant_message)

    return {
        "id": assistant_message.id,
        "pipeline_run_id": assistant_message.pipeline_run_id,
        "role": assistant_message.role,
        "content": assistant_message.content,
        "created_at": assistant_message.created_at,
        "suggested_action": suggested_action,
    }

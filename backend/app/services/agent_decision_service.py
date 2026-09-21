"""Thin persistence helper for AgentDecision rows — every agent's output passes through
this one function so the shape is enforced in one place rather than each agent (or the
orchestrator) hand-rolling DB writes."""
from __future__ import annotations

from sqlalchemy.orm import Session

from app.db.models import AgentDecision as AgentDecisionORM
from app.db.models import PipelineRun as PipelineRunORM


def record_decision(
    db: Session,
    pipeline_run: PipelineRunORM,
    agent_name: str,
    stage: str,
    decision: dict,
    confidence: float | None,
    reasoning: str,
    auto_approve: bool = False,
) -> AgentDecisionORM:
    """auto_approve=True is for stages that don't need a HITL gate (e.g. profiling is a
    factual summary, not a judgment call) — the decision row still exists for the audit
    trail/timeline, just pre-approved by the system rather than left `proposed`."""
    row = AgentDecisionORM(
        pipeline_run_id=pipeline_run.id,
        agent_name=agent_name,
        stage=stage,
        decision_json=decision,
        confidence=confidence,
        reasoning_text=reasoning,
        status="approved" if auto_approve else "proposed",
        approved_by="system" if auto_approve else None,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


def latest_decision(db: Session, pipeline_run_id: str, agent_name: str) -> AgentDecisionORM | None:
    return (
        db.query(AgentDecisionORM)
        .filter_by(pipeline_run_id=pipeline_run_id, agent_name=agent_name)
        .order_by(AgentDecisionORM.created_at.desc())
        .first()
    )


# --- Revision loop (Phase 1) -----------------------------------------------------------
# A rejected proposal no longer kills the run: the same agent is re-invoked with the
# rejected proposals and the reviewer's reasons as `feedback`, up to MAX_REVISIONS times.

MAX_REVISIONS = 2

# Agents whose proposal can meaningfully be regenerated from feedback. data_validation is a
# factual report (re-running it gives the same answer), so a rejection there still stops.
# A rejected final recommendation re-recommends the next-best model.
REVISABLE_AGENTS = {
    "problem_detection": "profiling",
    "cleaning_plan": "cleaning_plan",
    "transformation": "transformation",
    "train_test_split": "train_test_split",
    "algorithm_recommendation": "algorithm_recommendation",
    "recommendation": "evaluation",
}  # agent_name -> PipelineRun.status shown while the revision runs


def rejection_history(db: Session, pipeline_run_id: str, agent_name: str) -> list[dict]:
    """Every rejected proposal for this agent, oldest first, as `{"proposal", "reason"}`."""
    rows = (
        db.query(AgentDecisionORM)
        .filter_by(pipeline_run_id=pipeline_run_id, agent_name=agent_name, status="rejected")
        .order_by(AgentDecisionORM.created_at)
        .all()
    )
    return [{"proposal": r.decision_json, "reason": r.override_reason or ""} for r in rows]


def can_revise(db: Session, pipeline_run_id: str, agent_name: str) -> bool:
    if agent_name not in REVISABLE_AGENTS:
        return False
    return len(rejection_history(db, pipeline_run_id, agent_name)) <= MAX_REVISIONS


def all_decisions(db: Session, pipeline_run_id: str, agent_name: str) -> list[AgentDecisionORM]:
    return (
        db.query(AgentDecisionORM)
        .filter_by(pipeline_run_id=pipeline_run_id, agent_name=agent_name)
        .order_by(AgentDecisionORM.created_at)
        .all()
    )

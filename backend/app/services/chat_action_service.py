"""Chat -> gated actions (Phase 4).

The chat assistant may now *propose* an action when the user gives an instruction rather than
asks a question ("use random forest instead", "exclude knn", "use a 30% test set"). It never
executes anything itself: the proposal names the pending decision and the exact review request
(`reject` with the user's instruction as the reason), and only runs when the user confirms it
through the normal review endpoint — so the same audit trail, revision bounds and human gates
apply as for any other rejection. Rule-based on purpose: no LLM call (or quota) is needed.
"""
from __future__ import annotations

import re

from sqlalchemy.orm import Session

from app.db.models import AgentDecision as AgentDecisionORM
from app.services import agent_decision_service

_IMPERATIVE = re.compile(
    r"^\s*(please\s+|can you\s+|could you\s+)?(use|switch|change|try|exclude|include|remove|drop|skip|"
    r"don'?t|do not|instead|redo|re-?run|make it|set|pick|choose|keep)\b",
    re.IGNORECASE,
)


def detect_action(db: Session, pipeline_run_id: str, question: str) -> dict | None:
    if not _IMPERATIVE.search(question or ""):
        return None
    pending = (
        db.query(AgentDecisionORM)
        .filter_by(pipeline_run_id=pipeline_run_id, status="proposed")
        .order_by(AgentDecisionORM.created_at.desc())
        .first()
    )
    if pending is None or not agent_decision_service.can_revise(db, pipeline_run_id, pending.agent_name):
        return None
    stage = pending.agent_name.replace("_", " ")
    return {
        "type": "revise_pending_decision",
        "decision_id": pending.id,
        "agent_name": pending.agent_name,
        "review_request": {"action": "reject", "reason": question.strip()},
        "description": (
            f"Send the pending '{stage}' proposal back to its agent with your instruction: “{question.strip()}”. "
            "The agent will re-propose and you will review the new proposal."
        ),
    }

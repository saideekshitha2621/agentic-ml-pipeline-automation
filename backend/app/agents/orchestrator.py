"""Pipeline Orchestrator — public API.

The actual sequencing/state-machine logic lives in `pipeline_graph.py` as a compiled
LangGraph `StateGraph` (nodes = stages, conditional edges = auto-approve vs. HITL-gate vs.
fail routing). This module is a thin, stable API in front of that graph: every function here
keeps the exact name/signature the FastAPI routers already call as a `BackgroundTask`, so
nothing outside this file needs to know the pipeline is graph-based.

`pipeline_run_id` doubles as the LangGraph `thread_id` — the compiled graph's checkpointer
persists exactly where a run is paused, so resuming never needs to know which stage it's
resuming from; `graph.invoke(None, config)` picks up right where the thread left off.
"""
from __future__ import annotations

import logging

from app.agents.pipeline_graph import compiled_graph
from app.db.database import SessionLocal
from app.db.models import PipelineRun as PipelineRunORM
from app.services import agent_decision_service, task_queue_service

logger = logging.getLogger(__name__)

# Human-review gate node -> the agent whose decision the human reviews there.
_GATE_AGENTS = {
    "gate_problem": "problem_detection",
    "gate_validation": "data_validation",
    "gate_cleaning": "cleaning_plan",
    "gate_transformation": "transformation",
    "gate_split": "train_test_split",
    "gate_algorithm": "algorithm_recommendation",
    "gate_recommendation": "recommendation",
}


def _config(pipeline_run_id: str) -> dict:
    return {"configurable": {"thread_id": pipeline_run_id}}


def start(pipeline_run_id: str) -> None:
    compiled_graph.invoke({"pipeline_run_id": pipeline_run_id}, _config(pipeline_run_id))


def _resume(pipeline_run_id: str) -> None:
    compiled_graph.invoke(None, _config(pipeline_run_id))


def advance_after_problem_approval(pipeline_run_id: str) -> None:
    _resume(pipeline_run_id)


def advance_after_validation_approval(pipeline_run_id: str) -> None:
    _resume(pipeline_run_id)


def advance_after_cleaning_approval(pipeline_run_id: str) -> None:
    _resume(pipeline_run_id)


def advance_after_transformation_approval(pipeline_run_id: str) -> None:
    _resume(pipeline_run_id)


def advance_after_split_approval(pipeline_run_id: str) -> None:
    _resume(pipeline_run_id)


def advance_after_algorithm_approval(pipeline_run_id: str) -> None:
    _resume(pipeline_run_id)


def finalize_after_recommendation_approval(pipeline_run_id: str) -> None:
    _resume(pipeline_run_id)


def revise_after_rejection(pipeline_run_id: str) -> None:
    """Resumes the paused gate; the gate sees the rejected decision and routes back to the
    propose node that made it (see `pipeline_graph._make_revisable_gate`)."""
    _resume(pipeline_run_id)


def recover_stalled_runs() -> list[str]:
    """Resumes runs a server restart left stranded, and returns their ids.

    Resuming is queued on an in-memory worker pool (task_queue_service), so a restart between a
    reviewer's approval and the next stage finishing loses that work while the run still reads
    "awaiting_*_approval". The LangGraph checkpoint knows exactly where each run stopped:
    * paused at a review gate whose decision is still `proposed` -> genuinely waiting on a human, left alone;
    * paused at a gate whose decision was already reviewed, or at any non-gate node (interrupted
      mid-stage) -> nothing is running it, so it is resumed from the checkpoint.
    Call once at startup, when no task of this process can still be running any run."""
    resumed: list[str] = []
    db = SessionLocal()
    try:
        runs = db.query(PipelineRunORM).filter(PipelineRunORM.status.notin_(("completed", "failed"))).all()
        for run in runs:
            try:
                state = compiled_graph.get_state(_config(run.id))
                if not state.next:
                    continue  # finished, or never checkpointed
                node = state.next[0]
                if node in _GATE_AGENTS:
                    decision = agent_decision_service.latest_decision(db, run.id, _GATE_AGENTS[node])
                    if decision is None or decision.status == "proposed":
                        continue
                task_queue_service.submit(_resume, run.id, key=run.id)
                resumed.append(run.id)
                logger.warning("Resuming stalled pipeline run %s from node %r (status %s).", run.id, node, run.status)
            except Exception:  # noqa: BLE001 — one bad run must not block startup or the others
                logger.exception("Could not inspect pipeline run %s for recovery.", run.id)
    finally:
        db.close()
    return resumed

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

from app.agents.pipeline_graph import compiled_graph


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

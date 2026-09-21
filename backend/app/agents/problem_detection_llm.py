"""Problem Detection Agent — LLM investigator with a deterministic safety net (Phase 3).

Wraps `problem_detection_agent.detect` (kept verbatim as the fallback). When no target was
declared by the user and the heuristics produced ranked candidates, an LLM may *investigate*
with read-only tools (`column_stats`, `value_counts`, `leakage_probe`) and pick the target —
useful where name/cardinality heuristics are weak (e.g. a target called "col_17").

Guardrails — the model can never make the run unsafe:
  - a user-declared target is authoritative and never sent to the LLM;
  - the chosen target must be one of the heuristic candidates (or null => clustering);
  - classification-vs-regression is always recomputed from the column's real dtype/
    cardinality, never taken from the model;
  - any failure (no provider, no tool-use adapter, bad JSON, invalid pick, exhausted tool
    budget) returns the deterministic decision, annotated with why.
"""
from __future__ import annotations

import json

import pandas as pd

from app.agents import dataset_tools, problem_detection_agent
from app.services import llm_service

_SYSTEM = (
    "You are the problem-detection agent of an AutoML pipeline. Decide which column, if any, "
    "is the prediction target. Use the tools to inspect candidate columns before deciding — "
    "a good target is an outcome (not an identifier or an input feature) and is NOT trivially "
    "predictable from a single other column. You may only choose from `candidates`, or choose "
    "null to indicate there is no target (unsupervised clustering).\n\n"
    'When done, reply with ONLY one JSON object: {"target_column": "<name>"|null, '
    '"confidence": <0.0-1.0>, "rationale": "<why>"}. No markdown, no other text.'
)


def _with_fallback(base: dict, reason: str) -> dict:
    return {**base, "source": "deterministic", "fallback_reason": reason}


def detect(
    df: pd.DataFrame,
    profile: dict,
    declared_target: str | None = None,
    feedback: list[dict] | None = None,
    learning_type: str = "auto",
) -> dict:
    base = {
        **problem_detection_agent.detect(df, profile, declared_target, feedback, learning_type),
        "revision": len(feedback) if feedback else 0,
    }
    candidates = base.get("target_candidates")
    if not candidates:  # declared target, or nothing target-like — nothing for an LLM to decide
        return _with_fallback(base, "not_applicable")

    names = [c["column"] for c in candidates]
    user = json.dumps({
        "candidates": candidates[:10],
        "heuristic_choice": base["target_column"],
        "n_rows": len(df),
        "rejected_by_human": [
            {"target": f["proposal"].get("target_column"), "reason": f.get("reason")} for f in feedback or []
        ],
    }, default=str)

    try:
        result = llm_service.run_tool_loop(_SYSTEM, user, dataset_tools.build_tools(df, base["target_column"]), max_steps=6)
        parsed = llm_service.parse_json_object(result.text)
    except llm_service.ToolLoopUnavailable as exc:
        return _with_fallback(base, f"llm_unavailable: {exc}")
    except Exception as exc:  # noqa: BLE001
        return _with_fallback(base, f"llm_error: {exc}")

    if parsed is None or "target_column" not in parsed:
        return _with_fallback(base, "invalid_json")
    target = parsed["target_column"]
    if target is not None and target not in names:
        return _with_fallback(base, f"llm_chose_non_candidate: {target}")

    try:
        confidence = float(parsed.get("confidence", 0.6))
    except (TypeError, ValueError):
        confidence = 0.6
    rationale = str(parsed.get("rationale", "")).strip()

    if target is None and learning_type == "supervised":
        return _with_fallback(base, "llm_chose_no_target_but_user_wants_supervised")

    if target is None:
        decision = problem_detection_agent._decision(
            "clustering", None, min(max(confidence, 0.0), 0.85),
            [f"LLM investigation found no suitable target: {rationale}"], target_candidates=candidates,
        )
    else:
        problem_type, _c, dtype_reason = problem_detection_agent._classify_target_dtype(df[target])
        decision = problem_detection_agent._decision(
            problem_type, target, min(max(confidence, 0.0), 0.85),
            [f"LLM investigated {len(result.calls)} tool call(s) and chose '{target}': {rationale}", dtype_reason],
            target_candidates=candidates,
        )
    # The LLM's pick is still only a suggestion unless the heuristic already trusted it.
    decision["requires_target_selection"] = target is not None and not (
        base.get("target_column") == target and not base.get("requires_target_selection")
    )
    return {**decision, "source": "llm", "fallback_reason": None, "tool_calls": result.calls,
            "revision": len(feedback) if feedback else 0}


def summarize(result: dict) -> str:
    text = " ".join(result["reasoning"])
    if result.get("source") == "deterministic" and result.get("fallback_reason") not in (None, "not_applicable"):
        text += f" (deterministic fallback — {result['fallback_reason']})"
    return text

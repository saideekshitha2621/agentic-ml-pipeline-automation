"""Transformation Agent — LLM investigator with a deterministic safety net (Phase 3).

Wraps `transformation_agent.propose` (the fallback). The LLM may inspect numeric
distributions with `outlier_report` / `column_stats` and pick the scaler. Guardrails: the
choice must be one of the supported scalers, must not repeat a scaler a human already
rejected, and any failure returns the deterministic proposal.
"""
from __future__ import annotations

import json

import pandas as pd

from app.agents import dataset_tools, transformation_agent
from app.services import llm_service

_SYSTEM = (
    "You are the transformation agent of an AutoML pipeline. Choose the numeric scaling method. "
    "Use the tools to inspect outliers and distributions first. Allowed values for "
    '`scaling_method`: "standard", "robust", "minmax". Do not choose any method listed in '
    "`rejected_by_human`.\n\n"
    'Reply with ONLY one JSON object: {"scaling_method": "...", "confidence": <0.0-1.0>, '
    '"reason": "<one or two sentences citing what the tools showed>"}. No markdown.'
)


def propose(
    df: pd.DataFrame, plan_fields: dict, feedback: list[dict] | None = None,
    problem_type: str | None = None, target_column: str | None = None,
) -> dict:
    base = transformation_agent.propose(df, plan_fields, feedback, problem_type, target_column)
    rejected = sorted(transformation_agent._rejected_scalings(feedback))
    user = json.dumps({
        "numeric_columns": plan_fields["numerical_columns"],
        "heuristic_choice": base["scaling_method"],
        "rejected_by_human": [{"scaling_method": f["proposal"].get("scaling_method"), "reason": f.get("reason")} for f in feedback or []],
    }, default=str)

    def fallback(reason: str) -> dict:
        return {**base, "source": "deterministic", "fallback_reason": reason}

    try:
        result = llm_service.run_tool_loop(_SYSTEM, user, dataset_tools.build_tools(df), max_steps=5)
        parsed = llm_service.parse_json_object(result.text)
    except llm_service.ToolLoopUnavailable as exc:
        return fallback(f"llm_unavailable: {exc}")
    except Exception as exc:  # noqa: BLE001
        return fallback(f"llm_error: {exc}")

    method = (parsed or {}).get("scaling_method")
    if method not in transformation_agent.SCALING_ALTERNATIVES or method in rejected:
        return fallback(f"invalid_or_rejected_choice: {method}")
    try:
        llm_conf = float(parsed.get("confidence", 0.7))
    except (TypeError, ValueError):
        llm_conf = 0.7

    reason = str(parsed.get("reason", "")).strip() or transformation_agent._SCALING_LABELS[method]
    return {
        **base,
        "scaling_method": method,
        "scaling_reason": f"{transformation_agent._SCALING_LABELS[method]} — {reason}",
        # never more confident than the data-driven estimate: the LLM can lower it, not raise it
        "confidence": round(min(base["confidence"], max(llm_conf, 0.0)), 2),
        "source": "llm", "fallback_reason": None, "tool_calls": result.calls,
    }


summarize = transformation_agent.summarize

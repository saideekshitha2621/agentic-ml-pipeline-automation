"""Algorithm Recommendation Agent — LLM-driven, with a deterministic fallback.

Replaces `algorithm_shortlist_agent`'s row-count-only heuristic with an LLM call that
reasons over the dataset's actual profile, data-validation findings, class balance, and
leakage/correlation signals. `algorithm_shortlist_agent` is kept as-is and used verbatim as
the fallback — every failure mode below (no provider configured, a malformed/empty/API-
failed response, a response naming only unsupported algorithms, or the model self-reporting
low confidence) routes straight to it, so this stage can never block or crash the pipeline on
an LLM problem.

"Tools" here are plain Python functions gathering real computed data (dataset shape, feature
types, missingness, validation checks, class distribution, leakage columns, the closed set of
registered algorithm plugins) — all invoked upfront by this module, not autonomously chosen
by the model mid-conversation. This keeps the contract to "one response to validate" rather
than an open-ended tool-calling loop, which is what makes strict validation + fallback
tractable. See the LangGraph conversion plan for why: `llm_service.call()` is a single-shot
text completion wrapper with no multi-turn tool-use loop across any of its three providers.

This module never touches a DB session, a training service, or pipeline state — it takes
plain data in and returns a plain dict out. It's called from exactly the point in
`pipeline_graph.py`'s `algorithm_shortlist_node` where the old deterministic call was, so its
output still flows through the unmodified `_decide()` -> `approval_policy_service` -> HITL
gate path like every other stage.
"""
from __future__ import annotations

import json
import re

import pandas as pd

from app.agents import algorithm_shortlist_agent
from app.plugins.registry import CLASSIFICATION_PLUGIN_REGISTRY, PLUGIN_REGISTRY, REGRESSION_PLUGIN_REGISTRY
from app.services import llm_service

_SYSTEM_PROMPT = (
    "You are an algorithm-selection assistant inside an AutoML pipeline. You choose which "
    "already-registered algorithm plugins to shortlist for training on a dataset — you do "
    "not execute code, do not modify any data, and have no access to anything beyond the "
    "information given to you in this message.\n\n"
    "Respond with ONLY a single JSON object — no markdown code fences, no prose before or "
    "after it — matching exactly this shape:\n"
    "{\n"
    '  "shortlist": [\n'
    '    {"algorithm": "<name>", "recommended": true|false, "rationale": "<why>", "risks": ["<risk>", ...]}\n'
    "  ],\n"
    '  "selected_algorithms": ["<name>", ...],\n'
    '  "confidence": "high"|"medium"|"low"\n'
    "}\n\n"
    "`available_algorithms` in the user message is the ONLY set of algorithm names you may "
    "use in `shortlist[].algorithm` — naming anything else is invalid and will be discarded. "
    "Include every available algorithm in `shortlist` (recommended or not), explain why each "
    "recommended algorithm suits this specific dataset and why each non-recommended one "
    "doesn't, and note concrete risks or limitations (e.g. slow on this many rows, sensitive "
    "to the missingness/outliers/imbalance found, small-sample overfitting risk)."
)

_REGISTRY_BY_PROBLEM_TYPE = {
    "classification": CLASSIFICATION_PLUGIN_REGISTRY,
    "regression": REGRESSION_PLUGIN_REGISTRY,
    "clustering": PLUGIN_REGISTRY,
}
_RATIONALE_BY_PROBLEM_TYPE = {
    "classification": algorithm_shortlist_agent._CLASSIFICATION_RATIONALE,
    "regression": algorithm_shortlist_agent._REGRESSION_RATIONALE,
    "clustering": algorithm_shortlist_agent._CLUSTERING_RATIONALE,
}

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


# --- Tool functions: each gathers one real, precomputed fact about the dataset -----------

def _dataset_shape(profile: dict) -> dict:
    du = profile.get("dataset_understanding", {})
    return {"n_rows": du.get("n_rows", profile.get("n_rows", 0)), "n_columns": du.get("n_columns", profile.get("n_columns", 0))}


def _feature_types(profile: dict) -> dict:
    roles = profile.get("column_roles", {})
    counts: dict[str, int] = {}
    for meta in roles.values():
        counts[meta["role"]] = counts.get(meta["role"], 0) + 1
    return counts


def _missing_summary(profile: dict) -> dict:
    du = profile.get("dataset_understanding", {})
    return {
        "n_missing_values_total": du.get("n_missing_values", 0),
        "missing_by_column": du.get("missing_by_column", {}),
    }


def _validation_findings(validation: dict | None) -> list[dict]:
    if not validation:
        return []
    return [
        {"name": c["name"], "status": c["status"], "detail": c["detail"]}
        for c in validation.get("checks", [])
    ]


def _leakage_and_correlation(validation: dict | None) -> list[str]:
    if not validation:
        return []
    leakage_check = next((c for c in validation.get("checks", []) if c["name"] == "Data leakage risk"), None)
    return leakage_check["affected_columns"] if leakage_check else []


def _class_distribution(df: pd.DataFrame | None, target_column: str | None, problem_type: str) -> dict | None:
    if problem_type != "classification" or df is None or not target_column or target_column not in df.columns:
        return None
    return {str(k): int(v) for k, v in df[target_column].value_counts(dropna=True).items()}


def _available_algorithms(problem_type: str) -> list[dict]:
    registry = _REGISTRY_BY_PROBLEM_TYPE.get(problem_type, PLUGIN_REGISTRY)
    rationale_map = _RATIONALE_BY_PROBLEM_TYPE.get(problem_type, {})
    return [
        {"name": name, "description": rationale_map.get(name, "Registered candidate algorithm for this problem type.")}
        for name in registry
    ]


def _gather_tool_outputs(profile: dict, validation: dict | None, problem_type: str, df: pd.DataFrame | None, target_column: str | None) -> dict:
    return {
        "dataset_shape": _dataset_shape(profile),
        "feature_types": _feature_types(profile),
        "missing_summary": _missing_summary(profile),
        "validation_findings": _validation_findings(validation),
        "class_distribution": _class_distribution(df, target_column, problem_type),
        "leakage_correlation_columns": _leakage_and_correlation(validation),
        "available_algorithms": _available_algorithms(problem_type),
    }


# --- Prompt + parsing/validation -----------------------------------------------------------

def _build_user_message(tool_outputs: dict, problem_type: str) -> str:
    return (
        f"Problem type: {problem_type}\n\n"
        f"Dataset and validation information (from tool calls already run against this dataset):\n"
        f"{json.dumps(tool_outputs, indent=2, default=str)}\n\n"
        "Analyze this dataset and recommend which of the available algorithms to shortlist "
        "for training, with a rationale for each recommended and excluded algorithm and any "
        "risks or limitations you see."
    )


def _parse_and_validate(raw_text: str, valid_names: set[str]) -> tuple[dict | None, str | None]:
    if not raw_text or not raw_text.strip():
        return None, "empty_or_malformed_response"

    cleaned = _FENCE_RE.sub("", raw_text.strip()).strip()
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        return None, "invalid_json"

    if not isinstance(parsed, dict) or not isinstance(parsed.get("shortlist"), list) or not parsed["shortlist"]:
        return None, "empty_or_malformed_response"

    filtered_shortlist = [
        entry for entry in parsed["shortlist"]
        if isinstance(entry, dict) and entry.get("algorithm") in valid_names
    ]
    selected_algorithms = [e["algorithm"] for e in filtered_shortlist if e.get("recommended")]
    if not selected_algorithms:
        return None, "no_valid_algorithms_in_response"

    if str(parsed.get("confidence", "")).lower() == "low":
        return None, "low_confidence_response"

    return {
        "shortlist": [
            {
                "algorithm": e["algorithm"],
                "recommended": bool(e.get("recommended")),
                "rationale": str(e.get("rationale", "")),
                "risks": [str(r) for r in e.get("risks", [])] if isinstance(e.get("risks"), list) else [],
            }
            for e in filtered_shortlist
        ],
        "selected_algorithms": selected_algorithms,
        "llm_reasoning": str(parsed.get("reasoning", "")) or None,
    }, None


# --- Public API ------------------------------------------------------------------------

def _fallback(profile: dict, problem_type: str, tool_outputs: dict, reason: str) -> dict:
    base = algorithm_shortlist_agent.recommend(profile, problem_type)
    return {
        **base,
        "source": "fallback",
        "llm_reasoning": None,
        "tool_outputs_used": list(tool_outputs.keys()),
        "fallback_reason": reason,
    }


def recommend(profile: dict, validation: dict | None, problem_type: str, df: pd.DataFrame | None, target_column: str | None) -> dict:
    tool_outputs = _gather_tool_outputs(profile, validation, problem_type, df, target_column)
    valid_names = set(_REGISTRY_BY_PROBLEM_TYPE.get(problem_type, PLUGIN_REGISTRY))

    if not llm_service.is_configured():
        return _fallback(profile, problem_type, tool_outputs, "llm_not_configured")

    config = llm_service._active_config()
    try:
        raw = llm_service.call(
            config["provider"], config["api_key"], config["model"], _SYSTEM_PROMPT,
            [{"role": "user", "content": _build_user_message(tool_outputs, problem_type)}],
            max_tokens=1500,
        )
    except Exception as exc:  # noqa: BLE001
        return _fallback(profile, problem_type, tool_outputs, f"llm_api_error: {exc}")

    parsed, reason = _parse_and_validate(raw, valid_names)
    if parsed is None:
        return _fallback(profile, problem_type, tool_outputs, reason)

    return {
        **parsed,
        "source": "llm",
        "tool_outputs_used": list(tool_outputs.keys()),
        "fallback_reason": None,
    }


def summarize(result: dict) -> str:
    n_selected = len(result["selected_algorithms"])
    n_total = len(result["shortlist"])
    if result.get("source") == "llm":
        return f"LLM-recommended {n_selected} of {n_total} available algorithm(s) after analyzing the dataset profile and validation findings."
    reason = result.get("fallback_reason", "unknown")
    return (
        f"Recommending {n_selected} of {n_total} registered algorithm(s) based on dataset size and type "
        f"(deterministic fallback — {reason})."
    )

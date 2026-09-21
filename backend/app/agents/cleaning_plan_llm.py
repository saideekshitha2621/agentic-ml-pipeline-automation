"""Cleaning Plan Agent — LLM refinement with a deterministic safety net (Phase 3).

Wraps `cleaning_plan_agent.propose` (the fallback and the source of truth for which columns
are dropped or protected). The LLM gets each column's real statistics (missing %, skew,
cardinality) in ONE request — no tool loop, to respect small API quotas — and may only
*re-choose the imputation strategy* for columns that already need imputing.

Guardrails — the model can never make the plan riskier than the rules did:
  - it cannot drop a column, keep a dropped one, or touch no-information / leakage /
    identifier / constant rows (those aren't in the editable set at all);
  - each override must name an editable column and use an action valid for that column's type;
  - the plan's confidence can only be lowered by the LLM, never raised;
  - any failure (no provider, quota/rate limit, bad JSON, nothing valid) returns the
    deterministic plan, annotated with why.
"""
from __future__ import annotations

import json

import pandas as pd

from app.agents import cleaning_plan_agent
from app.services import llm_service

ALLOWED_ACTIONS = {"numeric": {"mean", "median", "fill_zero"}, "categorical": {"mode", "fill_zero"}}
_EDITABLE_ACTIONS = {"mean", "median", "mode", "fill_zero"}

_SYSTEM = (
    "You are the data-cleaning agent of an AutoML pipeline. For each column listed, choose the "
    "best strategy for filling its missing values, using the statistics given (missing %, skew, "
    "cardinality). Only choose from that column's `allowed` actions. Prefer the median for "
    "skewed numeric columns, the mean for roughly symmetric ones, and the mode for categories; "
    "use fill_zero only when a missing value plausibly means 'none/zero'. Change a column only "
    "if the current choice is clearly wrong.\n\n"
    'Reply with ONLY one JSON object: {"overrides": [{"column": "...", "action": "...", '
    '"reason": "<short, cites the statistic>"}], "confidence": <0.0-1.0>}. Use an empty '
    '"overrides" list if the current plan is fine. No markdown.'
)


def propose(
    df: pd.DataFrame, validation_result: dict, target_column: str | None = None, feedback: list[dict] | None = None
) -> dict:
    base = cleaning_plan_agent.propose(df, validation_result, target_column, feedback)

    def fallback(reason: str) -> dict:
        return {**base, "source": "deterministic", "fallback_reason": reason, "llm_overrides": []}

    editable = [r for r in base["recommendations"] if r["action"] in _EDITABLE_ACTIONS and r.get("missing_count", 0) > 0]
    if not editable:
        return fallback("nothing_to_refine")

    rows = []
    for r in editable:
        row = {
            "column": r["column"], "type": r["column_type"], "missing_pct": r["missing_pct"],
            "n_unique": int(df[r["column"]].nunique(dropna=True)),
            "current_action": r["action"], "allowed": sorted(ALLOWED_ACTIONS[r["column_type"]]),
        }
        if r["column_type"] == "numeric" and df[r["column"]].notna().sum() > 2:
            row["skew"] = round(float(df[r["column"]].skew()), 2)
        rows.append(row)
    user = json.dumps({
        "columns": rows,
        "rejected_by_human": [f.get("reason") for f in feedback or [] if f.get("reason")],
    }, default=str)

    try:
        parsed = llm_service.parse_json_object(llm_service.complete(_SYSTEM, user, max_tokens=1500))
    except llm_service.ToolLoopUnavailable as exc:
        return fallback(f"llm_unavailable: {exc}")
    except Exception as exc:  # noqa: BLE001
        return fallback(f"llm_error: {exc}")
    if parsed is None or not isinstance(parsed.get("overrides"), list):
        return fallback("invalid_json")

    by_col = {r["column"]: r for r in editable}
    applied: list[dict] = []
    new_recs = []
    overrides = {o.get("column"): o for o in parsed["overrides"] if isinstance(o, dict)}
    for rec in base["recommendations"]:
        o = overrides.get(rec["column"])
        if o and rec["column"] in by_col and o.get("action") in ALLOWED_ACTIONS[rec["column_type"]] and o["action"] != rec["action"]:
            reason = str(o.get("reason", "")).strip() or "chosen from the column's statistics"
            applied.append({"column": rec["column"], "from": rec["action"], "to": o["action"], "reason": reason})
            rec = {**rec, "action": o["action"], "reason": f"{o['action']} — {reason} (LLM-refined)"}
        new_recs.append(rec)

    try:
        llm_conf = float(parsed.get("confidence", 0.8))
    except (TypeError, ValueError):
        llm_conf = 0.8
    return {
        **base, "recommendations": new_recs, "source": "llm", "fallback_reason": None,
        "llm_overrides": applied, "llm_confidence": round(min(max(llm_conf, 0.0), 1.0), 2),
    }


summarize = cleaning_plan_agent.summarize
estimate_confidence = cleaning_plan_agent.estimate_confidence

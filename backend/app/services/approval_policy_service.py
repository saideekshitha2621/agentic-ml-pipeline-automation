"""Decides whether a stage's proposed decision can auto-proceed or must pause for a human.

Only two stages are ALWAYS human-gated regardless of confidence — problem type/target
confirmation and the final model recommendation — because they carry outsized, hard-to-undo
business consequences. Every other stage auto-approves when the agent's own confidence is
high AND no critical risk signal is present in what it just proposed; otherwise it pauses
and records exactly why, so `AgentDecision.decision_json["approval_reason"]` doubles as the
audit trail for every auto-approved (and every escalated) decision.
"""
from __future__ import annotations

CONFIDENCE_THRESHOLD = 0.75

# 100%-missing / high-severity missingness inside the Cleaning Plan stage.
CLEANING_TARGET_MISSING_ESCALATE_PCT = 20.0


def _pct(value: float | None) -> str:
    return f"{round((value or 0) * 100)}%"


def _data_validation_risk(decision: dict) -> str | None:
    critical = [c["name"] for c in decision.get("checks", []) if c.get("status") == "critical"]
    if critical:
        return f"critical data quality issue(s) detected: {', '.join(critical)}."
    return None


def _cleaning_plan_risk(decision: dict) -> str | None:
    recs = decision.get("recommendations", [])
    no_info_cols = [r["column"] for r in recs if r.get("no_information")]
    if no_info_cols:
        return f"column(s) with no usable information (100% missing): {', '.join(no_info_cols)}."
    # A column flagged for dropping specifically because of leakage risk is the moment the
    # *action* (discard a potentially strong, legitimate predictor) actually happens — the
    # Data Validation stage already flagged the correlation itself, but the human should
    # still get a say on whether dropping it is the right call, since a false-positive
    # leakage flag (a feature that's just very predictive, not actually leaked) silently
    # dropped here can badly hurt model quality with no further checkpoint downstream.
    leakage_drop_cols = [r["column"] for r in recs if r["action"] == "drop_column" and "leakage" in r.get("issue", "")]
    if leakage_drop_cols:
        return f"column(s) recommended for removal due to potential data leakage: {', '.join(leakage_drop_cols)}."
    target_missing = decision.get("target_missing")
    if target_missing and (target_missing.get("missing_pct") or 0) > CLEANING_TARGET_MISSING_ESCALATE_PCT:
        return f"a large share of rows ({target_missing['missing_pct']}%) are missing a target value."
    return None


_RISK_DETECTORS = {
    "data_validation": _data_validation_risk,
    "cleaning_plan": _cleaning_plan_risk,
}


def decide(stage: str, decision: dict, confidence: float | None) -> tuple[bool, str]:
    """Returns (auto_approve, reason) for a stage that's eligible for auto-approval — i.e.
    every stage except problem_detection and recommendation, which the orchestrator always
    forces to auto_approve=False before this is ever consulted."""
    if confidence is None or confidence < CONFIDENCE_THRESHOLD:
        return False, (
            f"Requires review — the agent's confidence ({_pct(confidence)}) is below the "
            f"{_pct(CONFIDENCE_THRESHOLD)} auto-approval threshold."
        )

    detector = _RISK_DETECTORS.get(stage)
    risk = detector(decision) if detector else None
    if risk:
        return False, f"Requires review — {risk}"

    return True, f"Auto-approved — confidence {_pct(confidence)} with no critical risk detected."


def mandatory_reason() -> str:
    return "Always requires human review — this decision has significant, hard-to-reverse business impact."

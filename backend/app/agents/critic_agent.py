"""Critic Agent (Phase 2).

An independent reviewer that reads the *finished* work of the other agents — training
outcome, the self-correction history, validation findings, the recommendation itself — and
raises concerns before the mandatory human gate. It never changes a decision; it makes the
reviewer's job better informed. Findings are deterministic and auditable; the narrative
wrapper may be LLM-polished (`llm_service.explain`) but falls back to plain text.
"""
from __future__ import annotations

from app.services import llm_service

SMALL_DATASET_ROWS = 200


def review(
    *,
    problem_type: str,
    recommendation: dict,
    quality: dict | None,
    validation: dict | None,
    cleaning: dict | None,
    n_rows: int,
    retry_remedies: list[str],
) -> dict:
    findings: list[dict] = []

    def add(severity: str, code: str, message: str) -> None:
        findings.append({"severity": severity, "code": code, "message": message})

    flags = (quality or {}).get("flags", [])
    if "suspected_leakage" in flags:
        add("high", "suspected_leakage",
            f"The best model's {quality['key_metric']} ({quality['value']:.3f}) is near-perfect. "
            "Real-world data rarely allows this — check for a feature that leaks the answer before trusting the model.")
    if "low_performance_unresolved" in flags:
        add("high", "low_performance",
            f"Even after {len(retry_remedies)} automatic self-correction attempt(s), {quality['key_metric']} "
            f"is {quality['value']:.3f} (bar: {quality['threshold']:.2f}). The data may not contain enough signal for this target.")
    if retry_remedies:
        add("info", "self_corrected",
            f"The pipeline self-corrected {len(retry_remedies)} time(s) before reaching this result: {', '.join(retry_remedies)}.")
    if recommendation.get("confidence") == "low":
        add("medium", "near_tie",
            "The top two models are nearly tied — the ranking is not strong evidence that #1 is genuinely better.")
    if n_rows < SMALL_DATASET_ROWS:
        add("medium", "small_dataset",
            f"Only {n_rows} rows — metrics from a small evaluation set are noisy; treat them as indicative.")
    flagged = [c["name"] for c in (validation or {}).get("checks", []) if c.get("status") in ("warning", "critical")]
    if flagged:
        add("medium", "data_quality", f"Data-quality checks still flagged at validation: {', '.join(flagged)}.")
    leak_drops = [r["column"] for r in (cleaning or {}).get("recommendations", [])
                  if r.get("action") == "drop_column" and "leakage" in r.get("issue", "")]
    if leak_drops:
        add("info", "leakage_columns_dropped",
            f"Column(s) removed for leakage risk: {', '.join(leak_drops)} — confirm none was a legitimate predictor.")

    severities = {f["severity"] for f in findings}
    verdict = "serious_concerns" if "high" in severities else "concerns" if "medium" in severities else "no_concerns"

    fallback = (
        "No concerns — the champion passed every automated check." if not findings
        else " ".join(f["message"] for f in findings if f["severity"] in ("high", "medium"))
        or " ".join(f["message"] for f in findings)
    )
    narrative = llm_service.explain("critic_review", {"verdict": verdict, "findings": findings}, fallback)
    return {"verdict": verdict, "findings": findings, "narrative": narrative}

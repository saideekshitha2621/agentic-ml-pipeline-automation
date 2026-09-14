"""Problem Detection Agent.

Deterministic, rule-based classification vs. regression vs. clustering decision — no
ML-on-ML. Ambiguous or unconfirmed cases are flagged `requires_review` rather than
silently guessed; the pipeline always confirms this decision at a HITL gate regardless,
but the flag tells the reviewing UI how much scrutiny to suggest.

Decision order:
  1. User-declared target (explicit intent) -> highest confidence.
  2. No declared target -> look for a *weak* structural signal (a non-id/free-text/
     datetime column in the last position -> possible label). Weak signals are always
     capped at moderate confidence and flagged for review.
  3. No plausible target at all -> clustering (the well-tested default path).
"""
from __future__ import annotations

import pandas as pd

REGRESSION_MIN_DISTINCT = 20
REGRESSION_MIN_DISTINCT_RATIO = 0.05
WEAK_SIGNAL_CONFIDENCE_CAP = 0.55
LOW_CONFIDENCE_THRESHOLD = 0.6


def _classify_target_dtype(series: pd.Series, role: str) -> tuple[str, float, str]:
    n = int(series.nunique(dropna=True))
    n_rows = len(series) or 1
    if role == "numeric":
        ratio = n / n_rows
        if n > REGRESSION_MIN_DISTINCT and ratio > REGRESSION_MIN_DISTINCT_RATIO:
            return (
                "regression",
                0.75,
                f"Numeric with {n} distinct values ({ratio:.0%} of rows) — looks continuous.",
            )
        return (
            "classification",
            0.65,
            f"Numeric but only {n} distinct value(s) — looks like a coded/ordinal label.",
        )
    return "classification", 0.8, f"Categorical/boolean target with {n} distinct value(s)."


def _decision(problem_type: str, target_column: str | None, confidence: float, reasoning: list[str]) -> dict:
    return {
        "problem_type": problem_type,
        "target_column": target_column,
        "confidence": round(confidence, 2),
        "reasoning": reasoning,
        "requires_review": confidence < LOW_CONFIDENCE_THRESHOLD,
    }


def detect(df: pd.DataFrame, profile: dict, declared_target: str | None = None) -> dict:
    roles = profile.get("column_roles", {})
    columns = list(df.columns)

    if declared_target:
        if declared_target not in columns:
            return _decision(
                "clustering",
                None,
                0.3,
                [f"Declared target '{declared_target}' was not found in the dataset — falling back to clustering."],
            )
        role_meta = roles.get(declared_target, {"role": "numeric"})
        problem_type, confidence, reason = _classify_target_dtype(df[declared_target], role_meta["role"])
        return _decision(
            problem_type,
            declared_target,
            max(confidence, 0.9),
            [f"User declared '{declared_target}' as the prediction target.", reason],
        )

    candidates = [c for c in columns if roles.get(c, {}).get("role") not in ("id_like", "free_text", "datetime")]
    if not candidates:
        return _decision(
            "clustering",
            None,
            0.9,
            ["No target-like column found — every column looks like an identifier, free text, or a timestamp."],
        )

    last_col = columns[-1]
    if last_col not in candidates:
        return _decision(
            "clustering",
            None,
            0.85,
            ["No plausible target column detected (no non-identifier column in a conventional label position)."],
        )

    role_meta = roles[last_col]
    problem_type, base_confidence, reason = _classify_target_dtype(df[last_col], role_meta["role"])
    confidence = min(base_confidence, WEAK_SIGNAL_CONFIDENCE_CAP)
    return _decision(
        problem_type,
        last_col,
        confidence,
        [
            f"No explicit target declared — '{last_col}' is the last column and not id-like/free-text/"
            "datetime, which is a weak signal it may be a label.",
            reason,
            "This is a low-confidence structural guess; please confirm or override.",
        ],
    )

"""Problem Detection Agent.

Deterministic, rule-based classification vs. regression vs. clustering decision — no
ML-on-ML. Ambiguous or unconfirmed cases are flagged `requires_review` rather than
silently guessed; the pipeline always confirms this decision at a HITL gate regardless,
but the flag tells the reviewing UI how much scrutiny to suggest.

Decision order:
  1. User-declared target (explicit intent) -> highest confidence. Unchanged: still a
     direct dtype/cardinality read of that one column, no ranking involved.
  2. No declared target -> every non-identifier/free-text/datetime column is scored as a
     target candidate (name pattern, cardinality, dtype, position) and ranked. The
     top-ranked candidate becomes the proposed target, but the full ranked list travels
     in `target_candidates` so the HITL review step can offer the alternatives instead of
     silently trusting a single structural guess.
  3. No plausible target at all -> clustering (the well-tested default path).
"""
from __future__ import annotations

import re

import pandas as pd

REGRESSION_MIN_DISTINCT = 20
REGRESSION_MIN_DISTINCT_RATIO = 0.05
# A column with this many distinct values is treated as continuous even if that count is
# under the 5% ratio, which happens on large datasets for naturally-quantized continuous
# values (e.g. body weight in kg to 1 decimal across 55k rows) that the ratio alone would
# otherwise misread as a "coded label" purely because the row count is big.
REGRESSION_MIN_DISTINCT_ABS = 200
WEAK_SIGNAL_CONFIDENCE_CAP = 0.55
LOW_CONFIDENCE_THRESHOLD = 0.6

_STRONG_TARGET_NAMES = {"target", "label", "y", "outcome", "class", "result"}
_CLASSIFICATION_NAME_HINTS = [
    "churn", "purchased", "purchase", "fraud", "default", "approved", "approval",
    "convert", "response", "status", "flag", "is_", "has_", "success", "failure",
    "dangerous", "risk", "outcome", "diagnosis", "attrition",
]
_REGRESSION_NAME_HINTS = [
    "price", "cost", "amount", "revenue", "sales", "salary", "income", "score",
    "rating", "value", "total", "charge", "fee", "duration", "demand", "profit",
]
_ID_NAME_PATTERN = re.compile(r"(^id$|_id$|^id_|uuid|guid|^key$|_key$|^code$|_code$)", re.IGNORECASE)


def _classify_target_dtype(series: pd.Series) -> tuple[str, float, str]:
    """Decides classification vs. regression from the target's actual dtype/cardinality —
    deliberately independent of the profiler's coarse column "role" tag. That tag's
    `id_like` bucket exists to exclude high-uniqueness *feature* columns (customer_id,
    order_id) from modeling; a continuous regression target (price, revenue) is *also*
    high-uniqueness by nature, so reusing that tag here used to misclassify a legitimate
    regression target as "id_like" -> categorical -> classification, which then crashed
    downstream (stratifying a train/test split on ~400 near-unique float values)."""
    n = int(series.nunique(dropna=True))
    n_rows = len(series) or 1
    if pd.api.types.is_numeric_dtype(series):
        ratio = n / n_rows
        if n > REGRESSION_MIN_DISTINCT and (ratio > REGRESSION_MIN_DISTINCT_RATIO or n >= REGRESSION_MIN_DISTINCT_ABS):
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


def _name_pattern_score(column: str) -> tuple[float, str | None]:
    normalized = column.strip().lower().replace(" ", "_")
    if _ID_NAME_PATTERN.search(normalized):
        return -0.3, f"Column name '{column}' looks like an identifier (id/key/code/uuid pattern)."
    if normalized in _STRONG_TARGET_NAMES:
        return 0.35, f"Column name '{column}' is a conventional target-name keyword."
    for kw in _CLASSIFICATION_NAME_HINTS:
        if kw in normalized:
            return 0.25, f"Column name contains '{kw}', a common outcome/label keyword."
    for kw in _REGRESSION_NAME_HINTS:
        if kw in normalized:
            return 0.2, f"Column name contains '{kw}', a common numeric-outcome keyword."
    return 0.0, None


def _cardinality_score(series: pd.Series) -> tuple[float, str]:
    n = int(series.nunique(dropna=True))
    n_rows = len(series) or 1
    if n < 2:
        return 0.0, f"Only {n} distinct value(s) — constant column, not a usable target."
    if pd.api.types.is_numeric_dtype(series):
        ratio = n / n_rows
        if n > REGRESSION_MIN_DISTINCT and (ratio > REGRESSION_MIN_DISTINCT_RATIO or n >= REGRESSION_MIN_DISTINCT_ABS):
            return 0.3, f"{n} distinct numeric values ({ratio:.0%} of rows) — looks continuous, a plausible regression target."
        if 2 <= n <= 20:
            return 0.25, f"{n} distinct numeric values — looks like a coded/ordinal label, a plausible classification target."
        return 0.05, f"{n} distinct numeric values — doesn't cleanly fit a label or a continuous target."
    if 2 <= n <= 20:
        return 0.3, f"{n} distinct categories — typical label cardinality for classification."
    return 0.1, f"{n} distinct categories — high-cardinality categorical, a weaker classification-target signal."


def _target_candidate_score(column: str, series: pd.Series, position_bonus: float) -> dict:
    name_score, name_reason = _name_pattern_score(column)
    card_score, card_reason = _cardinality_score(series)
    score = round(name_score + card_score + position_bonus, 3)
    problem_type, _confidence, _dtype_reason = _classify_target_dtype(series)
    reasoning = [r for r in (name_reason, card_reason) if r]
    return {
        "column": column,
        "score": score,
        "problem_type": problem_type,
        "reasoning": reasoning,
    }


def _rank_target_candidates(df: pd.DataFrame, candidates: list[str], columns: list[str]) -> list[dict]:
    last_col = columns[-1] if columns else None
    first_col = columns[0] if columns else None
    ranked = []
    for col in candidates:
        position_bonus = 0.1 if col == last_col else (-0.05 if col == first_col else 0.0)
        ranked.append(_target_candidate_score(col, df[col], position_bonus))
    ranked.sort(key=lambda c: c["score"], reverse=True)
    return ranked


def _decision(
    problem_type: str,
    target_column: str | None,
    confidence: float,
    reasoning: list[str],
    target_candidates: list[dict] | None = None,
) -> dict:
    decision = {
        "problem_type": problem_type,
        "target_column": target_column,
        "confidence": round(confidence, 2),
        "reasoning": reasoning,
        "requires_review": confidence < LOW_CONFIDENCE_THRESHOLD,
    }
    if target_candidates is not None:
        decision["target_candidates"] = target_candidates
    return decision


def detect(
    df: pd.DataFrame, profile: dict, declared_target: str | None = None, feedback: list[dict] | None = None
) -> dict:
    """`feedback` (Phase 1 revision loop): earlier proposals a human rejected. Every
    previously-proposed target is excluded from this pass, so a rejection moves the
    proposal to the next-ranked candidate (or to clustering when none remain) instead of
    re-proposing the same answer."""
    roles = profile.get("column_roles", {})
    columns = list(df.columns)

    rejected_targets = {f["proposal"].get("target_column") for f in feedback or [] if f.get("proposal")}
    rejected_targets.discard(None)
    if declared_target in rejected_targets:
        declared_target = None  # the reviewer already rejected this one
    revision_note = (
        [f"Revised after {len(feedback)} rejection(s) — excluding previously proposed target(s): "
         f"{sorted(rejected_targets) or 'none (clustering was rejected)'}."]
        if feedback else []
    )

    if declared_target:
        if declared_target not in columns:
            return _decision(
                "clustering",
                None,
                0.3,
                [f"Declared target '{declared_target}' was not found in the dataset — falling back to clustering."],
            )
        problem_type, confidence, reason = _classify_target_dtype(df[declared_target])
        return _decision(
            problem_type,
            declared_target,
            max(confidence, 0.9),
            [f"User declared '{declared_target}' as the prediction target.", reason],
        )

    candidates = [
        c for c in columns
        if roles.get(c, {}).get("role") not in ("id_like", "free_text", "datetime") and c not in rejected_targets
    ]
    if not candidates:
        return _decision(
            "clustering",
            None,
            0.9 if not feedback else 0.6,
            [*revision_note, "No target-like column found — every column looks like an identifier, free text, or a timestamp."
             if not feedback else "No remaining target candidates after the rejection — falling back to clustering."],
        )

    ranked = _rank_target_candidates(df, candidates, columns)
    top = ranked[0]
    problem_type, base_confidence, reason = _classify_target_dtype(df[top["column"]])
    confidence = min(base_confidence, WEAK_SIGNAL_CONFIDENCE_CAP)
    return _decision(
        problem_type,
        top["column"],
        confidence,
        [
            *revision_note,
            f"No explicit target declared — ranked {len(ranked)} candidate column(s) by name pattern, "
            f"cardinality, data type, and position; '{top['column']}' scored highest.",
            reason,
            "This is a low-confidence structural guess; please confirm or pick a different candidate below.",
        ],
        target_candidates=ranked,
    )

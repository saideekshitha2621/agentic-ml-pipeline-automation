"""Small helpers that turn a human reviewer's free-text rejection reason into structured
constraints a deterministic agent can act on when it re-proposes (Phase 1 revision loop).

`feedback` everywhere is the list returned by `agent_decision_service.rejection_history`:
`[{"proposal": <rejected decision_json>, "reason": <reviewer text>}, ...]`.
"""
from __future__ import annotations

import re

_EXCLUDE_WORDS = re.compile(r"\b(exclude|remove|drop|without|skip|avoid|no|not|don't|dont)\b", re.IGNORECASE)


def reasons(feedback: list[dict] | None) -> list[str]:
    return [f.get("reason", "") for f in feedback or [] if f.get("reason")]


def mentioned_columns(feedback: list[dict] | None, columns: list[str]) -> set[str]:
    """Columns whose exact name appears (case-insensitive, whole word) in any reason."""
    text = " ".join(reasons(feedback)).lower()
    found = set()
    for col in columns:
        if re.search(rf"(?<![\w]){re.escape(str(col).lower())}(?![\w])", text):
            found.add(col)
    return found


def mentioned_fraction(feedback: list[dict] | None, low: float = 0.1, high: float = 0.5) -> float | None:
    """A test-size the reviewer asked for, e.g. '30%' or '0.3' -> 0.3 (latest reason wins)."""
    for reason in reversed(reasons(feedback)):
        for m in re.finditer(r"(\d+(?:\.\d+)?)\s*%|\b(0?\.\d+)\b", reason):
            value = float(m.group(1)) / 100 if m.group(1) else float(m.group(2))
            if low <= value <= high:
                return round(value, 2)
    return None


def include_exclude_names(feedback: list[dict] | None, names: list[str]) -> tuple[set[str], set[str]]:
    """Clause-level include/exclude split for algorithm names mentioned in the reasons."""
    include: set[str] = set()
    exclude: set[str] = set()
    for reason in reasons(feedback):
        for clause in re.split(r"[.;,\n]", reason):
            lowered = clause.lower()
            hits = {n for n in names if n.lower() in lowered or n.lower().replace("_", " ") in lowered}
            if not hits:
                continue
            (exclude if _EXCLUDE_WORDS.search(lowered) else include).update(hits)
    return include, exclude

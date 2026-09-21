"""One honest, consistent rating of "how good is the recommended model?" for every place the
product shows it (executive summary, final report, PDF).

Previously the UI showed `recommendation.confidence` ("high"/"low") as a quality label, but that
field only says whether the top two models scored within a hair of each other. A near-perfect
model with a runner-up that is equally good therefore read as "low confidence / Moderate" right
next to "100 out of 100 predictions correct". Two different questions were being answered with
one word. They are now separate:

  * `level`   — how good the model's held-out score actually is (Strong / Good / Fair / Weak),
                or "Verify" when it is too perfect to trust without checking for data leakage;
  * `tie_note` — a neutral note when several models performed almost equally (which is *good news*:
                pick on simplicity, speed or cost), never presented as a weakness.
"""
from __future__ import annotations

# problem_type -> (metric key in the evaluation decision, [(minimum, level)] highest first, "too perfect" value)
_SCALES = {
    "classification": ("f1_macro", [(0.85, "Strong"), (0.70, "Good"), (0.60, "Fair")], 0.99),
    "regression": ("r2", [(0.80, "Strong"), (0.60, "Good"), (0.40, "Fair")], 0.99),
    "clustering": ("silhouette", [(0.50, "Strong"), (0.35, "Good"), (0.25, "Fair")], None),
}

_EXPLANATIONS = {
    "Strong": "The model performs strongly on data it has not seen and is suitable for decision support.",
    "Good": "The model performs well on data it has not seen; keep monitoring it once it is in use.",
    "Fair": "The model is usable but has clear room to improve — review where it makes mistakes before relying on it for important decisions.",
    "Weak": "The model is not yet reliable enough for decisions — more or better data, or different features, are needed.",
    "Verify": "The score is unusually perfect. Before trusting it, confirm that no column gives away the answer (data leakage) "
    "and that the test data is genuinely unseen. Perfect results on real data are rare.",
    "Unknown": "A performance rating will be available once the model has been evaluated.",
}

TIE_NOTE = (
    "Several of the top models performed almost equally well, so the final choice can be made on simplicity, speed or cost."
)


def key_metric_name(problem_type: str | None) -> str | None:
    scale = _SCALES.get(problem_type or "")
    return scale[0] if scale else None


def assess(problem_type: str | None, metrics: dict | None, near_tie: bool = False) -> dict:
    """`metrics`: the evaluation decision's metrics for the top model (short clustering keys ok)."""
    scale = _SCALES.get(problem_type or "")
    value = (metrics or {}).get(scale[0]) if scale else None
    if scale is None or not isinstance(value, (int, float)):
        level = "Unknown"
    elif scale[2] is not None and value >= scale[2]:
        level = "Verify"
    else:
        level = next((name for minimum, name in scale[1] if value >= minimum), "Weak")
    return {
        "level": level,
        "explanation": _EXPLANATIONS[level],
        "tie_note": TIE_NOTE if near_tie and level not in ("Unknown", "Weak") else None,
        "metric": scale[0] if scale else None,
        "value": round(float(value), 3) if isinstance(value, (int, float)) else None,
    }

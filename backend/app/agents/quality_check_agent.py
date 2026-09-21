"""Model Quality Check Agent — the closed loop (Phase 2).

After training/HPO/evaluation, inspects the best model's key metric and decides whether the
pipeline should *self-correct* (route back to training with a remedy applied, no human
needed) or proceed to the critic/recommendation gate. Retries are bounded by
`MAX_AUTO_RETRIES` and every attempt is a recorded AgentDecision, so the loop is fully
auditable and the mandatory human recommendation gate still has the final word.

Remedies, tried in order and never repeated:
  1. `broaden_algorithms` — the shortlist may have been trimmed too aggressively; train every
     registered algorithm for this problem type.
  2. `alternate_scaling`  — swap the scaler (standard <-> robust, minmax -> robust), since a
     poor scaler choice silently hurts distance/margin-based models.

A near-perfect score is NOT retried — it's a leakage smell, so it is flagged for the critic
and the human reviewer instead.
"""
from __future__ import annotations

MAX_AUTO_RETRIES = 2
REMEDIES = ["broaden_algorithms", "alternate_scaling"]

# problem_type -> (key metric name, minimum acceptable value, suspiciously-perfect value)
QUALITY_TARGETS = {
    "classification": ("f1_macro", 0.60, 0.99),
    "regression": ("r2", 0.40, 0.99),
    "clustering": ("silhouette", 0.25, None),
}

_SCALING_SWAP = {"standard": "robust", "robust": "standard", "minmax": "robust"}


def alternate_scaling(current: str) -> str:
    return _SCALING_SWAP.get(current, "robust")


def assess(
    problem_type: str,
    key_value: float | None,
    *,
    selected_algorithms: list[str],
    all_algorithms: list[str],
    tried_remedies: list[str],
) -> dict:
    metric, minimum, perfect = QUALITY_TARGETS[problem_type]
    attempts = len(tried_remedies)
    base = {
        "key_metric": metric, "value": key_value, "threshold": minimum,
        "attempt": attempts, "max_retries": MAX_AUTO_RETRIES, "tried_remedies": list(tried_remedies),
        "flags": [], "remedy": None,
    }

    if key_value is None:
        return {**base, "verdict": "proceed", "reasoning": f"No {metric} available to judge quality; proceeding to review."}

    if perfect is not None and key_value >= perfect:
        return {
            **base, "verdict": "proceed", "flags": ["suspected_leakage"],
            "reasoning": f"{metric}={key_value:.3f} is suspiciously close to perfect — not retrying; "
            "flagged as a possible data-leakage symptom for the critic and reviewer.",
        }

    if key_value >= minimum:
        return {**base, "verdict": "proceed", "reasoning": f"{metric}={key_value:.3f} meets the {minimum:.2f} quality bar."}

    if attempts < MAX_AUTO_RETRIES:
        for remedy in REMEDIES:
            if remedy in tried_remedies:
                continue
            if remedy == "broaden_algorithms" and set(selected_algorithms) >= set(all_algorithms):
                continue  # nothing left to broaden
            return {
                **base, "verdict": "retry", "remedy": remedy,
                "reasoning": f"{metric}={key_value:.3f} is below the {minimum:.2f} bar — self-correcting "
                f"(attempt {attempts + 1}/{MAX_AUTO_RETRIES}) with remedy '{remedy}'.",
            }

    return {
        **base, "verdict": "proceed", "flags": ["low_performance_unresolved"],
        "reasoning": f"{metric}={key_value:.3f} is below the {minimum:.2f} bar and no automatic remedies remain "
        f"({attempts} attempt(s) used) — escalating to the reviewer with a warning.",
    }

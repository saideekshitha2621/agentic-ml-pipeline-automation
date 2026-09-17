"""Translates technical pipeline output into plain business language.

Template-based by default (deterministic, no API dependency); `llm_service.explain()` is
used for the nicer phrasing when an API key is configured, with the template as its
fallback — same pattern already used everywhere else in this codebase.
"""
from __future__ import annotations

from app.services import llm_service

_ML_TYPE_LABELS = {
    "classification": "Category prediction (e.g. Yes/No, or which group a case belongs to)",
    "clustering": "Automatic grouping / segmentation of similar records",
    "regression": "Numeric value prediction",
}

_CONFIDENCE_BAND_ACTION = {
    "high": "Recommended action: act on this prediction directly.",
    "medium": "Recommended action: use this prediction, but spot-check important cases before acting.",
    "low": "Recommended action: verify manually before acting — the model isn't highly confident here.",
}

_STAGE_BUSINESS_IMPACT = {
    "profiling": "Understanding the data upfront avoids wasted modeling effort and builds trust in every result that follows.",
    "business_framing": "Framing this as a business question keeps every later decision anchored to a real outcome, not just a technical score.",
    "problem_detection": "Confirms whether this project should predict a category, a number, or find hidden groups — this single decision shapes everything downstream.",
    "data_validation": "Catching data issues now avoids an unreliable model that could lead to costly business decisions later.",
    "cleaning_plan": "Fixing missing or inconsistent data ensures the model learns from complete, trustworthy information instead of guessing.",
    "transformation": "Puts the data in a form the model can learn from accurately, without changing what any number or category means to the business.",
    "train_test_split": "Reserves unseen data to honestly test how the model will perform on new, real-world cases before it's trusted.",
    "algorithm_recommendation": "Focuses effort on the techniques most likely to work well for this data, saving time and compute cost.",
    "training": "Builds and compares candidate models so the best-performing option can be chosen with evidence, not guesswork.",
    "hyperparameter_optimization": "Fine-tunes the chosen approach to extract meaningfully better performance without changing the underlying method.",
    "evaluation": "Translates technical performance into a plain measure of how often the model's predictions can be trusted.",
    "recommendation": "Identifies the single model to put into production, backed by a clear trade-off analysis against the alternatives.",
    "reporting": "Summarizes the entire project into a decision-ready document for stakeholders who weren't in the technical weeds.",
}


_BUSINESS_VALUE_BY_PROBLEM_TYPE = {
    "classification": "Enables proactive, case-by-case decisions instead of reacting after the outcome is already known — "
    "the same information reviewed manually today, but consistently and at scale.",
    "clustering": "Reveals natural segments in the data so different groups can be targeted, served, or managed "
    "differently instead of applying a single one-size-fits-all approach.",
    "regression": "Turns a number that's normally only known after the fact into something that can be planned for "
    "ahead of time.",
}


def ml_type_business_label(problem_type: str | None) -> str:
    return _ML_TYPE_LABELS.get(problem_type or "", "Data analysis")


def business_value_statement(problem_type: str | None) -> str:
    return _BUSINESS_VALUE_BY_PROBLEM_TYPE.get(problem_type or "", "Turns raw historical data into a decision-ready model.")


def key_features_list(column_roles: dict, target_column: str | None, limit: int = 5) -> list[str]:
    """Feature columns most likely to matter, for the "based on X, Y, Z" phrasing —
    excludes id-like/free-text/datetime columns and the target itself."""
    candidates = [
        col for col, meta in column_roles.items()
        if col != target_column and meta.get("role") in ("numeric", "categorical")
    ]
    return candidates[:limit]


def prediction_objective(target_column: str | None, key_features: list[str], problem_type: str | None) -> str:
    if problem_type == "clustering" or not target_column:
        basis = f" based on {', '.join(key_features)}" if key_features else ""
        return f"Group records into natural segments{basis}."
    basis = f"based on {', '.join(key_features)}" if key_features else "based on the available historical data"
    return f"Predict '{target_column}' for new or upcoming records, {basis}."


def business_problem_statement(
    dataset_filename: str, n_rows: int, problem_type: str | None, target_column: str | None, db=None
) -> str:
    if problem_type == "clustering" or not target_column:
        goal = "discover natural groupings within the records so they can be treated differently by segment"
    else:
        goal = f"predict '{target_column}' for future or unseen records"
    template = (
        f"This project analyzes {dataset_filename} ({n_rows:,} historical records) to {goal}. "
        f"The goal is a decision-ready model, not just a technical exercise."
    )
    return llm_service.explain(
        "business_problem",
        {"dataset": dataset_filename, "n_rows": n_rows, "problem_type": problem_type, "target_column": target_column},
        fallback=template,
        db=db,
    )


def business_impact_for_stage(stage: str, decision: dict) -> str:
    base = _STAGE_BUSINESS_IMPACT.get(stage, "Supports a more reliable, explainable outcome for this project.")
    return base


def metric_sentence(metric_key: str, value: float | None, problem_type: str = "classification") -> str:
    if value is None:
        return ""
    if metric_key == "accuracy":
        n = round(value * 100)
        return f"{n} out of 100 predictions are expected to be correct."
    if metric_key == "f1_macro":
        n = round(value * 100)
        return f"On a balanced scale of catching true cases vs. avoiding false alarms across every category, this model scores {n} out of 100."
    if metric_key == "precision_macro":
        n = round(value * 100)
        return f"When the model flags something, it's right about {n} out of 100 times."
    if metric_key == "recall_macro":
        n = round(value * 100)
        return f"Of all the real cases that should be caught, the model finds about {n} out of 100 of them."
    if metric_key == "roc_auc":
        n = round(value * 100)
        return f"The model correctly tells the two outcomes apart about {n} out of 100 times, better than a coin flip (50)."
    if metric_key == "silhouette_score":
        pct = round(max(0.0, min(1.0, (value + 1) / 2)) * 100)
        return f"The groups the model found are distinct and well-separated {pct} out of 100 on a clarity scale."
    if metric_key == "davies_bouldin_score":
        quality = "very distinct" if value < 0.7 else "reasonably distinct" if value < 1.2 else "overlapping"
        return f"The groups found are {quality} from one another."
    if metric_key == "calinski_harabasz_score":
        return "The groups found are dense and well-separated from each other, based on how tightly records cluster within each group."
    return ""


def business_action_for_prediction(confidence: float | None, problem_type: str = "classification") -> str:
    if confidence is None:
        band = "medium"
    elif confidence >= 0.8:
        band = "high"
    elif confidence >= 0.55:
        band = "medium"
    else:
        band = "low"
    return _CONFIDENCE_BAND_ACTION[band]

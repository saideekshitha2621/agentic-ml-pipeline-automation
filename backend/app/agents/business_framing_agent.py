"""Business Framing Agent.

Runs once the problem-type HITL gate confirms `problem_type`/`target_column`, and turns
that technical decision into the plain-English framing the rest of the pipeline (and the
Executive Summary) is built around: what business question this project answers, what kind
of ML task that implies in plain terms, which column is being predicted (if any), the
concrete prediction objective, the key features driving it, and the business value of
having this model at all.
"""
from __future__ import annotations

from app.services import business_language_service


def frame(
    dataset_filename: str, n_rows: int, problem_type: str, target_column: str | None,
    column_roles: dict | None = None, db=None,
) -> dict:
    key_features = business_language_service.key_features_list(column_roles or {}, target_column)
    return {
        "business_problem_statement": business_language_service.business_problem_statement(
            dataset_filename, n_rows, problem_type, target_column, db=db
        ),
        "ml_type_business_label": business_language_service.ml_type_business_label(problem_type),
        "target_variable": target_column,
        "prediction_objective": business_language_service.prediction_objective(target_column, key_features, problem_type),
        "key_features": key_features,
        "business_value": business_language_service.business_value_statement(problem_type),
    }


def summarize(framing: dict) -> str:
    return framing["business_problem_statement"]

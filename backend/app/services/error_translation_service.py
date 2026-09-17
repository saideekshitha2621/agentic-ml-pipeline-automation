"""Turns whatever exception stopped a pipeline run into a message a non-technical
stakeholder can read — used by `agents/orchestrator.py::_fail()` so `PipelineRun.error_message`
(rendered directly in the UI via an Alert) is never a raw stack trace or sklearn message.
"""
from __future__ import annotations

from app.services.preprocessing_service import MissingValueValidationError

_PATTERNS = [
    ("least populated class", "One of the outcome categories has too few examples to reliably split into train/test sets. Try a larger dataset or a less granular target."),
    ("input contains nan", "The data still contains missing values in a place the model can't handle. Please review the Cleaning Plan step and make sure every column has a fill strategy."),
    ("found array with 0 sample", "There isn't enough data left to train on after cleaning — too many rows were excluded. Please review the dataset and cleaning choices."),
    ("could not convert string to float", "A column expected to be numeric contains text that can't be converted. Please review that column's data type."),
]


def to_business_message(exc: Exception) -> str:
    if isinstance(exc, MissingValueValidationError):
        return str(exc)  # already business-friendly, see preprocessing_service.py

    text = str(exc).lower()
    for pattern, message in _PATTERNS:
        if pattern in text:
            return message

    return (
        f"An unexpected technical issue stopped this run ({type(exc).__name__}). "
        "Please review the dataset and try again, or contact support with this run's ID."
    )

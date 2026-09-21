"""Train/Test Split Agent — recommends a split ratio (and whether to stratify) by dataset
size, for supervised problem types only (clustering has no held-out set)."""
from __future__ import annotations

import pandas as pd

from app.agents import feedback_utils

SMALL_DATASET_ROWS = 1000
TINY_DATASET_ROWS = 200
SUPERVISED_PROBLEM_TYPES = {"classification", "regression"}


TEST_SIZE_ALTERNATIVES = [0.2, 0.25, 0.3, 0.15]


def recommend(
    df: pd.DataFrame, problem_type: str, target_column: str | None = None, feedback: list[dict] | None = None
) -> dict:
    n_rows = len(df)
    if problem_type not in SUPERVISED_PROBLEM_TYPES:
        return {
            "applicable": False,
            "test_size": None,
            "stratify": False,
            "confidence": 1.0,
            "confidence_factors": [],
            "reasoning": "Clustering has no held-out target to evaluate against, so no train/test split is used.",
        }

    if n_rows < TINY_DATASET_ROWS:
        test_size, reason = 0.3, f"Only {n_rows} rows — a larger 30% test set is needed for a statistically meaningful evaluation."
    elif n_rows < SMALL_DATASET_ROWS:
        test_size, reason = 0.25, f"{n_rows} rows is a small dataset — 75/25 keeps enough held-out rows to trust the metrics."
    else:
        test_size, reason = 0.2, f"{n_rows} rows is large enough for a standard 80/20 split."

    # Stratifying only makes sense for a discrete/class target — a continuous regression
    # target has no "classes" to keep proportional between train and test.
    stratify = False
    if problem_type == "classification" and target_column and target_column in df.columns:
        stratify = df[target_column].nunique(dropna=True) <= max(20, int(n_rows * 0.05))

    # Phase 1 revision loop: honour a ratio the reviewer asked for, otherwise move to the
    # next ratio that hasn't already been rejected; "no stratif..." turns stratification off.
    if feedback:
        previous = {f["proposal"].get("test_size") for f in feedback if f.get("proposal")}
        requested = feedback_utils.mentioned_fraction(feedback)
        if requested is not None:
            test_size, reason = requested, f"Using the {requested:.0%} test share you asked for."
        elif test_size in previous:
            test_size = next((t for t in TEST_SIZE_ALTERNATIVES if t not in previous), test_size)
            reason = f"A {test_size:.0%} test share — chosen because you rejected the previous ratio."
        if any("no stratif" in r.lower() or "without stratif" in r.lower() for r in feedback_utils.reasons(feedback)):
            stratify = False

    confidence, factors = estimate_confidence(df, problem_type, target_column, revised=bool(feedback))
    return {
        "applicable": True,
        "test_size": test_size,
        "train_size": round(1 - test_size, 2),
        "stratify": stratify,
        "confidence": confidence,
        "confidence_factors": factors,
        "revision": len(feedback) if feedback else 0,
        "reasoning": reason + (
            " Stratifying by the target keeps class proportions consistent between train and test."
            if stratify
            else ""
        ),
    }


def estimate_confidence(
    df: pd.DataFrame, problem_type: str, target_column: str | None, revised: bool = False
) -> tuple[float, list[str]]:
    score, factors = 0.9, []
    n_rows = len(df)
    if n_rows < TINY_DATASET_ROWS:
        score -= 0.15
        factors.append(f"-0.15: only {n_rows} rows, so any split leaves a small evaluation set")
    if problem_type == "classification" and target_column in df.columns:
        counts = df[target_column].value_counts(dropna=True)
        if len(counts) and counts.min() < 10:
            score -= 0.2
            factors.append(f"-0.20: rarest class has only {int(counts.min())} row(s)")
        if len(counts) > 1 and counts.max() / max(counts.min(), 1) > 10:
            score -= 0.1
            factors.append("-0.10: classes are heavily imbalanced (>10:1)")
    if revised:
        score -= 0.05
        factors.append("-0.05: re-proposed after a rejection")
    return round(max(score, 0.3), 2), factors


def summarize(result: dict) -> str:
    if not result["applicable"]:
        return result["reasoning"]
    return f"Recommending a {result['train_size']:.0%}/{result['test_size']:.0%} train/test split. {result['reasoning']}"

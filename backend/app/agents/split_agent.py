"""Train/Test Split Agent — recommends a split ratio (and whether to stratify) by dataset
size, for classification only (clustering has no held-out set)."""
from __future__ import annotations

import pandas as pd

SMALL_DATASET_ROWS = 1000
TINY_DATASET_ROWS = 200


def recommend(df: pd.DataFrame, problem_type: str, target_column: str | None = None) -> dict:
    n_rows = len(df)
    if problem_type != "classification":
        return {
            "applicable": False,
            "test_size": None,
            "stratify": False,
            "reasoning": "Clustering has no held-out target to evaluate against, so no train/test split is used.",
        }

    if n_rows < TINY_DATASET_ROWS:
        test_size, reason = 0.3, f"Only {n_rows} rows — a larger 30% test set is needed for a statistically meaningful evaluation."
    elif n_rows < SMALL_DATASET_ROWS:
        test_size, reason = 0.25, f"{n_rows} rows is a small dataset — 75/25 keeps enough held-out rows to trust the metrics."
    else:
        test_size, reason = 0.2, f"{n_rows} rows is large enough for a standard 80/20 split."

    stratify = False
    if target_column and target_column in df.columns:
        stratify = df[target_column].nunique(dropna=True) <= max(20, int(n_rows * 0.05))

    return {
        "applicable": True,
        "test_size": test_size,
        "train_size": round(1 - test_size, 2),
        "stratify": stratify,
        "reasoning": reason + (
            " Stratifying by the target keeps class proportions consistent between train and test."
            if stratify
            else ""
        ),
    }


def summarize(result: dict) -> str:
    if not result["applicable"]:
        return result["reasoning"]
    return f"Recommending a {result['train_size']:.0%}/{result['test_size']:.0%} train/test split. {result['reasoning']}"

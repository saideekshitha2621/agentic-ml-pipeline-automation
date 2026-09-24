"""Data Validation Agent.

Runs a fixed checklist over the dataset (missing values, duplicates, constant columns,
outliers, high-cardinality columns, leakage risk against the declared/detected target,
invalid dtypes) and returns one row per check with a severity, so the HITL reviewer sees a
pass/warn/fail checklist rather than raw stats. Reuses `profiling_service`'s existing
missing-value/outlier detectors rather than recomputing them.
"""
from __future__ import annotations

import pandas as pd

from app.services import profiling_service

HIGH_CARDINALITY_ABS = 50
HIGH_CARDINALITY_RATIO = 0.5
MIN_ROWS_TO_TRAIN = 10
# Model-based leakage probe (replaces a numeric-only Pearson-correlation check that silently
# skipped categorical/date leak columns like `cancellation_date` or `churn_reason`, and whose
# 0.98 threshold a noisy real-world leak like `exit_survey_score` would rarely reach). A shallow
# single-feature decision tree can use ONE column (numeric or one-hot-encoded categorical) to
# predict the target; a near-perfect score means that column is very likely leaking the answer
# rather than genuinely predicting it.
LEAKAGE_PROBE_MAX_ROWS = 2000
LEAKAGE_PROBE_MAX_CATEGORIES = 50  # beyond this, one-hot-encoding a single column for the probe is too costly/noisy
# A single strong (but legitimate) numeric predictor can score high in absolute terms without
# being a leak (e.g. an "amount" feature that a skewed/imbalanced label was largely derived
# from can reach ~97% single-feature accuracy against an 88% majority-class baseline). Requiring
# BOTH a very high absolute score AND a large lift over the naive baseline avoids flagging that
# case, while a near-total single-feature score (>= NEAR_PERFECT) is still flagged regardless of
# lift, since that level of single-column determinism is implausible for a genuine feature.
LEAKAGE_PROBE_SCORE_THRESHOLD = 0.95  # accuracy (classification) or r2 (regression) from ONE column alone
LEAKAGE_PROBE_LIFT_THRESHOLD = 0.20  # required accuracy lift over the majority-class baseline (classification only)
LEAKAGE_PROBE_NEAR_PERFECT = 0.99  # flagged regardless of lift/baseline at this level
# Post-outcome fields are often only *populated* for one class (e.g. `exit_survey_score` or
# `cancellation_date` only exist for churned customers, NaN otherwise) — the value-based probe
# above can't see this at all, because dropping rows with a missing value in that column also
# drops every row that would prove it's a leak, leaving a single-class target it skips. This
# checks the null/non-null *pattern* itself against the target instead.
LEAKAGE_MISSINGNESS_GAP_THRESHOLD = 0.8  # gap between per-class null rates


def _leakage_scores(df: pd.DataFrame, target_column: str) -> dict[str, float]:
    """Single-feature leakage probe for every non-target, non-constant column — numeric or
    categorical alike (a shallow decision tree can use either). Runs on a capped sample so
    validation stays fast even on large datasets."""
    import numpy as np
    from sklearn.model_selection import cross_val_score
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

    sample = df if len(df) <= LEAKAGE_PROBE_MAX_ROWS else df.sample(LEAKAGE_PROBE_MAX_ROWS, random_state=0)
    y_full = sample[target_column]
    is_regression = pd.api.types.is_numeric_dtype(y_full) and y_full.nunique(dropna=True) > 20
    baseline = 0.0 if is_regression else float(y_full.value_counts(normalize=True, dropna=True).max() or 0.0)

    scores: dict[str, float] = {}
    for col in df.columns:
        if col == target_column or df[col].nunique(dropna=True) <= 1:
            continue
        is_numeric = pd.api.types.is_numeric_dtype(df[col])
        if not is_numeric and df[col].nunique(dropna=True) > LEAKAGE_PROBE_MAX_CATEGORIES:
            continue  # already surfaced separately as a high-cardinality column
        data = sample[[col, target_column]].dropna()
        if len(data) < 20:
            continue
        x = data[[col]] if is_numeric else pd.get_dummies(data[[col]])
        y = data[target_column]
        if not is_regression and y.nunique(dropna=True) < 2:
            continue
        model = DecisionTreeRegressor(max_depth=3) if is_regression else DecisionTreeClassifier(max_depth=3)
        try:
            score = float(np.mean(cross_val_score(model, x, y, cv=3, scoring="r2" if is_regression else "accuracy")))
        except ValueError:
            continue
        if is_regression:
            if score >= LEAKAGE_PROBE_NEAR_PERFECT:
                scores[col] = score
        elif score >= LEAKAGE_PROBE_NEAR_PERFECT or (
            score >= LEAKAGE_PROBE_SCORE_THRESHOLD and (score - baseline) >= LEAKAGE_PROBE_LIFT_THRESHOLD
        ):
            scores[col] = score
    return scores


def _missingness_leak_scores(df: pd.DataFrame, target_column: str) -> dict[str, float]:
    """Flags columns whose null/non-null pattern almost perfectly separates the target's
    classes — a common real-world leak the value-based probe above cannot see (see module
    docstring above `LEAKAGE_MISSINGNESS_GAP_THRESHOLD`). Classification only: a regression
    target has no small set of classes to group null-rates by."""
    y = df[target_column]
    if pd.api.types.is_numeric_dtype(y) and y.nunique(dropna=True) > 20:
        return {}
    classes = y.dropna().unique()
    if not (2 <= len(classes) <= 10):
        return {}
    scores: dict[str, float] = {}
    for col in df.columns:
        if col == target_column:
            continue
        is_null = df[col].isna()
        n_null = int(is_null.sum())
        if n_null == 0 or n_null == len(df):
            continue  # no missingness, or 100% missing (already flagged separately)
        null_rate_by_class = df.groupby(y, observed=True)[col].apply(lambda s: s.isna().mean())
        gap = float(null_rate_by_class.max() - null_rate_by_class.min())
        if gap >= LEAKAGE_MISSINGNESS_GAP_THRESHOLD:
            scores[col] = round(gap, 3)
    return scores


def _status(n_bad: int, warn_at: int = 1, critical_at: int | None = None) -> str:
    if n_bad == 0:
        return "ok"
    if critical_at is not None and n_bad >= critical_at:
        return "critical"
    return "warning"


def validate(df: pd.DataFrame, profile: dict, target_column: str | None = None, problem_type: str | None = None) -> dict:
    checks: list[dict] = []

    n_rows = len(df)
    checks.append(
        {
            "name": "Dataset size",
            "status": "critical" if n_rows == 0 else "critical" if n_rows < MIN_ROWS_TO_TRAIN else "ok",
            "detail": "The dataset is empty — there is nothing to train on."
            if n_rows == 0
            else f"Only {n_rows} row(s) — this is too few to train a reliable model (minimum recommended: {MIN_ROWS_TO_TRAIN})."
            if n_rows < MIN_ROWS_TO_TRAIN
            else f"{n_rows:,} rows is enough to proceed.",
            "affected_columns": [],
        }
    )

    if target_column:
        if target_column not in df.columns:
            checks.append(
                {
                    "name": "Target column validity",
                    "status": "critical",
                    "detail": f"The declared target column '{target_column}' does not exist in this dataset.",
                    "affected_columns": [target_column],
                }
            )
        else:
            non_null = df[target_column].dropna()
            n_distinct = int(non_null.nunique())
            if non_null.empty:
                status, detail = "critical", f"Every row is missing a '{target_column}' value — there's nothing to learn from."
            elif problem_type == "classification" and n_distinct < 2:
                status, detail = "critical", (
                    f"Every row has the same '{target_column}' outcome ({non_null.iloc[0]!r}) — there's nothing for a "
                    "classification model to learn to distinguish between."
                )
            elif problem_type == "regression" and n_distinct < 2:
                status, detail = "critical", (
                    f"Every row has the same '{target_column}' value ({non_null.iloc[0]!r}) — there's no variation for a "
                    "regression model to learn to predict."
                )
            else:
                status, detail = "ok", f"'{target_column}' has {n_distinct} distinct value(s) — valid to train against."
            checks.append({"name": "Target column validity", "status": status, "detail": detail, "affected_columns": [] if status == "ok" else [target_column]})

    missing = [m for m in profiling_service.missing_value_rows(df) if m["missing_count"] > 0]
    checks.append(
        {
            "name": "Missing values",
            "status": _status(len(missing), critical_at=max(1, len(df.columns) // 2)),
            "detail": f"{len(missing)} of {len(df.columns)} column(s) have missing values."
            if missing
            else "No missing values found.",
            "affected_columns": [m["column"] for m in missing],
        }
    )

    if target_column and target_column in df.columns:
        n_missing_target = int(df[target_column].isna().sum())
        checks.append(
            {
                "name": "Missing target values",
                "status": _status(n_missing_target, critical_at=max(1, int(len(df) * 0.3))),
                "detail": f"{n_missing_target} row(s) ({n_missing_target / max(len(df), 1):.1%}) have no "
                f"'{target_column}' value — a model can't learn from or be scored against an unlabeled "
                "row, so these rows will be excluded before training."
                if n_missing_target
                else f"Every row has a '{target_column}' value.",
                "affected_columns": [target_column] if n_missing_target else [],
            }
        )

    n_dupes = int(df.duplicated().sum())
    checks.append(
        {
            "name": "Duplicate rows",
            "status": _status(n_dupes, critical_at=max(1, int(len(df) * 0.1))),
            "detail": f"{n_dupes} duplicate row(s) found ({n_dupes / max(len(df), 1):.1%} of the dataset)."
            if n_dupes
            else "No duplicate rows found.",
            "affected_columns": [],
        }
    )

    constant_cols = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]
    checks.append(
        {
            "name": "Constant columns",
            "status": _status(len(constant_cols)),
            "detail": f"{len(constant_cols)} column(s) have a single distinct value and carry no signal."
            if constant_cols
            else "No constant columns found.",
            "affected_columns": constant_cols,
        }
    )

    outliers = profiling_service.detect_outliers(df)
    heavy_outliers = [o for o in outliers if o["pct_outliers"] >= 5]
    checks.append(
        {
            "name": "Outliers",
            "status": _status(len(heavy_outliers)),
            "detail": f"{len(outliers)} numeric column(s) show IQR outliers ({len(heavy_outliers)} above 5% of rows)."
            if outliers
            else "No significant outliers found in numeric columns.",
            "affected_columns": [o["column"] for o in outliers],
        }
    )

    n_rows = len(df) or 1
    high_card = [
        c
        for c in df.select_dtypes(exclude="number").columns
        if df[c].nunique(dropna=True) > min(HIGH_CARDINALITY_ABS, n_rows * HIGH_CARDINALITY_RATIO)
    ]
    checks.append(
        {
            "name": "High-cardinality columns",
            "status": _status(len(high_card)),
            "detail": f"{len(high_card)} categorical column(s) have very many distinct values — "
            "one-hot encoding them will explode dimensionality."
            if high_card
            else "No high-cardinality categorical columns found.",
            "affected_columns": high_card,
        }
    )

    leakage_cols: list[str] = []
    if target_column and target_column in df.columns and int(df[target_column].dropna().shape[0]) >= 20:
        try:
            leakage_cols = sorted(set(_leakage_scores(df, target_column)) | set(_missingness_leak_scores(df, target_column)))
        except Exception:
            leakage_cols = []
    checks.append(
        {
            "name": "Data leakage risk",
            "status": _status(len(leakage_cols), critical_at=1),
            "detail": f"{len(leakage_cols)} column(s), alone, can almost perfectly predict the target — "
            "likely leak the answer rather than genuinely predict it."
            if leakage_cols
            else "No columns show suspiciously high single-feature predictive power against the target.",
            "affected_columns": leakage_cols,
        }
    )

    invalid_dtype_cols = []
    for col in df.select_dtypes(exclude="number").columns:
        sample = df[col].dropna().astype(str).head(50)
        if sample.empty:
            continue
        numeric_like = pd.to_numeric(sample, errors="coerce").notna().mean()
        if numeric_like > 0.9:
            invalid_dtype_cols.append(col)
    checks.append(
        {
            "name": "Invalid data types",
            "status": _status(len(invalid_dtype_cols)),
            "detail": f"{len(invalid_dtype_cols)} column(s) look numeric but are stored as text."
            if invalid_dtype_cols
            else "All columns are typed consistently with their contents.",
            "affected_columns": invalid_dtype_cols,
        }
    )

    severities = [c["status"] for c in checks]
    overall_status = "critical" if "critical" in severities else "warning" if "warning" in severities else "ok"
    return {"checks": checks, "overall_status": overall_status}


def summarize(result: dict) -> str:
    n_ok = sum(1 for c in result["checks"] if c["status"] == "ok")
    n_warn = sum(1 for c in result["checks"] if c["status"] == "warning")
    n_crit = sum(1 for c in result["checks"] if c["status"] == "critical")
    return (
        f"Ran {len(result['checks'])} data-quality checks: {n_ok} passed, {n_warn} raised warnings, "
        f"{n_crit} critical. Overall status: {result['overall_status']}."
    )

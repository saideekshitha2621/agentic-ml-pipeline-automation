"""Cleaning Plan Agent.

Produces exactly one recommendation row per retained column, with the missing count/
percentage that drove the recommendation and a plain-English reason — the HITL reviewer
sees *why* each fix is proposed and can override any single row (including supplying a
custom fill value) before it's applied. This replaces the old always-auto-applied
`preprocessing_service.detect_default_plan()` call — the human-approved recommendations
here become the `PreprocessingPlanORM` fields (including the per-column `column_actions`
map) instead of a single dataset-wide strategy.
"""
from __future__ import annotations

import re

import pandas as pd

from app.agents import feedback_utils

ID_LIKE_UNIQUENESS_THRESHOLD = 0.95
HIGH_MISSING_ROW_DROP_THRESHOLD = 50.0  # % missing above which dropping affected rows beats imputing
FULLY_MISSING_OPTIONS = ["drop_column", "fill_zero", "fill_custom", "business_rule"]
_ID_NAME_PATTERN = re.compile(r"(^id$|_id$|^id_|uuid|guid|^key$|_key$|^code$|_code$)", re.IGNORECASE)


def _looks_like_identifier_name(col: str) -> bool:
    return bool(_ID_NAME_PATTERN.search(col))


def _missing_stats(series: pd.Series, n_rows: int) -> tuple[int, float]:
    n_missing = int(series.isna().sum())
    return n_missing, round((n_missing / n_rows * 100) if n_rows else 0.0, 2)


def propose(
    df: pd.DataFrame, validation_result: dict, target_column: str | None = None, feedback: list[dict] | None = None
) -> dict:
    """`feedback` (Phase 1 revision loop) switches on a conservative re-plan: columns the
    reviewer names in their rejection reason are protected from being dropped, leakage-
    flagged columns are kept (imputed) instead of dropped, and 'drop the affected rows'
    becomes imputation — the three ways a rejected cleaning plan most often over-reached."""
    protected = feedback_utils.mentioned_columns(feedback, list(df.columns))
    conservative = bool(feedback)
    revision_notes: list[str] = []
    checks_by_name = {c["name"]: c for c in validation_result["checks"]}
    constant_cols = set(checks_by_name.get("Constant columns", {}).get("affected_columns", []))
    leakage_cols = set(checks_by_name.get("Data leakage risk", {}).get("affected_columns", []))
    high_card_cols = set(checks_by_name.get("High-cardinality columns", {}).get("affected_columns", []))

    n_rows = len(df) or 1
    # Float columns are excluded outright — continuous measurements (sensor readings,
    # prices, scores) are often naturally all-unique without being identifiers. A
    # high-cardinality *integer* column is genuinely ambiguous (customer_id vs. sqft/
    # age_in_days/price_in_cents all look identical by uniqueness alone), so it's only
    # treated as id-like when its name also looks like one — otherwise a legitimate,
    # often highly-predictive numeric feature gets silently dropped from modeling (this
    # is exactly what happened to a regression target's most important feature).
    id_like = set()
    # Integer, near-unique, non-id-named columns (e.g. `customer_number`, `acct_no`) are
    # deliberately not auto-dropped as id_like (see comment above), but silently keeping them
    # as an ordinary numeric feature risks the model memorizing a sequential/cohort-correlated
    # ID rather than learning a real pattern — since train_test_split sends each unique ID to
    # only one side, such memorization can look good on the held-out test set without
    # generalizing. Surface these for human review instead of silently deciding either way.
    ambiguous_id_like = set()
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]):
            continue
        if df[c].nunique(dropna=True) / n_rows <= ID_LIKE_UNIQUENESS_THRESHOLD:
            continue
        if pd.api.types.is_integer_dtype(df[c]) and not _looks_like_identifier_name(c):
            ambiguous_id_like.add(c)
            continue
        id_like.add(c)

    recommendations: list[dict] = []
    target_missing_note: dict | None = None
    for col in df.columns:
        if col == target_column:
            # The label isn't a feature to impute — a missing target can't be filled or
            # guessed without inventing the answer, so affected rows are simply excluded
            # before splitting (see preprocessing_service.split_target). Surfaced here as a
            # read-only note rather than an editable recommendation.
            n_missing, missing_pct = _missing_stats(df[col], n_rows)
            if n_missing:
                target_missing_note = {
                    "column": col, "column_type": "target", "missing_count": n_missing, "missing_pct": missing_pct,
                    "issue": f"{n_missing} row(s) have no target value", "action": "drop_rows_for_target",
                    "reason": "Rows without a label can't be used for training or evaluation and will be "
                    "excluded automatically before the train/test split.",
                }
            continue
        n_missing, missing_pct = _missing_stats(df[col], n_rows)
        column_type = "numeric" if pd.api.types.is_numeric_dtype(df[col]) else "categorical"
        base = {"column": col, "column_type": column_type, "missing_count": n_missing, "missing_pct": missing_pct}

        # 100%-missing takes priority over id-like/constant/leakage — those checks would
        # otherwise also fire on a fully-empty column (nunique==0 looks "constant") but with
        # a less specific reason than "no information available", and without the dedicated
        # HITL options (drop / fill 0 / fill custom / business rule) this case needs.
        if missing_pct >= 100.0:
            recommendations.append({
                **base,
                "issue": "No information available (100% missing)",
                "action": "drop_column",
                "reason": "Every value in this column is missing — there is nothing to learn or impute from, "
                "so keeping it would only add noise. Dropping is recommended.",
                "no_information": True,
                "options": FULLY_MISSING_OPTIONS,
            })
            continue

        if col in id_like and col not in protected:
            recommendations.append({**base, "issue": "identifier-like column", "action": "drop_column",
                                     "reason": f"{df[col].nunique()} nearly-unique values — an identifier carries no predictive signal."})
            continue
        if col in constant_cols and col not in protected:
            recommendations.append({**base, "issue": "constant column", "action": "drop_column",
                                     "reason": "Every row has the same value — this column can't inform any model."})
            continue
        if col in leakage_cols and not conservative and col not in protected:
            recommendations.append({**base, "issue": "data leakage risk", "action": "drop_column",
                                     "reason": "Near-perfect correlation with the target suggests this column leaks the answer."})
            continue

        flags = ["high_cardinality"] if col in high_card_cols and column_type == "categorical" else []
        if col in ambiguous_id_like:
            flags.append("ambiguous_id_like")
            revision_notes.append(
                f"'{col}' is nearly all-unique but its name doesn't look like an identifier — kept as a "
                "feature; confirm it isn't a row/customer ID before training, since the model could otherwise "
                "memorize it rather than learn a generalizable pattern."
            )
        if col in leakage_cols:
            flags.append("possible_leakage_kept_after_review")
            revision_notes.append(f"Kept '{col}' (flagged for possible leakage) after your rejection — verify it is available at prediction time.")
        elif col in protected:
            revision_notes.append(f"Kept '{col}' because your rejection reason named it.")

        if n_missing == 0:
            if "possible_leakage_kept_after_review" in flags:
                issue = "Possible leakage (kept after review)"
                reason = "Flagged as a potential leak of the target, but kept because you rejected dropping it — confirm it is known at prediction time."
            elif "ambiguous_id_like" in flags:
                issue = "Possibly an identifier (kept for review)"
                reason = f"{df[col].nunique()} nearly-unique values but the name doesn't look like an ID — kept as a feature; confirm it isn't a row/customer identifier."
            else:
                issue = "High cardinality" if flags else "No issues detected."
                reason = "Consider grouping rare categories or dropping this column before encoding." if flags else "No missing values — no action needed."
            recommendations.append({**base, "issue": issue, "action": "keep", "reason": reason, **({"flags": flags} if flags else {})})
            continue

        if missing_pct > HIGH_MISSING_ROW_DROP_THRESHOLD and not conservative:
            action = "drop_rows"
            reason = f"{missing_pct:.0f}% missing — too sparse to impute reliably; dropping the affected rows is safer than inventing values for most of the column."
        elif column_type == "numeric":
            skew = float(df[col].skew()) if df[col].notna().sum() > 2 else 0.0
            if abs(skew) > 1:
                action, reason = "median", f"Numeric and skewed (skew={skew:.2f}) with outlier sensitivity — median resists outliers better than the mean."
            else:
                action, reason = "mean", "Numeric and approximately normally distributed — the mean is a representative fill value."
        else:
            action, reason = "mode", "Categorical — the most frequent value is the safest fill."

        issue = f"{n_missing} missing value(s) ({missing_pct:.1f}%)" + (" + high cardinality" if flags else "")
        recommendations.append({**base, "issue": issue, "action": action, "reason": reason, **({"flags": flags} if flags else {})})

    n_dupes = int(df.duplicated().sum())
    return {
        "recommendations": recommendations,
        "n_duplicates": n_dupes,
        "drop_duplicates": n_dupes > 0,
        "target_missing": target_missing_note,  # informational only — see preprocessing_service.split_target
        "revision": len(feedback) if feedback else 0,
        "revision_notes": revision_notes,
    }


def estimate_confidence(plan: dict, n_columns: int) -> tuple[float, list[str]]:
    """Confidence derived from what the plan actually contains rather than a flat constant:
    every risky action lowers it, feeding straight into the auto-approval policy."""
    recs = plan["recommendations"]
    score, factors = 0.95, []

    def penalize(amount: float, why: str) -> None:
        nonlocal score
        score -= amount
        factors.append(f"-{amount:.2f}: {why}")

    if any(r.get("no_information") for r in recs):
        penalize(0.4, "column(s) with no usable information")
    if any(r["action"] == "drop_rows" for r in recs):
        penalize(0.2, "plan drops rows instead of imputing")
    if any(r["action"] == "drop_column" and "leakage" in r.get("issue", "") for r in recs):
        penalize(0.15, "plan removes column(s) for suspected leakage")
    heavy = [r for r in recs if r.get("missing_pct", 0) > 20 and r["action"] in ("mean", "median", "mode")]
    if heavy:
        penalize(min(0.05 * len(heavy), 0.2), f"{len(heavy)} column(s) imputed with >20% missing values")
    dropped = sum(1 for r in recs if r["action"] == "drop_column")
    if n_columns and dropped / n_columns > 0.3:
        penalize(0.15, f"{dropped} of {n_columns} columns dropped")
    if plan.get("revision"):
        penalize(0.05, "re-proposed after a rejection")
    return round(max(score, 0.2), 2), factors


def summarize(plan: dict) -> str:
    recs = plan["recommendations"]
    n_drop_cols = sum(1 for r in recs if r["action"] == "drop_column")
    n_impute = sum(1 for r in recs if r["action"] in ("mean", "median", "mode"))
    n_drop_rows = sum(1 for r in recs if r["action"] == "drop_rows")
    n_no_info = sum(1 for r in recs if r.get("no_information"))
    parts = [f"{n_impute} column(s) to impute", f"{n_drop_cols} column(s) to drop"]
    if n_no_info:
        parts.append(f"{n_no_info} of those are 100% missing (no information available)")
    if plan.get("target_missing"):
        parts.append(f"{plan['target_missing']['missing_count']} row(s) with no target value will be excluded")
    if n_drop_rows:
        parts.append(f"{n_drop_rows} column(s) recommend dropping the affected rows instead of imputing")
    if plan["drop_duplicates"]:
        parts.append(f"{plan['n_duplicates']} duplicate row(s) to remove")
    return "; ".join(parts) + "."


def to_preprocessing_plan_fields(plan: dict) -> dict:
    """Converts the approved (possibly human-edited) cleaning plan into
    PreprocessingPlanORM constructor fields, deriving retained/dropped columns and the
    per-column `column_actions` map directly from the (possibly edited) recommendations —
    the single source of truth a human reviewer may have overridden row by row."""
    recs = plan["recommendations"]
    numerical_columns = [r["column"] for r in recs if r["column_type"] == "numeric" and r["action"] != "drop_column"]
    categorical_columns = [r["column"] for r in recs if r["column_type"] == "categorical" and r["action"] != "drop_column"]
    dropped_columns = [r["column"] for r in recs if r["action"] == "drop_column"]
    column_actions = {
        r["column"]: {"action": r["action"], "custom_value": r.get("custom_value")}
        for r in recs if r["action"] != "drop_column"
    }
    return {
        "numerical_columns": numerical_columns,
        "categorical_columns": categorical_columns,
        "dropped_columns": dropped_columns,
        # Legacy global-strategy fields — unused when column_actions is populated (every
        # retained column has its own action), kept only because PreprocessingPlanORM
        # requires non-null values and the old manual flow still reads them directly.
        "numerical_impute_strategy": "median",
        "categorical_impute_strategy": "mode",
        "scaling_method": "standard",
        "drop_duplicates": plan["drop_duplicates"],
        "pca_enabled": False,
        "pca_variance_target": 0.95,
        "column_actions": column_actions,
    }

"""Feature engineering (Phase 4): stateless, reproducible column transforms.

Currently one transform — `log1p` for heavily right-skewed, non-negative numeric columns
(prices, counts, durations), which linear/distance-based models handle far better after
compression. It is *in place* (the column keeps its name), so every downstream column list,
one-hot mapping and the Prediction Playground schema stay valid.

Stateless on purpose: log1p has no fitted parameters, so applying it to the whole dataset
before the train/test split cannot leak test information, and the exact same function is
reused at prediction time on a single raw row (see champion_service.ChampionPipeline).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SKEW_THRESHOLD = 1.5
MIN_DISTINCT = 10  # skip coded/binary columns — compressing a 0/1 flag is meaningless
SUPPORTED_OPS = {"log1p"}


def propose_log_transforms(df: pd.DataFrame, numeric_columns: list[str], target_column: str | None = None) -> list[dict]:
    specs = []
    for col in numeric_columns:
        if col == target_column or col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(series) < 20 or series.nunique() < MIN_DISTINCT or series.min() < 0:
            continue
        skew = float(series.skew())
        if skew > SKEW_THRESHOLD:
            specs.append({
                "column": col, "op": "log1p", "skew_before": round(skew, 2),
                "skew_after": round(float(np.log1p(series).skew()), 2),
                "reason": f"Strongly right-skewed (skew {skew:.1f}) and non-negative — log compression makes it far more model-friendly.",
            })
    return specs


def apply(df: pd.DataFrame, specs: list[dict] | None) -> pd.DataFrame:
    """Returns a copy with each spec applied; specs naming absent columns are ignored."""
    if not specs:
        return df
    out = df.copy()
    for spec in specs:
        col, op = spec.get("column"), spec.get("op")
        if op not in SUPPORTED_OPS or col not in out.columns:
            continue
        values = pd.to_numeric(out[col], errors="coerce")
        out[col] = np.log1p(values.clip(lower=0))  # clip: an unseen negative at predict time must not yield NaN
    return out


def inverse_value(value: float | None, col: str, specs: list[dict] | None) -> float | None:
    """Maps a value from the transformed scale back to the raw scale (for displayed defaults)."""
    if value is None or not specs or not any(s.get("column") == col and s.get("op") == "log1p" for s in specs):
        return value
    return float(np.expm1(value))

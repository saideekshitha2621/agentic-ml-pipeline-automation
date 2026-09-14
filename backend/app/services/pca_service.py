from __future__ import annotations

import pandas as pd

from ml_automation.dimensionality_reduction import apply_pca, explained_variance_curve


def preview(encoded_df: pd.DataFrame) -> dict:
    curve = explained_variance_curve(encoded_df)
    return {
        "explained_variance_ratio": [round(float(v), 4) for v in curve["explained_variance_ratio"]],
        "cumulative_variance": [round(float(v), 4) for v in curve["cumulative_variance"]],
        "components": curve["component"].tolist(),
    }


def apply(encoded_df: pd.DataFrame, n_components: float) -> tuple[pd.DataFrame, dict]:
    return apply_pca(encoded_df, n_components=n_components)

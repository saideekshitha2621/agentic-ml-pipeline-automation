"""Optional PCA step, with the decision to apply it left to the human."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def apply_pca(
    df: pd.DataFrame,
    n_components: int | float | None = None,
    random_state: int = 42,
) -> tuple[pd.DataFrame, dict]:
    """Fit PCA on a numeric/scaled dataframe and return the transformed frame + report.

    n_components can be an int (# of components), a float in (0,1] (variance to retain),
    or None (defaults to min(n_features, n_samples) capped at retaining 95% variance).
    """
    if n_components is None:
        n_components = 0.95

    pca = PCA(n_components=n_components, random_state=random_state)
    transformed = pca.fit_transform(df.values)

    component_cols = [f"PC{i + 1}" for i in range(transformed.shape[1])]
    pca_df = pd.DataFrame(transformed, columns=component_cols, index=df.index)

    report = {
        "n_components": int(transformed.shape[1]),
        "explained_variance_ratio": [round(float(v), 4) for v in pca.explained_variance_ratio_],
        "cumulative_explained_variance": [
            round(float(v), 4) for v in np.cumsum(pca.explained_variance_ratio_)
        ],
        "total_variance_retained": round(float(np.sum(pca.explained_variance_ratio_)), 4),
        "original_n_features": df.shape[1],
    }
    return pca_df, report


def explained_variance_curve(df: pd.DataFrame, max_components: int | None = None) -> pd.DataFrame:
    """Full explained-variance curve for a scree plot, useful for the human review step."""
    max_components = max_components or min(df.shape[0], df.shape[1])
    pca = PCA(n_components=max_components, random_state=42)
    pca.fit(df.values)
    return pd.DataFrame(
        {
            "component": [f"PC{i + 1}" for i in range(max_components)],
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "cumulative_variance": np.cumsum(pca.explained_variance_ratio_),
        }
    )

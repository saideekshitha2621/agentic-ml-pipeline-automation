from __future__ import annotations

import numpy as np
import pandas as pd

from ml_automation.interpretation import interpret_clusters


def explain(cleaned_df: pd.DataFrame, encoded_df: pd.DataFrame, labels: np.ndarray) -> dict:
    result = interpret_clusters(cleaned_df, encoded_df, labels)
    return {
        "profiles": result["profiles"].to_dict("records"),
        "feature_importance": result["feature_importance"].to_dict("records"),
        "summaries": {str(k): v for k, v in result["summaries"].items()},
        "suggested_names": {str(k): v for k, v in result["suggested_names"].items()},
    }

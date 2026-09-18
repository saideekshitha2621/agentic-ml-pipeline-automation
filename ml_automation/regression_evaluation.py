"""Evaluation metrics for regression runs."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score


def evaluate_run(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute RMSE, MAE, MAPE, and R² for one regression run."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    try:
        mape = float(mean_absolute_percentage_error(y_true, y_pred))
    except ValueError:
        mape = None  # e.g. y_true contains zeros

    return {
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "mape": round(mape, 4) if mape is not None else None,
        "r2": round(r2, 4),
    }

"""Evaluation metrics for classification runs."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_run(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray | None = None) -> dict:
    """Compute accuracy, macro precision/recall/F1, ROC-AUC, log loss, and a confusion
    matrix for one classification run. ROC-AUC and log loss are only defined when the
    model produced class probabilities (y_proba) — some estimators/param combinations
    don't (e.g. an SVM without probability=True), so both are None in that case rather
    than raising.
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    labels = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    roc_auc = None
    logloss = None
    if y_proba is not None:
        n_classes = y_proba.shape[1]
        try:
            if n_classes == 2:
                roc_auc = roc_auc_score(y_true, y_proba[:, 1])
            else:
                roc_auc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
        except ValueError:
            roc_auc = None  # e.g. a class missing from the test split
        try:
            logloss = log_loss(y_true, y_proba, labels=labels)
        except ValueError:
            logloss = None

    return {
        "accuracy": round(float(accuracy), 4),
        "precision_macro": round(float(precision), 4),
        "recall_macro": round(float(recall), 4),
        "f1_macro": round(float(f1), 4),
        "roc_auc": round(float(roc_auc), 4) if roc_auc is not None else None,
        "log_loss": round(float(logloss), 4) if logloss is not None else None,
        "confusion_matrix": {"labels": [str(l) for l in labels], "matrix": cm.tolist()},
    }

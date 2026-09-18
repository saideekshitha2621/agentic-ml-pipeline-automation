from __future__ import annotations

import numpy as np

from app.plugins.base import PluginRun
from app.plugins.classification_base import ClassificationPluginRun
from app.plugins.regression_base import RegressionPluginRun
from ml_automation.classification_evaluation import evaluate_run as evaluate_classification_run
from ml_automation.clustering import ClusterRun as _ClusterRunAdapter
from ml_automation.evaluation import evaluate_run
from ml_automation.regression_evaluation import evaluate_run as evaluate_regression_run


def evaluate(X: np.ndarray, run: PluginRun) -> dict:
    adapter = _ClusterRunAdapter(run.algorithm, run.params, run.labels, run.n_clusters, run.n_noise)
    return evaluate_run(X, adapter)


def evaluate_classification(y_test: np.ndarray, run: ClassificationPluginRun) -> dict:
    return evaluate_classification_run(y_test, run.y_pred, run.y_proba)


def evaluate_regression(y_test: np.ndarray, run: RegressionPluginRun) -> dict:
    return evaluate_regression_run(y_test, run.y_pred)

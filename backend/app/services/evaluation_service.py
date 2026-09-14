from __future__ import annotations

import numpy as np

from app.plugins.base import PluginRun
from ml_automation.clustering import ClusterRun as _ClusterRunAdapter
from ml_automation.evaluation import evaluate_run


def evaluate(X: np.ndarray, run: PluginRun) -> dict:
    adapter = _ClusterRunAdapter(run.algorithm, run.params, run.labels, run.n_clusters, run.n_noise)
    return evaluate_run(X, adapter)

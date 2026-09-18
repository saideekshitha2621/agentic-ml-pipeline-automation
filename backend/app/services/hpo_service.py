"""Hyperparameter Optimization service — classification and regression.

Reuses each plugin's existing `param_grid(config)` merged-config dict directly as an
sklearn `GridSearchCV`/`RandomizedSearchCV` search space — the values in
`DEFAULT_CLASSIFICATION_HYPERPARAMETER_CONFIG`/`DEFAULT_REGRESSION_HYPERPARAMETER_CONFIG`
(and job overrides) are already `{sklearn_param_name: [values...]}`, exactly the shape
GridSearchCV wants, so no plugin changes are needed. Baseline is the plugin's first
(default) param combination; optimized is the best combination found by k-fold CV search
on the training split, then re-evaluated on the same held-out test split every other stage
uses (so the comparison is apples-to-apples).
"""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from app.core.config import (
    DEFAULT_CLASSIFICATION_HYPERPARAMETER_CONFIG,
    DEFAULT_REGRESSION_HYPERPARAMETER_CONFIG,
    RANDOM_STATE,
)
from app.plugins.classification_base import ClassificationPluginRun
from app.plugins.registry import CLASSIFICATION_PLUGIN_REGISTRY, REGRESSION_PLUGIN_REGISTRY
from app.plugins.regression_base import RegressionPluginRun
from app.services import evaluation_service

RANDOM_SEARCH_THRESHOLD = 30
RANDOM_SEARCH_N_ITER = 15

_PROBLEM_TYPE_CONFIG = {
    "classification": {
        "registry": CLASSIFICATION_PLUGIN_REGISTRY,
        "default_config": DEFAULT_CLASSIFICATION_HYPERPARAMETER_CONFIG,
        "scoring": "f1_macro",
        "run_cls": ClassificationPluginRun,
        "evaluate": evaluation_service.evaluate_classification,
    },
    "regression": {
        "registry": REGRESSION_PLUGIN_REGISTRY,
        "default_config": DEFAULT_REGRESSION_HYPERPARAMETER_CONFIG,
        "scoring": "r2",
        "run_cls": RegressionPluginRun,
        "evaluate": evaluation_service.evaluate_regression,
    },
}


def _grid_size(search_space: dict) -> int:
    size = 1
    for values in search_space.values():
        size *= max(len(values), 1)
    return size


def _build_run(run_cls, algorithm: str, params: dict, model, X_test: np.ndarray):
    if run_cls is ClassificationPluginRun:
        return ClassificationPluginRun(
            algorithm, params, model, model.predict(X_test),
            model.predict_proba(X_test) if hasattr(model, "predict_proba") else None,
        )
    return RegressionPluginRun(algorithm, params, model, model.predict(X_test))


def optimize(
    algorithm: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    config: dict | None = None,
    problem_type: str = "classification",
) -> dict:
    settings = _PROBLEM_TYPE_CONFIG[problem_type]
    plugin = settings["registry"][algorithm]
    config = config or {}
    search_space = {**settings["default_config"].get(algorithm, {}), **config.get(algorithm, {})}
    if not search_space:
        return {"algorithm": algorithm, "skipped": True, "reason": "No hyperparameter search space registered."}

    baseline_params = plugin.param_grid({k: v for k, v in search_space.items()})[0]
    baseline_model = plugin.build_model(baseline_params)
    baseline_model.fit(X_train, y_train)
    baseline_run = _build_run(settings["run_cls"], algorithm, baseline_params, baseline_model, X_test)
    baseline_metrics = settings["evaluate"](y_test, baseline_run)

    cv_folds = 5 if len(X_train) >= 100 else max(2, min(5, len(X_train) // 20 or 2))
    n_combos = _grid_size(search_space)
    estimator = plugin.build_model(baseline_params)
    try:
        if n_combos > RANDOM_SEARCH_THRESHOLD:
            search = RandomizedSearchCV(
                estimator, search_space, n_iter=min(RANDOM_SEARCH_N_ITER, n_combos),
                cv=cv_folds, scoring=settings["scoring"], random_state=RANDOM_STATE, n_jobs=1,
            )
        else:
            search = GridSearchCV(estimator, search_space, cv=cv_folds, scoring=settings["scoring"], n_jobs=1)
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        best_params = search.best_params_
        n_trials = len(search.cv_results_["params"])
    except Exception:
        best_model, best_params, n_trials = baseline_model, baseline_params, 1

    optimized_run = _build_run(settings["run_cls"], algorithm, best_params, best_model, X_test)
    optimized_metrics = settings["evaluate"](y_test, optimized_run)

    return {
        "algorithm": algorithm,
        "skipped": False,
        "baseline_params": baseline_params,
        "baseline_metrics": baseline_metrics,
        "best_params": best_params,
        "optimized_metrics": optimized_metrics,
        "n_trials": n_trials,
        "cv_folds": cv_folds,
        "run": optimized_run,
    }

from __future__ import annotations

from app.plugins.base import ClusteringPlugin
from app.plugins.classification_base import ClassificationPlugin
from app.plugins.regression_base import RegressionPlugin

PLUGIN_REGISTRY: dict[str, ClusteringPlugin] = {}
CLASSIFICATION_PLUGIN_REGISTRY: dict[str, ClassificationPlugin] = {}
REGRESSION_PLUGIN_REGISTRY: dict[str, RegressionPlugin] = {}


def register_plugin(cls):
    instance = cls()
    PLUGIN_REGISTRY[instance.name] = instance
    return cls


def register_classification_plugin(cls):
    instance = cls()
    CLASSIFICATION_PLUGIN_REGISTRY[instance.name] = instance
    return cls


def register_regression_plugin(cls):
    instance = cls()
    REGRESSION_PLUGIN_REGISTRY[instance.name] = instance
    return cls


def get_plugin(name: str) -> ClusteringPlugin:
    return PLUGIN_REGISTRY[name]


def get_classification_plugin(name: str) -> ClassificationPlugin:
    return CLASSIFICATION_PLUGIN_REGISTRY[name]


def get_regression_plugin(name: str) -> RegressionPlugin:
    return REGRESSION_PLUGIN_REGISTRY[name]

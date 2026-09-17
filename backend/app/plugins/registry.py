from __future__ import annotations

from app.plugins.base import ClusteringPlugin
from app.plugins.classification_base import ClassificationPlugin

PLUGIN_REGISTRY: dict[str, ClusteringPlugin] = {}
CLASSIFICATION_PLUGIN_REGISTRY: dict[str, ClassificationPlugin] = {}


def register_plugin(cls):
    instance = cls()
    PLUGIN_REGISTRY[instance.name] = instance
    return cls


def register_classification_plugin(cls):
    instance = cls()
    CLASSIFICATION_PLUGIN_REGISTRY[instance.name] = instance
    return cls


def get_plugin(name: str) -> ClusteringPlugin:
    return PLUGIN_REGISTRY[name]


def get_classification_plugin(name: str) -> ClassificationPlugin:
    return CLASSIFICATION_PLUGIN_REGISTRY[name]

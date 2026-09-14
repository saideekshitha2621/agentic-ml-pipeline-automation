from __future__ import annotations

from app.plugins.base import ClusteringPlugin

PLUGIN_REGISTRY: dict[str, ClusteringPlugin] = {}


def register_plugin(cls):
    instance = cls()
    PLUGIN_REGISTRY[instance.name] = instance
    return cls


def get_plugin(name: str) -> ClusteringPlugin:
    return PLUGIN_REGISTRY[name]

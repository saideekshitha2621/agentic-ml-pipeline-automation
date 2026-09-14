from app.plugins import (  # noqa: F401  (import triggers self-registration)
    birch_plugin,
    dbscan_plugin,
    gmm_plugin,
    hierarchical_plugin,
    kmeans_plugin,
    optics_plugin,
    spectral_plugin,
)
from app.plugins.registry import PLUGIN_REGISTRY, get_plugin, register_plugin

__all__ = ["PLUGIN_REGISTRY", "get_plugin", "register_plugin"]

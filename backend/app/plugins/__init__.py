from app.plugins import (  # noqa: F401  (import triggers self-registration)
    birch_plugin,
    dbscan_plugin,
    gmm_plugin,
    gradient_boosting_classifier_plugin,
    hierarchical_plugin,
    kmeans_plugin,
    knn_classifier_plugin,
    logistic_regression_plugin,
    optics_plugin,
    random_forest_classifier_plugin,
    spectral_plugin,
    svm_classifier_plugin,
)
from app.plugins.registry import (
    CLASSIFICATION_PLUGIN_REGISTRY,
    PLUGIN_REGISTRY,
    get_classification_plugin,
    get_plugin,
    register_classification_plugin,
    register_plugin,
)

__all__ = [
    "PLUGIN_REGISTRY",
    "CLASSIFICATION_PLUGIN_REGISTRY",
    "get_plugin",
    "get_classification_plugin",
    "register_plugin",
    "register_classification_plugin",
]

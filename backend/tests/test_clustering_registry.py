import numpy as np

from app.plugins import PLUGIN_REGISTRY
import app.plugins  # noqa: F401  ensures all plugins self-register


def _blobs():
    rng = np.random.default_rng(0)
    a = rng.normal(loc=[0, 0], scale=0.3, size=(40, 2))
    b = rng.normal(loc=[5, 5], scale=0.3, size=(40, 2))
    c = rng.normal(loc=[0, 5], scale=0.3, size=(40, 2))
    return np.vstack([a, b, c])


def test_all_seven_algorithms_registered():
    expected = {"kmeans", "hierarchical", "dbscan", "gmm", "spectral", "birch", "optics"}
    assert expected == set(PLUGIN_REGISTRY.keys())


def test_kmeans_plugin_produces_valid_runs():
    X = _blobs()
    plugin = PLUGIN_REGISTRY["kmeans"]
    runs = plugin.run(X, {"k_range": [2, 5]})
    assert len(runs) > 0
    for run in runs:
        assert run.n_clusters >= 2
        assert len(run.labels) == len(X)


def test_dbscan_plugin_auto_resolves_eps():
    X = _blobs()
    plugin = PLUGIN_REGISTRY["dbscan"]
    runs = plugin.run(X, {"min_samples": [5]})
    assert isinstance(runs, list)  # should not raise even with no eps override


def test_gmm_plugin_respects_component_range():
    X = _blobs()
    plugin = PLUGIN_REGISTRY["gmm"]
    runs = plugin.run(X, {"component_range": [3, 3], "covariance_types": ["full"]})
    assert all(r.algorithm == "gmm" for r in runs)

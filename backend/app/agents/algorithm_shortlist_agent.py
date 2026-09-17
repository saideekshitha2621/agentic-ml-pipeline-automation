"""Algorithm Recommendation Agent — pre-training shortlist.

Rather than always brute-forcing every registered plugin, propose a shortlist with a
rationale per algorithm so the HITL reviewer can trim it. Reasoning is heuristic
(dataset size, column-type mix) — deliberately simple, since this only decides *which*
algorithms to try, not their hyperparameters (that's the HPO stage).
"""
from __future__ import annotations

from app.plugins.registry import CLASSIFICATION_PLUGIN_REGISTRY, PLUGIN_REGISTRY

LARGE_DATASET_ROWS = 5000

_CLASSIFICATION_RATIONALE = {
    "logistic_regression": "A fast, interpretable linear baseline — good for establishing a performance floor.",
    "random_forest": "Handles nonlinear relationships and mixed numeric/categorical features well, with little tuning.",
    "gradient_boosting": "Often the strongest tabular-data performer, at the cost of longer training time.",
    "knn": "Simple distance-based method — effective when the dataset is small enough to stay fast.",
    "svm": "Strong for high-dimensional or well-separated data, but can be slow on larger datasets.",
}

_CLUSTERING_RATIONALE = {
    "kmeans": "Fast, well-understood baseline for roughly spherical, similarly-sized clusters.",
    "dbscan": "Finds arbitrarily-shaped clusters and flags noise/outliers, without needing a cluster count.",
    "hierarchical": "Produces a dendrogram of nested groupings — useful when cluster count is unclear.",
    "gmm": "Soft, probabilistic clustering — useful when clusters overlap rather than being cleanly separated.",
    "spectral": "Effective for non-convex cluster shapes, at higher computational cost.",
    "birch": "Memory-efficient and scales well to larger datasets.",
    "optics": "Like DBSCAN but tolerant of clusters with varying density.",
}


def recommend(profile: dict, problem_type: str) -> dict:
    n_rows = profile.get("n_rows", 0)
    is_large = n_rows > LARGE_DATASET_ROWS

    if problem_type == "classification":
        registry, rationale_map = CLASSIFICATION_PLUGIN_REGISTRY, _CLASSIFICATION_RATIONALE
    else:
        registry, rationale_map = PLUGIN_REGISTRY, _CLUSTERING_RATIONALE

    shortlist = []
    for name in registry:
        rationale = rationale_map.get(name, "Registered candidate algorithm for this problem type.")
        recommended = True
        if is_large and name in ("knn", "svm", "spectral", "hierarchical"):
            recommended = False
            rationale += f" Not recommended by default — {n_rows:,} rows may make this slow; you can still include it."
        shortlist.append({"algorithm": name, "recommended": recommended, "rationale": rationale})

    return {
        "shortlist": shortlist,
        "selected_algorithms": [s["algorithm"] for s in shortlist if s["recommended"]],
    }


def summarize(result: dict) -> str:
    n_selected = len(result["selected_algorithms"])
    n_total = len(result["shortlist"])
    return f"Recommending {n_selected} of {n_total} registered algorithm(s) based on dataset size and type."

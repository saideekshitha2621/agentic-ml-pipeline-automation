"""HITL Approval screen: top-3 models with plain-language strengths/weaknesses."""
from __future__ import annotations

import numpy as np


def strengths_and_weaknesses(run: dict) -> tuple[list[str], list[str]]:
    strengths, weaknesses = [], []
    sil = run.get("silhouette_score")
    db = run.get("davies_bouldin_score")
    ch = run.get("calinski_harabasz_score")

    if sil is not None:
        if sil > 0.5:
            strengths.append(f"Strong cluster separation (silhouette {sil:.3f}).")
        elif sil < 0.25:
            weaknesses.append(f"Weak cluster separation (silhouette {sil:.3f}).")

    if db is not None:
        if db < 0.7:
            strengths.append(f"Compact, well-separated clusters (Davies-Bouldin {db:.3f}).")
        elif db > 1.2:
            weaknesses.append(f"Clusters overlap or are dispersed (Davies-Bouldin {db:.3f}).")

    if ch is not None:
        strengths.append(f"Calinski-Harabasz of {ch:,.1f} indicates dense, well-separated groupings.")

    noise_pct = run.get("noise_pct")
    if run.get("n_noise", 0) > 0:
        if noise_pct is not None and noise_pct >= 30:
            weaknesses.append(
                f"⚠️ {run['n_noise']} points ({noise_pct:.0f}% of the dataset) classified as noise / "
                "unassigned — the metrics above only reflect the remaining points, not the full data."
            )
        else:
            weaknesses.append(f"{run['n_noise']} points classified as noise / unassigned.")

    n_clusters = run.get("n_clusters", 0)
    if n_clusters <= 1:
        weaknesses.append("Produced too few clusters to be actionable.")
    elif n_clusters > 15:
        weaknesses.append(f"Produced {n_clusters} clusters, which may be too granular for business use.")
    else:
        strengths.append(f"Produced {n_clusters} clusters — a manageable number for segmentation.")

    if not strengths:
        strengths.append("Completed successfully with valid cluster assignments.")
    return strengths, weaknesses


def cluster_size_breakdown(labels: np.ndarray) -> dict[str, int]:
    unique, counts = np.unique(labels, return_counts=True)
    return {("noise" if u == -1 else str(int(u))): int(c) for u, c in zip(unique, counts)}

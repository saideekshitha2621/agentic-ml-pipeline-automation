"""Builds a leaderboard from evaluated clustering runs and ranks them."""
from __future__ import annotations

import pandas as pd


def build_leaderboard(evaluated_runs: list[dict], noise_penalty_cap: float = 0.5) -> pd.DataFrame:
    """evaluated_runs: list of dicts merging ClusterRun info + evaluate_run() output.

    Ranking uses a composite score: higher silhouette is better, lower Davies-Bouldin
    is better, higher Calinski-Harabasz is better. Each metric is min-max normalized
    across valid (non-null) runs, then averaged with equal weight, so no single metric
    dominates just because of its natural scale.

    The raw metrics are computed only on non-noise points (see evaluate_run), so a run
    that discards most of the dataset as noise (e.g. DBSCAN/OPTICS with aggressive
    params) can still score deceptively well on the points it kept. To prevent that from
    outranking a run that actually clusters the whole dataset, the composite score is
    scaled down by `(1 - noise_ratio)`. `noise_penalty_cap` additionally excludes runs
    whose noise ratio exceeds it from ranking altogether (default: 50% of points).
    """
    rows = []
    for r in evaluated_runs:
        n_noise = r.get("n_noise", 0)
        n_evaluated = r.get("n_samples_evaluated", 0) or 0
        total = n_noise + n_evaluated
        noise_ratio = (n_noise / total) if total else 0.0
        rows.append(
            {
                "algorithm": r["algorithm"],
                "params": r["params"],
                "n_clusters": r["n_clusters"],
                "n_noise": n_noise,
                "noise_pct": round(noise_ratio * 100, 2),
                "silhouette_score": r["silhouette_score"],
                "davies_bouldin_score": r["davies_bouldin_score"],
                "calinski_harabasz_score": r["calinski_harabasz_score"],
                "run_id": r["run_id"],
            }
        )
    df = pd.DataFrame(rows)
    valid = df.dropna(subset=["silhouette_score", "davies_bouldin_score", "calinski_harabasz_score"]).copy()
    valid = valid[valid["noise_pct"] / 100 <= noise_penalty_cap]

    if valid.empty:
        df["composite_score"] = None
        df["rank"] = None
        return df.sort_values("algorithm").reset_index(drop=True)

    def norm(series: pd.Series, invert: bool = False) -> pd.Series:
        lo, hi = series.min(), series.max()
        if hi == lo:
            out = pd.Series(1.0, index=series.index)
        else:
            out = (series - lo) / (hi - lo)
        return 1 - out if invert else out

    valid["sil_norm"] = norm(valid["silhouette_score"])
    valid["db_norm"] = norm(valid["davies_bouldin_score"], invert=True)
    valid["ch_norm"] = norm(valid["calinski_harabasz_score"])
    valid["noise_penalty_factor"] = 1 - (valid["noise_pct"] / 100)
    valid["composite_score"] = (
        (valid["sil_norm"] + valid["db_norm"] + valid["ch_norm"]) / 3
    ) * valid["noise_penalty_factor"]
    valid = valid.sort_values("composite_score", ascending=False).reset_index(drop=True)
    valid["rank"] = valid.index + 1

    df = df.merge(
        valid[["run_id", "composite_score", "rank"]], on="run_id", how="left"
    )
    df = df.sort_values(["rank"], na_position="last").reset_index(drop=True)
    return df.drop(columns=["sil_norm", "db_norm", "ch_norm", "noise_penalty_factor"], errors="ignore")


def top_n(leaderboard: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    return leaderboard.dropna(subset=["rank"]).sort_values("rank").head(n).reset_index(drop=True)


def explain_ranking(row: pd.Series) -> str:
    """Human-readable rationale for why a model ranked where it did."""
    reasons = []
    sil = row["silhouette_score"]
    db = row["davies_bouldin_score"]
    ch = row["calinski_harabasz_score"]

    if sil is not None:
        if sil > 0.5:
            reasons.append(f"strong cluster separation (silhouette={sil:.3f})")
        elif sil > 0.25:
            reasons.append(f"moderate cluster separation (silhouette={sil:.3f})")
        else:
            reasons.append(f"weak cluster separation (silhouette={sil:.3f})")
    if db is not None:
        reasons.append(f"cluster compactness/Davies-Bouldin of {db:.3f} (lower is better)")
    if ch is not None:
        reasons.append(f"Calinski-Harabasz of {ch:,.1f} (higher indicates denser, well-separated clusters)")

    n_clusters = row["n_clusters"]
    reasons.append(f"produced {n_clusters} clusters")
    noise_pct = row.get("noise_pct")
    if row.get("n_noise", 0):
        if noise_pct is not None and noise_pct >= 30:
            reasons.append(
                f"⚠️ {int(row['n_noise'])} points ({noise_pct:.0f}% of the data) flagged as noise — "
                "metrics above are computed only on the remaining points"
            )
        else:
            reasons.append(f"with {int(row['n_noise'])} points flagged as noise")

    return f"{row['algorithm']} ({row['params']}) ranked #{int(row['rank'])}: " + "; ".join(reasons) + "."

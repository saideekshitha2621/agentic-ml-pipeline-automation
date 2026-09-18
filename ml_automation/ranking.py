"""Builds a leaderboard from evaluated runs and ranks them.

The normalize-average-rank algorithm here has nothing clustering-specific about it — it
only needs a list of metric names and, for each, whether higher or lower is better.
METRIC_SPECS makes that the one thing that varies per problem_type, so a future
classification or regression metric set (e.g. {"f1_macro": invert=False, "log_loss":
invert=True}) plugs into the same function instead of a copy-pasted one. Today only
"clustering" has an entry — see ARCHITECTURE.md / the ModelRun design notes for why
classification/regression aren't wired up yet.
"""
from __future__ import annotations

import pandas as pd

METRIC_SPECS: dict[str, dict[str, bool]] = {
    # metric name -> invert (True when a *lower* raw value should rank higher, e.g. Davies-Bouldin)
    "clustering": {
        "silhouette_score": False,
        "davies_bouldin_score": True,
        "calinski_harabasz_score": False,
    },
    # accuracy/f1_macro are always computed (see classification_evaluation.evaluate_run);
    # roc_auc/log_loss are left out of the composite deliberately — they're None whenever a
    # test split doesn't have every class represented (common on the small datasets this
    # platform targets), and dropna(subset=metric_names) below would drop the whole run.
    "classification": {
        "f1_macro": False,
        "accuracy": False,
    },
    # rmse/mae are lower-is-better (inverted); r2 is higher-is-better. mape is left out of
    # the composite for the same reason roc_auc/log_loss are left out of classification —
    # it's undefined when a true value is 0, which would drop the whole run via dropna().
    "regression": {
        "rmse": True,
        "mae": True,
        "r2": False,
    },
}


def build_leaderboard(
    evaluated_runs: list[dict], noise_penalty_cap: float = 0.5, problem_type: str = "clustering"
) -> pd.DataFrame:
    """evaluated_runs: list of dicts merging ModelRun info + evaluate_run() output.

    Ranking uses a composite score: each metric in METRIC_SPECS[problem_type] is
    min-max normalized across valid (non-null) runs (inverted first when lower-is-better),
    then averaged with equal weight, so no single metric dominates just because of its
    natural scale.

    The raw metrics are computed only on non-noise points (see evaluate_run), so a run
    that discards most of the dataset as noise (e.g. DBSCAN/OPTICS with aggressive
    params) can still score deceptively well on the points it kept. To prevent that from
    outranking a run that actually clusters the whole dataset, the composite score is
    scaled down by `(1 - noise_ratio)`. `noise_penalty_cap` additionally excludes runs
    whose noise ratio exceeds it from ranking altogether (default: 50% of points).
    The noise penalty is clustering-specific and a no-op (noise_pct always 0) for any
    future problem_type that doesn't produce noise points.
    """
    metric_spec = METRIC_SPECS[problem_type]
    metric_names = list(metric_spec)
    rows = []
    for r in evaluated_runs:
        n_noise = r.get("n_noise", 0)
        n_evaluated = r.get("n_samples_evaluated", 0) or 0
        total = n_noise + n_evaluated
        noise_ratio = (n_noise / total) if total else 0.0
        row = {
            "algorithm": r["algorithm"],
            "params": r["params"],
            "n_clusters": r.get("n_clusters", 0),
            "n_noise": n_noise,
            "noise_pct": round(noise_ratio * 100, 2),
            "run_id": r["run_id"],
        }
        for name in metric_names:
            row[name] = r[name]
        rows.append(row)
    df = pd.DataFrame(rows)
    valid = df.dropna(subset=metric_names).copy()
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

    norm_cols = []
    for name, invert in metric_spec.items():
        col = f"_{name}_norm"
        valid[col] = norm(valid[name], invert=invert)
        norm_cols.append(col)

    valid["noise_penalty_factor"] = 1 - (valid["noise_pct"] / 100)
    valid["composite_score"] = valid[norm_cols].mean(axis=1) * valid["noise_penalty_factor"]
    valid = valid.sort_values("composite_score", ascending=False).reset_index(drop=True)
    valid["rank"] = valid.index + 1

    df = df.merge(
        valid[["run_id", "composite_score", "rank"]], on="run_id", how="left"
    )
    df = df.sort_values(["rank"], na_position="last").reset_index(drop=True)
    return df.drop(columns=norm_cols + ["noise_penalty_factor"], errors="ignore")


def top_n(leaderboard: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    return leaderboard.dropna(subset=["rank"]).sort_values("rank").head(n).reset_index(drop=True)


def _explain_clustering(row: pd.Series) -> str:
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


def _explain_classification(row: pd.Series) -> str:
    reasons = []
    f1 = row.get("f1_macro")
    acc = row.get("accuracy")
    roc = row.get("roc_auc")

    if f1 is not None:
        if f1 > 0.85:
            reasons.append(f"strong, balanced performance across classes (F1={f1:.3f})")
        elif f1 > 0.6:
            reasons.append(f"moderate performance across classes (F1={f1:.3f})")
        else:
            reasons.append(f"weak performance across classes (F1={f1:.3f})")
    if acc is not None:
        reasons.append(f"{acc:.1%} accuracy on the held-out test set")
    if roc is not None:
        reasons.append(f"ROC-AUC of {roc:.3f}")
    else:
        reasons.append("ROC-AUC unavailable (test split didn't cover every class)")

    return f"{row['algorithm']} ({row['params']}) ranked #{int(row['rank'])}: " + "; ".join(reasons) + "."


def _explain_regression(row: pd.Series) -> str:
    reasons = []
    r2 = row.get("r2")
    rmse = row.get("rmse")
    mae = row.get("mae")

    if r2 is not None:
        if r2 > 0.8:
            reasons.append(f"explains most of the variance in the target (R²={r2:.3f})")
        elif r2 > 0.5:
            reasons.append(f"explains a moderate share of the variance in the target (R²={r2:.3f})")
        else:
            reasons.append(f"explains little of the variance in the target (R²={r2:.3f})")
    if rmse is not None:
        reasons.append(f"typical prediction error (RMSE) of {rmse:.3f}")
    if mae is not None:
        reasons.append(f"average absolute error (MAE) of {mae:.3f}")

    return f"{row['algorithm']} ({row['params']}) ranked #{int(row['rank'])}: " + "; ".join(reasons) + "."


def explain_ranking(row: pd.Series, problem_type: str = "clustering") -> str:
    """Human-readable rationale for why a model ranked where it did."""
    if problem_type == "classification":
        return _explain_classification(row)
    if problem_type == "regression":
        return _explain_regression(row)
    return _explain_clustering(row)

"""Turns a chosen clustering into human-readable cluster profiles and names."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def cluster_sizes(labels: np.ndarray) -> pd.DataFrame:
    series = pd.Series(labels, name="cluster")
    sizes = series.value_counts().sort_index()
    pct = (sizes / sizes.sum() * 100).round(2)
    return pd.DataFrame({"cluster": sizes.index, "count": sizes.values, "pct_of_total": pct.values})


def cluster_profiles(original_df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Mean (numeric) / mode (categorical) feature values per cluster, plus overall baseline."""
    df = original_df.copy()
    df["cluster"] = labels
    df = df[df["cluster"] != -1]  # drop DBSCAN noise from profiling

    numeric_cols = [c for c in original_df.columns if pd.api.types.is_numeric_dtype(original_df[c])]
    cat_cols = [c for c in original_df.columns if c not in numeric_cols]

    profiles = []
    for cluster_id, group in df.groupby("cluster"):
        row = {"cluster": cluster_id, "size": len(group)}
        for col in numeric_cols:
            row[col] = round(float(group[col].mean()), 3)
        for col in cat_cols:
            mode = group[col].mode(dropna=True)
            row[col] = mode.iloc[0] if not mode.empty else None
        profiles.append(row)

    profile_df = pd.DataFrame(profiles)

    overall = {"cluster": "overall", "size": len(df)}
    for col in numeric_cols:
        overall[col] = round(float(df[col].mean()), 3)
    for col in cat_cols:
        mode = df[col].mode(dropna=True)
        overall[col] = mode.iloc[0] if not mode.empty else None

    return pd.concat([profile_df, pd.DataFrame([overall])], ignore_index=True)


def feature_importance_per_cluster(
    encoded_df: pd.DataFrame, labels: np.ndarray, random_state: int = 42
) -> pd.DataFrame:
    """Trains a one-vs-rest RandomForest per cluster to rank which features most
    distinguish that cluster from the rest of the data (a standard, model-agnostic
    proxy for "feature importance" in unsupervised settings)."""
    mask = labels != -1
    X = encoded_df.loc[mask]
    y = labels[mask]

    results = []
    for cluster_id in sorted(set(y)):
        binary_target = (y == cluster_id).astype(int)
        if binary_target.sum() == 0 or binary_target.sum() == len(binary_target):
            continue
        clf = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=random_state)
        clf.fit(X, binary_target)
        importances = pd.Series(clf.feature_importances_, index=X.columns).sort_values(ascending=False)
        for rank, (feature, importance) in enumerate(importances.head(5).items(), start=1):
            results.append(
                {
                    "cluster": cluster_id,
                    "rank": rank,
                    "feature": feature,
                    "importance": round(float(importance), 4),
                }
            )
    return pd.DataFrame(results)


def business_summary(
    profile_row: pd.Series, overall_row: pd.Series, top_features: pd.DataFrame, threshold: float = 0.15
) -> str:
    """Plain-language description of how a cluster differs from the overall population."""
    diffs = []
    numeric_cols = [
        c
        for c in profile_row.index
        if c not in ("cluster", "size") and pd.api.types.is_number(profile_row[c])
    ]
    for col in numeric_cols:
        overall_val = overall_row.get(col)
        cluster_val = profile_row.get(col)
        if overall_val in (None, 0) or pd.isna(overall_val) or pd.isna(cluster_val):
            continue
        rel_diff = (cluster_val - overall_val) / (abs(overall_val) + 1e-9)
        if abs(rel_diff) >= threshold:
            direction = "higher" if rel_diff > 0 else "lower"
            diffs.append((abs(rel_diff), f"{col} is {abs(rel_diff) * 100:.0f}% {direction} than average"))

    diffs.sort(key=lambda x: x[0], reverse=True)
    highlights = [text for _, text in diffs[:4]]

    cluster_features = (
        top_features[top_features["cluster"] == profile_row["cluster"]]["feature"].tolist()
        if not top_features.empty
        else []
    )

    sentence = f"This segment ({int(profile_row['size'])} records) "
    if highlights:
        sentence += "stands out because " + "; ".join(highlights) + "."
    else:
        sentence += "is close to the overall population average across most features."
    if cluster_features:
        sentence += f" The features that most distinguish this group are: {', '.join(cluster_features)}."
    return sentence


def suggest_cluster_name(profile_row: pd.Series, overall_row: pd.Series, threshold: float = 0.15) -> str:
    """Heuristic short label built from the most extreme numeric deviations."""
    numeric_cols = [
        c
        for c in profile_row.index
        if c not in ("cluster", "size") and pd.api.types.is_number(profile_row[c])
    ]
    scored = []
    for col in numeric_cols:
        overall_val = overall_row.get(col)
        cluster_val = profile_row.get(col)
        if overall_val in (None, 0) or pd.isna(overall_val) or pd.isna(cluster_val):
            continue
        rel_diff = (cluster_val - overall_val) / (abs(overall_val) + 1e-9)
        if abs(rel_diff) >= threshold:
            label = "High" if rel_diff > 0 else "Low"
            scored.append((abs(rel_diff), f"{label} {col}"))

    scored.sort(key=lambda x: x[0], reverse=True)
    if not scored:
        return f"Cluster {profile_row['cluster']} (Baseline Segment)"
    parts = [text for _, text in scored[:2]]
    return " / ".join(parts)


def interpret_clusters(
    original_df: pd.DataFrame, encoded_df: pd.DataFrame, labels: np.ndarray
) -> dict:
    sizes = cluster_sizes(labels)
    profiles = cluster_profiles(original_df, labels)
    importances = feature_importance_per_cluster(encoded_df, labels)

    overall_row = profiles[profiles["cluster"] == "overall"].iloc[0]
    summaries, names = {}, {}
    for _, row in profiles[profiles["cluster"] != "overall"].iterrows():
        summaries[row["cluster"]] = business_summary(row, overall_row, importances)
        names[row["cluster"]] = suggest_cluster_name(row, overall_row)

    return {
        "sizes": sizes,
        "profiles": profiles,
        "feature_importance": importances,
        "summaries": summaries,
        "suggested_names": names,
    }

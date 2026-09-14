"""Visualizations and downloadable export bundle for the final chosen model."""
from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def plot_cluster_scatter(X: np.ndarray, labels: np.ndarray, title: str = "Cluster Visualization"):
    """2D scatter of the clustering result. If X has >2 dims, project with PCA first."""
    if X.shape[1] > 2:
        coords = PCA(n_components=2, random_state=42).fit_transform(X)
        xlabel, ylabel = "PC1", "PC2"
    else:
        coords = X
        xlabel, ylabel = "Feature 1", "Feature 2"

    fig, ax = plt.subplots(figsize=(7, 5.5))
    unique_labels = sorted(set(labels))
    cmap = plt.get_cmap("tab10")
    for i, lab in enumerate(unique_labels):
        mask = labels == lab
        color = "lightgray" if lab == -1 else cmap(i % 10)
        name = "Noise" if lab == -1 else f"Cluster {lab}"
        ax.scatter(coords[mask, 0], coords[mask, 1], s=18, alpha=0.7, label=name, color=color)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    return fig


def plot_cluster_sizes(sizes_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(sizes_df["cluster"].astype(str), sizes_df["count"], color="#4C72B0")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Count")
    ax.set_title("Cluster Sizes")
    fig.tight_layout()
    return fig


def plot_leaderboard_metric(leaderboard: pd.DataFrame, metric: str = "composite_score", top_n: int = 10):
    df = leaderboard.dropna(subset=[metric]).sort_values(metric, ascending=False).head(top_n)
    labels = [f"{r.algorithm}\n{r.params}" for r in df.itertuples()]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(labels[::-1], df[metric][::-1], color="#55A868")
    ax.set_xlabel(metric.replace("_", " ").title())
    ax.set_title("Model Leaderboard")
    fig.tight_layout()
    return fig


def plot_explained_variance(explained_variance_ratio: list[float]):
    fig, ax = plt.subplots(figsize=(6, 4))
    cum = np.cumsum(explained_variance_ratio)
    ax.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, alpha=0.6, label="Individual")
    ax.plot(range(1, len(explained_variance_ratio) + 1), cum, color="red", marker="o", label="Cumulative")
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Explained Variance Ratio")
    ax.set_title("PCA Explained Variance")
    ax.legend()
    fig.tight_layout()
    return fig


def build_results_bundle(
    output_dir: str | Path,
    preprocessing_report: dict,
    leaderboard: pd.DataFrame,
    chosen_model_info: dict,
    interpretation: dict,
    clustered_df: pd.DataFrame,
    figures: dict[str, "plt.Figure"] | None = None,
) -> Path:
    """Writes CSV/JSON/PNG artifacts and zips them into one downloadable file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "preprocessing_report.json").write_text(
        json.dumps(preprocessing_report, indent=2, default=str)
    )
    leaderboard.to_csv(output_dir / "leaderboard.csv", index=False)
    (output_dir / "chosen_model.json").write_text(json.dumps(chosen_model_info, indent=2, default=str))
    clustered_df.to_csv(output_dir / "clustered_data.csv", index=False)
    interpretation["profiles"].to_csv(output_dir / "cluster_profiles.csv", index=False)
    interpretation["sizes"].to_csv(output_dir / "cluster_sizes.csv", index=False)
    if not interpretation["feature_importance"].empty:
        interpretation["feature_importance"].to_csv(output_dir / "feature_importance.csv", index=False)
    (output_dir / "cluster_summaries.json").write_text(
        json.dumps(
            {
                "suggested_names": interpretation["suggested_names"],
                "summaries": interpretation["summaries"],
            },
            indent=2,
            default=str,
        )
    )

    if figures:
        for name, fig in figures.items():
            fig.savefig(output_dir / f"{name}.png", dpi=150)

    zip_path = output_dir / "results_bundle.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file in output_dir.iterdir():
            if file.name == "results_bundle.zip":
                continue
            zf.write(file, arcname=file.name)

    return zip_path


def fig_to_bytes(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    return buf.read()

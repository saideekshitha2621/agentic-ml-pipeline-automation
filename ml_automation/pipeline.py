"""Orchestrates preprocessing -> (optional PCA) -> clustering -> evaluation -> ranking
automatically, then stops and hands control to a human for final model selection and
cluster interpretation review.

Typical flow:
    pipeline = AutoClusteringPipeline(df)
    plan = pipeline.build_preprocessing_plan()      # human may edit `plan` here
    pipeline.preprocess(plan)
    pipeline.review_pca(apply=True, n_components=0.95)   # human decides whether to call this
    leaderboard = pipeline.train_and_rank()          # fully automatic
    top3 = pipeline.recommend(n=3)                    # human reads these + rationale
    pipeline.select_model(run_id=top3.iloc[0]["run_id"])  # human's final choice
    interpretation = pipeline.interpret()
    zip_path = pipeline.export("output/")
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from .clustering import ClusterRun, HyperparameterGrid, run_all_algorithms
from .dimensionality_reduction import apply_pca, explained_variance_curve
from .evaluation import evaluate_run
from .interpretation import interpret_clusters
from .preprocessing import DataPreprocessor, PreprocessingPlan
from .ranking import build_leaderboard, explain_ranking, top_n as top_n_fn
from .reporting import (
    build_results_bundle,
    plot_cluster_scatter,
    plot_cluster_sizes,
    plot_explained_variance,
    plot_leaderboard_metric,
)


class AutoClusteringPipeline:
    def __init__(self, df: pd.DataFrame, id_columns: Optional[list[str]] = None, random_state: int = 42):
        self.df = df
        self.random_state = random_state
        self.preprocessor = DataPreprocessor(df, id_columns=id_columns)

        self.plan: Optional[PreprocessingPlan] = None
        self.preprocessing_report: Optional[dict] = None
        self.cleaned_df: Optional[pd.DataFrame] = None
        self.encoded_df: Optional[pd.DataFrame] = None

        self.pca_applied = False
        self.pca_report: Optional[dict] = None
        self.model_df: Optional[pd.DataFrame] = None  # what clustering actually trains on

        self.runs: list[ClusterRun] = []
        self.evaluated_runs: list[dict] = []
        self.leaderboard: Optional[pd.DataFrame] = None

        self.chosen_run: Optional[dict] = None
        self.interpretation: Optional[dict] = None

    # ------------------------------------------------------------------ #
    # 1. Preprocessing (+ optional human review)
    # ------------------------------------------------------------------ #
    def build_preprocessing_plan(self) -> PreprocessingPlan:
        self.plan = self.preprocessor.build_default_plan()
        return self.plan

    def missing_value_summary(self) -> pd.DataFrame:
        return self.preprocessor.missing_value_summary()

    def preprocess(self, plan: Optional[PreprocessingPlan] = None) -> dict:
        self.plan = plan or self.plan or self.build_preprocessing_plan()
        self.cleaned_df, self.encoded_df, self.preprocessing_report = self.preprocessor.run(self.plan)
        self.model_df = self.encoded_df
        return self.preprocessing_report

    # ------------------------------------------------------------------ #
    # 2. Optional PCA (human decides whether to apply it)
    # ------------------------------------------------------------------ #
    def explained_variance_preview(self) -> pd.DataFrame:
        assert self.encoded_df is not None, "call preprocess() first"
        return explained_variance_curve(self.encoded_df)

    def review_pca(self, apply: bool, n_components: int | float | None = None) -> Optional[dict]:
        assert self.encoded_df is not None, "call preprocess() first"
        if not apply:
            self.pca_applied = False
            self.model_df = self.encoded_df
            return None
        pca_df, report = apply_pca(self.encoded_df, n_components=n_components, random_state=self.random_state)
        self.pca_applied = True
        self.pca_report = report
        self.model_df = pca_df
        return report

    # ------------------------------------------------------------------ #
    # 3-6. Clustering + hyperparameter search + evaluation + ranking (automatic)
    # ------------------------------------------------------------------ #
    def train_and_rank(self, grid: Optional[HyperparameterGrid] = None) -> pd.DataFrame:
        assert self.model_df is not None, "call preprocess() [and optionally review_pca()] first"
        X = self.model_df.values

        self.runs = run_all_algorithms(X, grid=grid, random_state=self.random_state)

        self.evaluated_runs = []
        for i, run in enumerate(self.runs):
            metrics = evaluate_run(X, run)
            self.evaluated_runs.append(
                {
                    "run_id": i,
                    "algorithm": run.algorithm,
                    "params": run.params,
                    "n_clusters": run.n_clusters,
                    "n_noise": run.n_noise,
                    "labels": run.labels,
                    "model": run.model,
                    **metrics,
                }
            )

        self.leaderboard = build_leaderboard(self.evaluated_runs)
        return self.leaderboard

    # ------------------------------------------------------------------ #
    # 7-8. Recommendations + Human-in-the-loop final selection
    # ------------------------------------------------------------------ #
    def recommend(self, n: int = 3) -> pd.DataFrame:
        assert self.leaderboard is not None, "call train_and_rank() first"
        top = top_n_fn(self.leaderboard, n=n)
        top = top.copy()
        top["rationale"] = top.apply(explain_ranking, axis=1)
        return top

    def get_run(self, run_id: int) -> dict:
        matches = [r for r in self.evaluated_runs if r["run_id"] == run_id]
        if not matches:
            raise ValueError(f"No run with run_id={run_id}")
        return matches[0]

    def select_model(self, run_id: int) -> dict:
        """The human's final choice. Nothing before this point commits to a model."""
        self.chosen_run = self.get_run(run_id)
        return self.chosen_run

    # ------------------------------------------------------------------ #
    # 9. Cluster interpretation (runs after human selects a model)
    # ------------------------------------------------------------------ #
    def interpret(self) -> dict:
        assert self.chosen_run is not None, "call select_model() first"
        labels = self.chosen_run["labels"]
        self.interpretation = interpret_clusters(self.cleaned_df, self.encoded_df, labels)
        return self.interpretation

    # ------------------------------------------------------------------ #
    # 10. Final output: report, visualizations, downloadable bundle
    # ------------------------------------------------------------------ #
    def export(self, output_dir: str) -> str:
        assert self.chosen_run is not None and self.interpretation is not None
        labels = self.chosen_run["labels"]
        X = self.model_df.values

        figures = {
            "cluster_scatter": plot_cluster_scatter(
                X, labels, title=f"{self.chosen_run['algorithm']} {self.chosen_run['params']}"
            ),
            "cluster_sizes": plot_cluster_sizes(self.interpretation["sizes"]),
            "leaderboard": plot_leaderboard_metric(self.leaderboard),
        }
        if self.pca_applied and self.pca_report:
            figures["explained_variance"] = plot_explained_variance(
                self.pca_report["explained_variance_ratio"]
            )

        clustered_df = self.cleaned_df.copy()
        clustered_df["cluster"] = labels
        clustered_df["cluster_name"] = clustered_df["cluster"].map(
            self.interpretation["suggested_names"]
        ).fillna("Noise")

        chosen_model_info = {
            "algorithm": self.chosen_run["algorithm"],
            "params": self.chosen_run["params"],
            "n_clusters": self.chosen_run["n_clusters"],
            "n_noise": self.chosen_run["n_noise"],
            "silhouette_score": self.chosen_run["silhouette_score"],
            "davies_bouldin_score": self.chosen_run["davies_bouldin_score"],
            "calinski_harabasz_score": self.chosen_run["calinski_harabasz_score"],
            "pca_applied": self.pca_applied,
            "pca_report": self.pca_report,
        }

        zip_path = build_results_bundle(
            output_dir=output_dir,
            preprocessing_report=self.preprocessing_report,
            leaderboard=self.leaderboard.drop(columns=["labels", "model"], errors="ignore"),
            chosen_model_info=chosen_model_info,
            interpretation=self.interpretation,
            clustered_df=clustered_df,
            figures=figures,
        )
        return str(zip_path)

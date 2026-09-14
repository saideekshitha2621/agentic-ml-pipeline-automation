"""Streamlit UI for the Unsupervised ML Automation Pipeline with Human-in-the-Loop review.

Run with:
    streamlit run app.py
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from ml_automation import AutoClusteringPipeline
from ml_automation.clustering import HyperparameterGrid
from ml_automation.preprocessing import PreprocessingPlan
from ml_automation.reporting import plot_cluster_scatter, plot_cluster_sizes, plot_explained_variance

st.set_page_config(page_title="Unsupervised ML Automation", layout="wide")
st.title("🔍 Unsupervised ML Automation Pipeline")
st.caption("Automated preprocessing, clustering, evaluation & ranking — with the final model choice left to you.")

if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
if "stage" not in st.session_state:
    st.session_state.stage = "upload"


def reset():
    for key in ["pipeline", "stage", "plan", "leaderboard", "chosen_run_id", "interpretation"]:
        st.session_state.pop(key, None)
    st.session_state.stage = "upload"


with st.sidebar:
    st.header("Pipeline Progress")
    stages = ["upload", "preprocess_review", "pca_review", "train_rank", "select_model", "interpret", "export"]
    labels = {
        "upload": "1. Upload Data",
        "preprocess_review": "2. Preprocessing Review",
        "pca_review": "3. Dimensionality Reduction",
        "train_rank": "4-7. Train, Evaluate & Rank",
        "select_model": "8. Human Model Selection",
        "interpret": "9. Cluster Interpretation",
        "export": "10. Final Output",
    }
    current_idx = stages.index(st.session_state.stage) if st.session_state.stage in stages else 0
    for i, s in enumerate(stages):
        marker = "✅" if i < current_idx else ("➡️" if i == current_idx else "⬜")
        st.write(f"{marker} {labels[s]}")
    st.divider()
    if st.button("Start Over"):
        reset()
        st.rerun()

# ---------------------------------------------------------------------- #
# Stage 1: Upload
# ---------------------------------------------------------------------- #
if st.session_state.stage == "upload":
    uploaded = st.file_uploader("Upload a CSV file", type=["csv"])
    use_sample = st.checkbox("...or use the bundled synthetic sample dataset")

    if use_sample:
        from sample_data import make_sample_dataset

        df = make_sample_dataset()
        st.session_state.raw_df = df
    elif uploaded is not None:
        st.session_state.raw_df = pd.read_csv(uploaded)

    if "raw_df" in st.session_state:
        df = st.session_state.raw_df
        st.write(f"Loaded {df.shape[0]} rows x {df.shape[1]} columns")
        st.dataframe(df.head(20), use_container_width=True)

        id_cols = st.multiselect("ID / columns to exclude from modeling", options=list(df.columns))
        if st.button("Continue to Preprocessing Review", type="primary"):
            st.session_state.pipeline = AutoClusteringPipeline(df, id_columns=id_cols)
            st.session_state.plan = st.session_state.pipeline.build_preprocessing_plan()
            st.session_state.stage = "preprocess_review"
            st.rerun()

# ---------------------------------------------------------------------- #
# Stage 2: Preprocessing review (human can override detected types / strategies)
# ---------------------------------------------------------------------- #
elif st.session_state.stage == "preprocess_review":
    pipeline = st.session_state.pipeline
    st.subheader("Preprocessing Review")

    st.write("**Missing value summary**")
    st.dataframe(pipeline.missing_value_summary(), use_container_width=True)

    all_cols = list(st.session_state.raw_df.columns)
    default_plan: PreprocessingPlan = st.session_state.plan

    col1, col2 = st.columns(2)
    with col1:
        numerical = st.multiselect("Numerical columns", options=all_cols, default=default_plan.numerical_columns)
    with col2:
        remaining = [c for c in all_cols if c not in numerical and c not in default_plan.dropped_columns]
        categorical = st.multiselect("Categorical columns", options=remaining, default=[
            c for c in default_plan.categorical_columns if c in remaining
        ])

    num_strategy = st.selectbox("Numerical imputation strategy", ["median", "mean"], index=0)
    cat_strategy = st.selectbox("Categorical imputation strategy", ["mode"], index=0)
    drop_dupes = st.checkbox("Remove duplicate records", value=True)
    scale = st.checkbox("Apply StandardScaler feature scaling", value=True)

    if st.button("Apply Preprocessing", type="primary"):
        plan = PreprocessingPlan(
            numerical_columns=numerical,
            categorical_columns=categorical,
            dropped_columns=default_plan.dropped_columns,
            numerical_impute_strategy=num_strategy,
            categorical_impute_strategy=cat_strategy,
            drop_duplicates=drop_dupes,
            scale_features=scale,
        )
        report = pipeline.preprocess(plan)
        st.session_state.preprocessing_report = report
        st.session_state.stage = "pca_review"
        st.rerun()

# ---------------------------------------------------------------------- #
# Stage 3: PCA review
# ---------------------------------------------------------------------- #
elif st.session_state.stage == "pca_review":
    pipeline = st.session_state.pipeline
    st.subheader("Dimensionality Reduction (Optional)")

    report = st.session_state.preprocessing_report
    st.success(
        f"Preprocessing complete. Final shape: {report['final_shape']}. "
        f"Removed {report['duplicates_removed']} duplicate rows. "
        f"Imputed {len(report['imputation'])} columns with missing values."
    )
    with st.expander("Full preprocessing report"):
        st.json(report)

    curve = pipeline.explained_variance_preview()
    fig = plot_explained_variance(curve["explained_variance_ratio"].tolist())
    st.pyplot(fig)

    apply_pca = st.checkbox("Apply PCA before clustering", value=False)
    variance_target = st.slider("Variance to retain (if PCA applied)", 0.5, 0.99, 0.95)

    if st.button("Continue to Model Training", type="primary"):
        pca_report = pipeline.review_pca(apply=apply_pca, n_components=variance_target if apply_pca else None)
        st.session_state.pca_report = pca_report
        st.session_state.stage = "train_rank"
        st.rerun()

# ---------------------------------------------------------------------- #
# Stage 4-7: Automatic training, evaluation, ranking
# ---------------------------------------------------------------------- #
elif st.session_state.stage == "train_rank":
    pipeline = st.session_state.pipeline
    st.subheader("Model Training, Evaluation & Ranking")

    with st.expander("Hyperparameter search space (defaults — edit if desired)"):
        k_min, k_max = st.slider("K-Means / Hierarchical: cluster count range", 2, 15, (2, 8))
        linkages = st.multiselect("Hierarchical linkage methods", ["ward", "complete", "average", "single"],
                                   default=["ward", "complete", "average", "single"])
        min_samples_opts = st.multiselect("DBSCAN min_samples options", [3, 5, 10, 15], default=[3, 5, 10])
        gmm_min, gmm_max = st.slider("GMM component count range", 2, 15, (2, 8))

    if st.button("Run Training", type="primary"):
        grid = HyperparameterGrid(
            kmeans_k=list(range(k_min, k_max + 1)),
            hierarchical_k=list(range(k_min, k_max + 1)),
            hierarchical_linkages=linkages or ["ward"],
            dbscan_min_samples=min_samples_opts or [5],
            gmm_components=list(range(gmm_min, gmm_max + 1)),
        )
        with st.spinner("Training K-Means, Hierarchical, DBSCAN, and GMM across the grid..."):
            leaderboard = pipeline.train_and_rank(grid=grid)
        st.session_state.leaderboard = leaderboard
        st.session_state.stage = "select_model"
        st.rerun()

# ---------------------------------------------------------------------- #
# Stage 8: Human-in-the-loop model selection
# ---------------------------------------------------------------------- #
elif st.session_state.stage == "select_model":
    pipeline = st.session_state.pipeline
    st.subheader("Leaderboard & Human Model Selection")

    leaderboard = st.session_state.leaderboard
    st.write(f"{len(leaderboard)} clustering runs completed.")
    st.dataframe(leaderboard.drop(columns=["labels", "model"], errors="ignore"), use_container_width=True, height=300)

    top = pipeline.recommend(n=3)
    st.markdown("### 🏆 Top 3 Recommendations")
    options = {}
    for _, row in top.iterrows():
        with st.container(border=True):
            st.markdown(f"**#{int(row['rank'])} — {row['algorithm']} `{row['params']}`**")
            st.write(row["rationale"])
            run = pipeline.get_run(int(row["run_id"]))
            sizes = pd.Series(run["labels"]).value_counts().sort_index()
            st.write("Cluster sizes:", dict(sizes))
            fig = plot_cluster_scatter(pipeline.model_df.values, run["labels"], title=f"{row['algorithm']} preview")
            st.pyplot(fig)
        options[f"#{int(row['rank'])} — {row['algorithm']} {row['params']}"] = int(row["run_id"])

    st.markdown("### Also browse the full leaderboard")
    all_options = {
        f"{r['algorithm']} {r['params']} (rank {r.get('rank', 'n/a')})": r["run_id"]
        for r in leaderboard.to_dict("records")
    }
    choice_label = st.selectbox("Select your final model", options=list(options.keys()) + list(all_options.keys()))
    chosen_run_id = options.get(choice_label, all_options.get(choice_label))

    if st.button("Confirm Final Model Selection", type="primary"):
        pipeline.select_model(chosen_run_id)
        st.session_state.stage = "interpret"
        st.rerun()

# ---------------------------------------------------------------------- #
# Stage 9: Cluster interpretation
# ---------------------------------------------------------------------- #
elif st.session_state.stage == "interpret":
    pipeline = st.session_state.pipeline
    st.subheader("Cluster Interpretation")

    with st.spinner("Profiling clusters and computing feature importance..."):
        interpretation = pipeline.interpret()
    st.session_state.interpretation = interpretation

    st.markdown("### Cluster Sizes")
    st.dataframe(interpretation["sizes"], use_container_width=True)
    st.pyplot(plot_cluster_sizes(interpretation["sizes"]))

    st.markdown("### Cluster Profiles")
    st.dataframe(interpretation["profiles"], use_container_width=True)

    if not interpretation["feature_importance"].empty:
        st.markdown("### Feature Importance per Cluster")
        st.dataframe(interpretation["feature_importance"], use_container_width=True)

    st.markdown("### Business Summaries & Suggested Names")
    for cluster_id, summary in interpretation["summaries"].items():
        name = interpretation["suggested_names"][cluster_id]
        st.markdown(f"**Cluster {cluster_id} — \"{name}\"**")
        st.write(summary)

    if st.button("Continue to Final Output", type="primary"):
        st.session_state.stage = "export"
        st.rerun()

# ---------------------------------------------------------------------- #
# Stage 10: Final output & downloadable bundle
# ---------------------------------------------------------------------- #
elif st.session_state.stage == "export":
    pipeline = st.session_state.pipeline
    st.subheader("Final Output")

    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = pipeline.export(tmpdir)
        zip_bytes = Path(zip_path).read_bytes()
        st.success("Results bundle generated.")
        st.download_button(
            "Download Results Bundle (.zip)",
            data=zip_bytes,
            file_name="clustering_results.zip",
            mime="application/zip",
        )

    st.markdown("### Chosen Model")
    st.json(
        {
            "algorithm": pipeline.chosen_run["algorithm"],
            "params": pipeline.chosen_run["params"],
            "silhouette_score": pipeline.chosen_run["silhouette_score"],
            "davies_bouldin_score": pipeline.chosen_run["davies_bouldin_score"],
            "calinski_harabasz_score": pipeline.chosen_run["calinski_harabasz_score"],
        }
    )

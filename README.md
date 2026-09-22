# Unsupervised ML Automation Pipeline (with Human-in-the-Loop)

Automates preprocessing, clustering, evaluation, and ranking for unsupervised
learning — while keeping **final model selection under human control**.

This repo has two ways to use the pipeline:
- **This package** (`ml_automation/` + `cli.py` / `app.py`): a Streamlit app and headless
  CLI — good for local/single-user use.
- **`backend/` + `frontend/`**: a full FastAPI + React/TypeScript/MUI web platform built on
  top of this same core package. Uploading a dataset launches an **agentic pipeline**
  (profiling → cleaning → transformation → training → tuning → evaluation) that pauses at
  human-in-the-loop checkpoints for review, and finishes with a business report (PDF export)
  and a prediction playground — see [AGENTIC_WORKFLOW.md](AGENTIC_WORKFLOW.md). Also adds
  Spectral/Birch/OPTICS clustering and a plugin framework used internally by that pipeline.
  See [ARCHITECTURE.md](ARCHITECTURE.md) for the design and `backend/README.md` /
  `frontend/README.md`-equivalent run instructions below.

  ```
  # backend (from backend/)
  pip install -r requirements.txt
  uvicorn app.main:app --reload

  # frontend (from frontend/)
  npm install
  npm run dev
  ```

## Install

```
pip install -r requirements.txt
```

## Run the interactive UI

```
streamlit run app.py
```

Walks through: upload → preprocessing review → optional PCA → automatic
training/evaluation/ranking → human picks the final model from the top-3
recommendations (or the full leaderboard) → cluster interpretation →
downloadable results bundle.

## Run headless from the CLI

```
python cli.py --input sample_customers.csv --id-columns customer_id --output out/
```

Add `--interactive` to be prompted at each human-in-the-loop checkpoint
(preprocessing plan, PCA, final model choice) instead of taking automatic
defaults.

Generate a synthetic demo dataset:

```
python sample_data.py
```

## Library usage

```python
from ml_automation import AutoClusteringPipeline

pipeline = AutoClusteringPipeline(df, id_columns=["customer_id"])

plan = pipeline.build_preprocessing_plan()   # inspect/edit `plan` here (HITL)
pipeline.preprocess(plan)

pipeline.review_pca(apply=False)             # human decides whether to apply PCA

leaderboard = pipeline.train_and_rank()      # K-Means, Hierarchical, DBSCAN, GMM
                                              # across hyperparameter grids, scored on
                                              # Silhouette / Davies-Bouldin / Calinski-Harabasz

top3 = pipeline.recommend(n=3)               # top 3 + plain-language rationale
pipeline.select_model(run_id=int(top3.iloc[0]["run_id"]))  # human's final call

interpretation = pipeline.interpret()        # profiles, feature importance,
                                              # business summaries, suggested names

zip_path = pipeline.export("output/")        # report + visualizations + CSVs, zipped
```

## Architecture

| Module | Responsibility |
| --- | --- |
| `ml_automation/preprocessing.py` | Column type detection, missing-value imputation, dedup, scaling |
| `ml_automation/dimensionality_reduction.py` | Optional PCA with explained-variance reporting |
| `ml_automation/clustering.py` | K-Means, Hierarchical, DBSCAN, GMM hyperparameter search |
| `ml_automation/evaluation.py` | Silhouette, Davies-Bouldin, Calinski-Harabasz scoring |
| `ml_automation/ranking.py` | Composite-score leaderboard + ranking rationale |
| `ml_automation/interpretation.py` | Cluster profiles, feature importance, business summaries, naming |
| `ml_automation/reporting.py` | Plots and the downloadable results bundle |
| `ml_automation/pipeline.py` | Orchestrates the above; pauses at HITL checkpoints |
| `app.py` | Streamlit UI exposing every HITL checkpoint |
| `cli.py` | Headless (or `--interactive`) command-line runner |

Everything through ranking runs automatically. The pipeline never picks the
final model for you — `select_model()` requires an explicit `run_id` from a
human, and cluster interpretation/export only run after that call.

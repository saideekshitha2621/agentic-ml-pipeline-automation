from app.services import ranking_service


def _evaluated_runs():
    return [
        {
            "run_id": 0,
            "algorithm": "kmeans",
            "params": {"n_clusters": 3},
            "n_clusters": 3,
            "n_noise": 0,
            "n_samples_evaluated": 100,
            "silhouette_score": 0.6,
            "davies_bouldin_score": 0.4,
            "calinski_harabasz_score": 200.0,
        },
        {
            "run_id": 1,
            "algorithm": "dbscan",
            "params": {"eps": 0.5, "min_samples": 5},
            "n_clusters": 4,
            "n_noise": 10,
            "n_samples_evaluated": 90,
            "silhouette_score": 0.3,
            "davies_bouldin_score": 1.1,
            "calinski_harabasz_score": 80.0,
        },
        {
            "run_id": 2,
            "algorithm": "gmm",
            "params": {"n_components": 2},
            "n_clusters": 1,
            "n_noise": 0,
            "n_samples_evaluated": 100,
            "silhouette_score": None,
            "davies_bouldin_score": None,
            "calinski_harabasz_score": None,
        },
    ]


def test_leaderboard_ranks_better_metrics_first():
    leaderboard = ranking_service.rank(_evaluated_runs())
    ranked = leaderboard.dropna(subset=["rank"]).sort_values("rank")
    assert ranked.iloc[0]["algorithm"] == "kmeans"


def test_leaderboard_keeps_invalid_runs_unranked():
    leaderboard = ranking_service.rank(_evaluated_runs())
    invalid_row = leaderboard[leaderboard["run_id"] == 2].iloc[0]
    assert invalid_row["rank"] is None or invalid_row["rank"] != invalid_row["rank"]  # NaN check


def test_leaderboard_penalizes_high_noise_runs():
    """A run that only scores well on the small minority of points it didn't call noise
    should not outrank a run that actually covers the whole dataset — regression test
    for the bug where a 75%-noise DBSCAN run ranked #1 purely because its metrics were
    computed only on the 25% of points it kept."""
    runs = [
        {
            "run_id": 0,
            "algorithm": "kmeans",
            "params": {"n_clusters": 3},
            "n_clusters": 3,
            "n_noise": 0,
            "n_samples_evaluated": 600,
            "silhouette_score": 0.5,
            "davies_bouldin_score": 0.6,
            "calinski_harabasz_score": 150.0,
        },
        {
            "run_id": 1,
            "algorithm": "dbscan",
            "params": {"eps": 0.9, "min_samples": 10},
            "n_clusters": 9,
            "n_noise": 451,
            "n_samples_evaluated": 149,  # only 25% of the 600 points were actually clustered
            "silhouette_score": 0.9,
            "davies_bouldin_score": 0.2,
            "calinski_harabasz_score": 400.0,
        },
    ]
    leaderboard = ranking_service.rank(runs)
    dbscan_row = leaderboard[leaderboard["run_id"] == 1].iloc[0]
    assert dbscan_row["noise_pct"] > 50
    # excluded from ranking outright (default noise_penalty_cap=0.5) despite the flashier raw metrics
    assert dbscan_row["rank"] is None or dbscan_row["rank"] != dbscan_row["rank"]
    kmeans_row = leaderboard[leaderboard["run_id"] == 0].iloc[0]
    assert kmeans_row["rank"] == 1

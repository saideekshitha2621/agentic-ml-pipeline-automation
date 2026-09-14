"""Headless CLI for the unsupervised ML automation pipeline.

Automatic mode:
    python cli.py --input sample_customers.csv --id-columns customer_id --output out/

Interactive mode (prompts at each HITL checkpoint):
    python cli.py --input sample_customers.csv --id-columns customer_id --output out/ --interactive
"""
from __future__ import annotations

import argparse
import sys

import pandas as pd

from ml_automation import AutoClusteringPipeline


def ask_yes_no(prompt: str, default: bool = True) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    resp = input(f"{prompt} {suffix}: ").strip().lower()
    if not resp:
        return default
    return resp.startswith("y")


def main():
    parser = argparse.ArgumentParser(description="Unsupervised ML Automation Pipeline with HITL support")
    parser.add_argument("--input", required=True, help="Path to input CSV")
    parser.add_argument("--id-columns", nargs="*", default=[], help="Columns to exclude from modeling (IDs)")
    parser.add_argument("--output", default="output", help="Output directory for the results bundle")
    parser.add_argument("--interactive", action="store_true", help="Pause for human review at each checkpoint")
    parser.add_argument("--pca", choices=["auto", "yes", "no"], default="auto")
    parser.add_argument("--top-n", type=int, default=3, help="How many models to recommend")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    pipeline = AutoClusteringPipeline(df, id_columns=args.id_columns)

    # --- Step 1-2: preprocessing (+ optional human review) ---
    plan = pipeline.build_preprocessing_plan()
    print("\n=== Preprocessing Plan (detected) ===")
    print(f"Numerical columns:   {plan.numerical_columns}")
    print(f"Categorical columns: {plan.categorical_columns}")
    print(f"Dropped/ID columns:  {plan.dropped_columns}")
    print("\nMissing value summary:")
    print(pipeline.missing_value_summary().to_string(index=False))

    if args.interactive and not ask_yes_no("\nProceed with this preprocessing plan?"):
        print("Edit the plan by re-running with different --id-columns, or edit cli.py to customize it.")
        sys.exit(0)

    report = pipeline.preprocess(plan)
    print(f"\nPreprocessing done. Final shape: {report['final_shape']}. "
          f"Removed {report['duplicates_removed']} duplicate rows.")

    # --- Step 3: optional PCA ---
    apply_pca = args.pca == "yes"
    if args.pca == "auto" and args.interactive:
        curve = pipeline.explained_variance_preview()
        print("\n=== PCA Explained Variance Preview ===")
        print(curve.to_string(index=False))
        apply_pca = ask_yes_no("Apply PCA before clustering?", default=False)

    pca_report = pipeline.review_pca(apply=apply_pca, n_components=0.95 if apply_pca else None)
    if pca_report:
        print(f"\nPCA applied: {pca_report['n_components']} components retaining "
              f"{pca_report['total_variance_retained'] * 100:.1f}% variance.")

    # --- Steps 4-7: automatic training, evaluation, ranking ---
    print("\nRunning K-Means, Hierarchical, DBSCAN, and GMM across hyperparameter grids...")
    leaderboard = pipeline.train_and_rank()
    print(f"Completed {len(leaderboard)} clustering runs.\n")
    print("=== Leaderboard (top 10) ===")
    print(leaderboard.head(10).to_string(index=False))

    # --- Step 8: HITL selection ---
    top = pipeline.recommend(n=args.top_n)
    print(f"\n=== Top {args.top_n} Recommendations ===")
    for _, row in top.iterrows():
        print(f"\n[run_id={row['run_id']}] {row['rationale']}")

    if args.interactive:
        chosen_id = int(input(f"\nEnter run_id of your final chosen model (from the recommendations above): "))
    else:
        chosen_id = int(top.iloc[0]["run_id"])
        print(f"\n--interactive not set: defaulting to top-ranked run_id={chosen_id}. "
              f"Re-run with --interactive to choose manually.")

    pipeline.select_model(chosen_id)

    # --- Step 9: interpretation ---
    print("\nInterpreting clusters (profiles, feature importance, business summaries, names)...")
    interpretation = pipeline.interpret()
    print("\n=== Cluster Sizes ===")
    print(interpretation["sizes"].to_string(index=False))
    print("\n=== Cluster Summaries ===")
    for cluster_id, summary in interpretation["summaries"].items():
        name = interpretation["suggested_names"][cluster_id]
        print(f"\nCluster {cluster_id} — \"{name}\"\n  {summary}")

    # --- Step 10: export ---
    zip_path = pipeline.export(args.output)
    print(f"\nResults bundle written to: {zip_path}")


if __name__ == "__main__":
    main()

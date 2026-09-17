"""One-off migration: cluster_runs -> model_runs (ModelRun architecture, phase 0-1).

This project has no Alembic/migration framework — schema changes so far have been
applied by hand against backend/app/storage/app.db (see: `jobs.problem_type` and
`jobs.target_column` already existed in the live DB before models.py declared either,
evidence of a prior ad hoc column add). This script follows the same pattern for the
ClusterRun -> ModelRun rename:

  1. jobs.problem_type: add only if missing (a prior migration may have already added it).
  2. cluster_runs -> model_runs: rename the table, add problem_type/metrics_json/
     artifacts_json, backfill metrics_json/artifacts_json from the existing typed columns
     for any pre-existing rows (safe no-op if the table is empty).
  3. approvals / cluster_interpretations: their cluster_run_id FK still references the old
     table name in their CREATE TABLE text. SQLite does not enforce foreign keys unless
     `PRAGMA foreign_keys=ON` is set (this app never sets it — see db/database.py), so this
     is not a functional break, only a stale schema annotation. Both tables are recreated
     here with the FK pointed at model_runs so `sqlite_master` stays accurate, which is
     safe precisely because both are empty in every environment this has been tested
     against; the script aborts instead of doing this step if it ever finds rows, so it is
     never run silently against data it wasn't verified for.

Idempotent: safe to run multiple times, and safe to run against a fresh DB that
`init_db()` created directly from the current models.py (every step checks first).
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[1] / "app" / "storage" / "app.db"


def _columns(cur: sqlite3.Cursor, table: str) -> set[str]:
    cur.execute(f"PRAGMA table_info({table})")
    return {row[1] for row in cur.fetchall()}


def _table_exists(cur: sqlite3.Cursor, table: str) -> bool:
    cur.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,))
    return cur.fetchone() is not None


def migrate(db_path: Path = DB_PATH) -> None:
    con = sqlite3.connect(str(db_path))
    cur = con.cursor()

    if not _table_exists(cur, "jobs"):
        print("No `jobs` table yet — nothing to migrate (fresh DB, init_db() will create the current schema).")
        con.close()
        return

    # 1. jobs.problem_type
    if "problem_type" not in _columns(cur, "jobs"):
        cur.execute("ALTER TABLE jobs ADD COLUMN problem_type VARCHAR DEFAULT 'clustering'")
        print("Added jobs.problem_type")
    else:
        print("jobs.problem_type already present — skipped")

    # 2. cluster_runs -> model_runs
    if _table_exists(cur, "cluster_runs") and not _table_exists(cur, "model_runs"):
        cur.execute("ALTER TABLE cluster_runs RENAME TO model_runs")
        print("Renamed cluster_runs -> model_runs")
    elif _table_exists(cur, "cluster_runs") and _table_exists(cur, "model_runs"):
        # Both exist — e.g. a running --reload dev server already called init_db()
        # against the updated models.py and create_all() made a fresh, empty model_runs
        # table before this script ran. The old cluster_runs is now dead (unmapped by any
        # ORM class); drop it, but only if it's empty so nothing is ever silently lost.
        cur.execute("SELECT COUNT(*) FROM cluster_runs")
        n_old = cur.fetchone()[0]
        if n_old == 0:
            cur.execute("DROP TABLE cluster_runs")
            print("Both cluster_runs and model_runs existed; dropped the empty orphaned cluster_runs")
        else:
            print(
                f"Both cluster_runs ({n_old} row(s)) and model_runs exist — not auto-dropping "
                "cluster_runs since it still has data. Migrate those rows manually, then drop it."
            )
    else:
        print("model_runs already present — skipped rename")

    if _table_exists(cur, "model_runs"):
        cols = _columns(cur, "model_runs")
        if "problem_type" not in cols:
            cur.execute("ALTER TABLE model_runs ADD COLUMN problem_type VARCHAR DEFAULT 'clustering'")
            print("Added model_runs.problem_type")
        if "metrics_json" not in cols:
            cur.execute("ALTER TABLE model_runs ADD COLUMN metrics_json JSON DEFAULT '{}'")
            print("Added model_runs.metrics_json")
        if "artifacts_json" not in cols:
            cur.execute("ALTER TABLE model_runs ADD COLUMN artifacts_json JSON DEFAULT '{}'")
            print("Added model_runs.artifacts_json")

        cur.execute("SELECT COUNT(*) FROM model_runs")
        n_runs = cur.fetchone()[0]
        if n_runs:
            cur.execute(
                """
                UPDATE model_runs
                SET metrics_json = json_object(
                        'silhouette', silhouette_score,
                        'davies_bouldin', davies_bouldin_score,
                        'calinski_harabasz', calinski_harabasz_score
                    ),
                    artifacts_json = json_object('labels_path', labels_path)
                WHERE metrics_json = '{}' OR metrics_json IS NULL
                """
            )
            print(f"Backfilled metrics_json/artifacts_json for {n_runs} existing model_run row(s)")
        else:
            print("model_runs is empty — no backfill needed")

    # 3. Repoint the FK annotation on child tables, only if they're empty.
    for child, fk_col in (("approvals", "cluster_run_id"), ("cluster_interpretations", "cluster_run_id")):
        if not _table_exists(cur, child):
            continue
        cur.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{child}'")
        create_sql = cur.fetchone()[0]
        if "REFERENCES model_runs" in create_sql:
            print(f"{child}.{fk_col} FK already points at model_runs — skipped")
            continue
        cur.execute(f"SELECT COUNT(*) FROM {child}")
        n_rows = cur.fetchone()[0]
        if n_rows:
            print(
                f"ABORTING FK repoint for {child}: table has {n_rows} row(s) and this script "
                "only recreates empty tables. Repoint manually or extend this script with a "
                "verified copy-and-swap before rerunning."
            )
            continue
        new_sql = create_sql.replace("REFERENCES cluster_runs", "REFERENCES model_runs")
        cur.execute(f"DROP TABLE {child}")
        cur.execute(new_sql)
        print(f"Recreated empty {child} with its FK repointed at model_runs")

    con.commit()
    con.close()
    print("Migration complete.")


if __name__ == "__main__":
    migrate(Path(sys.argv[1]) if len(sys.argv) > 1 else DB_PATH)

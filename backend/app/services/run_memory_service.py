"""Cross-run memory (Phase 5): what happened the last time this dataset/target was modelled?

A completed earlier run on the same problem (same problem type and target, and the same file
name or column count) tells us which algorithm actually won. That history is used two ways, both
advisory and both visible in the audit trail:
  * the algorithm shortlist notes prior winners and makes sure they are included;
  * the critic flags when the new champion disagrees with what usually won before.
Memory never overrides a decision or skips a human gate.
"""
from __future__ import annotations

from collections import Counter

from sqlalchemy.orm import Session

from app.db.models import Dataset as DatasetORM
from app.db.models import ModelRun as ModelRunORM
from app.db.models import PipelineRun as PipelineRunORM

MAX_HISTORY = 5
_KEY_METRIC = {"classification": "f1_macro", "regression": "r2", "clustering": "silhouette_score"}


def similar_runs(db: Session, run: PipelineRunORM) -> list[dict]:
    if not run.declared_target or not run.problem_type:
        return []
    dataset = db.get(DatasetORM, run.dataset_id)
    candidates = (
        db.query(PipelineRunORM)
        .filter(
            PipelineRunORM.id != run.id,
            PipelineRunORM.status == "completed",
            PipelineRunORM.problem_type == run.problem_type,
            PipelineRunORM.declared_target == run.declared_target,
            PipelineRunORM.champion_run_id.isnot(None),
        )
        .order_by(PipelineRunORM.created_at.desc())
        .limit(MAX_HISTORY * 4)
        .all()
    )
    history = []
    for other in candidates:
        other_ds = db.get(DatasetORM, other.dataset_id)
        if other_ds is None or not (other_ds.filename == dataset.filename or other_ds.n_columns == dataset.n_columns):
            continue
        champion = db.get(ModelRunORM, other.champion_run_id)
        if champion is None:
            continue
        metric = _KEY_METRIC.get(run.problem_type)
        value = getattr(champion, metric, None) if run.problem_type == "clustering" else (champion.metrics_json or {}).get(metric)
        history.append({
            "run_id": other.id, "dataset": other_ds.filename, "champion_algorithm": champion.algorithm,
            "key_metric": metric, "value": value,
        })
        if len(history) >= MAX_HISTORY:
            break
    return history


def prior_winners(history: list[dict]) -> Counter:
    return Counter(h["champion_algorithm"] for h in history)


def apply_to_shortlist(shortlist: dict, history: list[dict], valid_names: list[str]) -> dict:
    """Annotate prior winners and guarantee they are in the shortlist (never removes anything)."""
    if not history:
        return shortlist
    winners = prior_winners(history)
    selected = list(shortlist["selected_algorithms"])
    entries = []
    for entry in shortlist["shortlist"]:
        wins = winners.get(entry["algorithm"], 0)
        if wins:
            entry = {
                **entry, "recommended": True,
                "rationale": f"{entry['rationale']} Won {wins} of {len(history)} earlier run(s) on this dataset/target.",
            }
            if entry["algorithm"] not in selected and entry["algorithm"] in valid_names:
                selected.append(entry["algorithm"])
        entries.append(entry)
    top, top_wins = winners.most_common(1)[0]
    return {
        **shortlist, "shortlist": entries, "selected_algorithms": selected,
        "prior_runs": history[:3],
        "memory_note": f"{top} won {top_wins} of {len(history)} earlier run(s) on this dataset/target.",
    }

"""Read-only dry run of the agent stages against the datasets already uploaded to the app.

    python scripts/dry_run_uploaded.py                 # every uploaded dataset whose file still exists
    python scripts/dry_run_uploaded.py "Salary"        # only datasets whose filename contains this text

Nothing is written: the DB is opened read-only, no pipeline run is created, no model is trained
and no LLM is called. It prints what each agent WOULD propose (problem type, validation,
cleaning, transformation, split, algorithm shortlist) so you can sanity-check them on real data.
"""
from __future__ import annotations

import json
import sqlite3
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from app.agents import (  # noqa: E402
    algorithm_selection_llm, cleaning_plan_agent, data_profiling_agent, data_validation_agent,
    problem_detection_agent, split_agent, transformation_agent,
)
from app.core.config import STORAGE_DIR  # noqa: E402
from app.services import llm_service  # noqa: E402


def main() -> int:
    needle = (sys.argv[1] if len(sys.argv) > 1 else "").lower()
    llm_service._active_config = lambda: None  # guarantee: no LLM calls, deterministic agents only

    con = sqlite3.connect(f"file:{(STORAGE_DIR / 'app.db').as_posix()}?mode=ro", uri=True)
    rows = con.execute(
        "select d.id, d.filename, d.storage_path, d.profile_json, "
        "(select declared_target from pipeline_runs p where p.dataset_id=d.id order by p.created_at desc limit 1) "
        "from datasets d order by d.uploaded_at"
    ).fetchall()

    for _id, filename, path, profile_json, target in rows:
        if needle and needle not in filename.lower():
            continue
        print("=" * 78)
        if not Path(path).exists():
            print(f"{filename}: SKIPPED (file no longer on disk)")
            continue
        df = pd.read_csv(path)
        print(f"{filename}: {len(df)} rows x {len(df.columns)} cols | last declared target: {target}")
        try:
            profile = data_profiling_agent.analyze(df, json.loads(profile_json) if profile_json else {})
            det = problem_detection_agent.detect(df, profile, declared_target=target)
            heur = problem_detection_agent.detect(df, profile)
            print(f"  problem   : {det['problem_type']} on '{det['target_column']}' (conf {det['confidence']}) "
                  f"| with NO declared target the heuristic would pick '{heur['target_column']}' ({heur['problem_type']})")
            ptype, tgt = det["problem_type"], det["target_column"]

            val = data_validation_agent.validate(df, profile, target_column=tgt, problem_type=ptype)
            flagged = [f"{c['name']}={c['status']}" for c in val["checks"] if c["status"] != "ok"]
            print(f"  validation: {val['overall_status']} | flagged: {flagged or 'none'}")

            plan = cleaning_plan_agent.propose(df, val, target_column=tgt)
            conf, factors = cleaning_plan_agent.estimate_confidence(plan, len(df.columns))
            print(f"  cleaning  : {cleaning_plan_agent.summarize(plan)}  [conf {conf}]")
            for f in factors:
                print(f"              {f}")

            fields = cleaning_plan_agent.to_preprocessing_plan_fields(plan)
            tr = transformation_agent.propose(df, fields)
            print(f"  transform : {tr['scaling_method']}, {tr['encoding_method']}  [conf {tr['confidence']}] {tr['confidence_factors']}")

            if ptype in ("classification", "regression"):
                sp = split_agent.recommend(df, ptype, target_column=tgt)
                print(f"  split     : test_size={sp['test_size']} stratify={sp['stratify']}  [conf {sp['confidence']}] {sp['confidence_factors']}")
            else:
                print("  split     : n/a (clustering)")

            al = algorithm_selection_llm.recommend(profile, val, ptype, df, tgt)
            print(f"  algorithms: {al['selected_algorithms']}  [conf {al['confidence']}]")
        except Exception as exc:  # noqa: BLE001
            print(f"  ERROR while dry-running this dataset: {type(exc).__name__}: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

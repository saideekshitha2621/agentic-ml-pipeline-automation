from __future__ import annotations

import pandas as pd

from ml_automation.ranking import build_leaderboard, explain_ranking


def rank(evaluated_runs: list[dict], problem_type: str = "clustering") -> pd.DataFrame:
    return build_leaderboard(evaluated_runs, problem_type=problem_type)


def rationale_for(row: pd.Series, problem_type: str = "clustering") -> str:
    return explain_ranking(row, problem_type=problem_type)

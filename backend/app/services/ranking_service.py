from __future__ import annotations

import pandas as pd

from ml_automation.ranking import build_leaderboard, explain_ranking


def rank(evaluated_runs: list[dict]) -> pd.DataFrame:
    return build_leaderboard(evaluated_runs)


def rationale_for(row: pd.Series) -> str:
    return explain_ranking(row)

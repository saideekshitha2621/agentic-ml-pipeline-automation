"""Agent quality metrics (Phase 5): are the agents actually good, and how much do humans trust them?

A pure function over recorded AgentDecision rows, so it can be tested without a database and
reused by any endpoint. It answers the questions that tell you whether the agentic system is
earning its autonomy:

  * per agent — how often a human approved / edited / rejected its proposals (a high rejection
    or edit rate means the agent is proposing the wrong things), how often the policy
    auto-approved it, its average confidence, and how often the LLM (vs the deterministic
    fallback) produced the answer and why it fell back;
  * per run — how many revisions a human forced and how often the self-correction loop had to
    retry, which are direct measures of first-attempt quality.
"""
from __future__ import annotations

from collections import Counter, defaultdict


def summarize(decisions: list[dict]) -> dict:
    """`decisions`: dicts with agent_name, status, approved_by, confidence, decision_json, pipeline_run_id."""
    by_agent: dict[str, list[dict]] = defaultdict(list)
    for d in decisions:
        by_agent[d["agent_name"]].append(d)

    agents = {}
    for name, rows in sorted(by_agent.items()):
        auto = sum(1 for r in rows if r["status"] == "approved" and r.get("approved_by") == "system")
        human_ok = sum(1 for r in rows if r["status"] == "approved" and r.get("approved_by") not in (None, "system"))
        edited = sum(1 for r in rows if r["status"] == "edited")
        rejected = sum(1 for r in rows if r["status"] == "rejected")
        human_reviewed = human_ok + edited + rejected
        confidences = [r["confidence"] for r in rows if isinstance(r.get("confidence"), (int, float))]
        sources = [(r.get("decision_json") or {}).get("source") for r in rows]
        with_source = [s for s in sources if s]
        fallbacks = Counter(
            str((r.get("decision_json") or {}).get("fallback_reason", "")).split(":")[0]
            for r in rows if (r.get("decision_json") or {}).get("fallback_reason")
        )
        agents[name] = {
            "decisions": len(rows),
            "auto_approved": auto,
            "human_approved": human_ok,
            "edited": edited,
            "rejected": rejected,
            "human_rejection_rate": round(rejected / human_reviewed, 3) if human_reviewed else None,
            "human_edit_rate": round(edited / human_reviewed, 3) if human_reviewed else None,
            "avg_confidence": round(sum(confidences) / len(confidences), 3) if confidences else None,
            "llm_share": round(sum(1 for s in with_source if s == "llm") / len(with_source), 3) if with_source else None,
            "fallback_reasons": dict(fallbacks),
        }

    run_ids = {d["pipeline_run_id"] for d in decisions}
    rejected_per_run = Counter(d["pipeline_run_id"] for d in decisions if d["status"] == "rejected")
    retries_per_run = Counter(
        d["pipeline_run_id"] for d in decisions
        if d["agent_name"] == "quality_check" and (d.get("decision_json") or {}).get("verdict") == "retry"
    )
    n_runs = len(run_ids) or 1
    approved_or_edited = [d for d in decisions if d["status"] in ("approved", "edited")]
    return {
        "n_runs": len(run_ids),
        "n_decisions": len(decisions),
        "auto_approval_rate": round(
            sum(1 for d in approved_or_edited if d.get("approved_by") == "system") / len(approved_or_edited), 3
        ) if approved_or_edited else None,
        "avg_human_revisions_per_run": round(sum(rejected_per_run.values()) / n_runs, 3),
        "avg_self_corrections_per_run": round(sum(retries_per_run.values()) / n_runs, 3),
        "runs_needing_self_correction": len(retries_per_run),
        "agents": agents,
    }

"""Reporting Agent.

Synthesizes the full agent_decisions trail for a pipeline run into a business-readable
report: no new judgment happens here, just composition of what earlier agents already
decided and what a human approved/overrode. Reuses existing services for the underlying
data (profiling, job leaderboard); does not touch `export_service.py`'s existing
clustering-only PDF/Excel templates — this is an additive report format alongside them.
"""
from __future__ import annotations

from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer

from app.db.models import AgentDecision as AgentDecisionORM
from app.db.models import Dataset as DatasetORM
from app.db.models import Job as JobORM
from app.db.models import PipelineRun as PipelineRunORM


def _decision_by_agent(decisions: list[AgentDecisionORM], agent_name: str) -> AgentDecisionORM | None:
    matches = [d for d in decisions if d.agent_name == agent_name]
    return matches[-1] if matches else None


def _human_decision_summary(decision: AgentDecisionORM) -> dict:
    return {
        "stage": decision.stage,
        "agent": decision.agent_name,
        "status": decision.status,
        "approved_by": decision.approved_by,
        "override_reason": decision.override_reason,
        "human_edits": decision.human_edits_json,
    }


def build_report(
    pipeline_run: PipelineRunORM,
    decisions: list[AgentDecisionORM],
    dataset: DatasetORM,
    job: JobORM | None = None,
) -> dict:
    profiling = _decision_by_agent(decisions, "data_profiling")
    problem = _decision_by_agent(decisions, "problem_detection")
    model_selection = _decision_by_agent(decisions, "model_selection")
    recommendation = _decision_by_agent(decisions, "recommendation")

    models_evaluated = []
    if job:
        runs = (
            job.runs
            if hasattr(job, "runs")
            else []
        )
        for run in sorted(runs, key=lambda r: (r.rank is None, r.rank)):
            models_evaluated.append(
                {
                    "algorithm": run.algorithm,
                    "rank": run.rank,
                    "composite_score": run.composite_score,
                    "metrics": run.metrics_json,
                }
            )

    human_decisions = [_human_decision_summary(d) for d in decisions if d.status in ("approved", "edited", "rejected")]

    return {
        "dataset_summary": {
            "filename": dataset.filename,
            "n_rows": dataset.n_rows,
            "n_columns": dataset.n_columns,
            "data_quality_score": dataset.data_quality_score,
        },
        "analysis_summary": {
            "problem_type": pipeline_run.problem_type,
            "reasoning": problem.reasoning_text if problem else None,
            "confidence": problem.confidence if problem else None,
        },
        "decisions_taken": [
            {"agent": d.agent_name, "stage": d.stage, "reasoning": d.reasoning_text, "status": d.status}
            for d in decisions
        ],
        "models_evaluated": models_evaluated,
        "recommendation_details": recommendation.decision_json if recommendation else None,
        "human_decisions": human_decisions,
        "final_outcome": pipeline_run.status,
    }


def export_pdf(report: dict, out_path: Path) -> Path:
    doc = SimpleDocTemplate(str(out_path), pagesize=letter)
    styles = getSampleStyleSheet()
    story = [Paragraph("Agentic ML Pipeline — Run Report", styles["Title"]), Spacer(1, 16)]

    def section(title: str, body: str):
        story.append(Paragraph(title, styles["Heading2"]))
        story.append(Paragraph(body.replace("\n", "<br/>"), styles["BodyText"]))
        story.append(Spacer(1, 12))

    ds = report["dataset_summary"]
    section(
        "Dataset Summary",
        f"File: {ds['filename']} — {ds['n_rows']} rows, {ds['n_columns']} columns, "
        f"data quality score {ds['data_quality_score']}/100.",
    )

    analysis = report["analysis_summary"]
    section(
        "Analysis Summary",
        f"Detected problem type: <b>{analysis['problem_type']}</b> "
        f"(confidence {analysis['confidence']}).<br/>{analysis['reasoning']}",
    )

    decisions_text = "<br/>".join(
        f"[{d['agent']}] {d['stage']} — {d['status']}: {d['reasoning']}" for d in report["decisions_taken"]
    )
    section("Decisions Taken", decisions_text or "No decisions recorded.")

    if report["models_evaluated"]:
        def _metrics_str(metrics: dict) -> str:
            return ", ".join(f"{k}={v}" for k, v in (metrics or {}).items() if v is not None)

        models_text = "<br/>".join(
            f"#{m['rank']} {m['algorithm']} — {_metrics_str(m['metrics'])}, composite {m['composite_score']}"
            for m in report["models_evaluated"]
        )
        section("Models Evaluated", models_text)

    if report["recommendation_details"]:
        top = report["recommendation_details"]["top_choice"]
        section("Recommendation Details", f"{top['algorithm']} — {top['rationale']}")

    human_text = "<br/>".join(
        f"[{h['stage']}] {h['status']} by {h['approved_by']}"
        + (f" — override reason: {h['override_reason']}" if h["override_reason"] else "")
        for h in report["human_decisions"]
    )
    section("Human Decisions", human_text or "No human review recorded yet.")

    section("Final Outcome", report["final_outcome"])

    doc.build(story)
    return out_path

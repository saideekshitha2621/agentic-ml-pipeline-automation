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


def _business_findings(validation: AgentDecisionORM | None, cleaning: AgentDecisionORM | None) -> str:
    if not validation:
        return "Data quality was not yet assessed."
    checks = validation.decision_json.get("checks", [])
    issues = [c for c in checks if c["status"] != "ok"]
    if not issues:
        base = "No significant data quality issues were found — the dataset was clean and ready to model."
    else:
        issue_names = ", ".join(c["name"].lower() for c in issues)
        base = f"{len(issues)} data quality issue(s) were found ({issue_names}) and addressed before modeling."
    if cleaning:
        n_dropped = len(cleaning.decision_json.get("dropped_columns", []) or [
            r["column"] for r in cleaning.decision_json.get("recommendations", []) if r.get("action") == "drop_column"
        ])
        n_imputed = sum(1 for r in cleaning.decision_json.get("recommendations", []) if r.get("action") in ("mean", "median", "mode"))
        base += f" {n_imputed} column(s) had missing values filled in, and {n_dropped} column(s) were removed as unreliable or uninformative."
    return base


def _actions_taken(decisions: list[AgentDecisionORM]) -> list[str]:
    actions = []
    for d in decisions:
        impact = d.decision_json.get("business_impact")
        if impact:
            actions.append(f"{d.agent_name.replace('_', ' ').title()}: {impact}")
    return actions


def _business_insights(recommendation: AgentDecisionORM | None, validation: AgentDecisionORM | None) -> list[str]:
    insights = []
    if recommendation:
        top = recommendation.decision_json["top_choice"]
        insights.extend(top.get("business_benefits", []))
        insights.extend(f"Watch point: {w}" for w in top.get("weaknesses", []))
    if validation:
        critical = [c for c in validation.decision_json.get("checks", []) if c["status"] == "critical"]
        insights.extend(f"Data risk: {c['detail']}" for c in critical)
    return insights or ["No significant concerns were identified."]


def _recommendations(recommendation: AgentDecisionORM | None, pipeline_run: PipelineRunORM) -> list[str]:
    if not recommendation:
        return ["Complete the remaining review steps to receive a model recommendation."]
    top = recommendation.decision_json["top_choice"]
    items = [
        f"Deploy the {top['algorithm'].replace('_', ' ')} model recommended by this analysis for "
        f"{pipeline_run.declared_target or 'the target outcome'}.",
        "Monitor prediction performance periodically and retrain if real-world outcomes drift from these results.",
    ]
    if recommendation.decision_json.get("confidence") == "low":
        items.append("Confidence in this recommendation is modest — consider a pilot rollout before full deployment.")
    return items


def build_report(
    pipeline_run: PipelineRunORM,
    decisions: list[AgentDecisionORM],
    dataset: DatasetORM,
    job: JobORM | None = None,
) -> dict:
    profiling = _decision_by_agent(decisions, "data_profiling")
    problem = _decision_by_agent(decisions, "problem_detection")
    framing = _decision_by_agent(decisions, "business_framing")
    validation = _decision_by_agent(decisions, "data_validation")
    cleaning = _decision_by_agent(decisions, "cleaning_plan")
    model_selection = _decision_by_agent(decisions, "model_selection")
    evaluation = _decision_by_agent(decisions, "evaluation")
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
        # Business-language sections (requirement: a report non-technical stakeholders can read end to end)
        "executive_summary": {
            "business_problem": framing.decision_json["business_problem_statement"] if framing else None,
            "dataset_overview": f"{dataset.n_rows:,} rows, {dataset.n_columns} columns, "
            f"data quality score {dataset.data_quality_score}/100.",
            "ml_type": framing.decision_json["ml_type_business_label"] if framing else None,
            "target_variable": framing.decision_json["target_variable"] if framing else None,
            "recommended_model": recommendation.decision_json["top_choice"]["algorithm"] if recommendation else None,
            "performance_summary": next(
                iter(((evaluation.decision_json.get("business_metrics") if evaluation else {}) or {}).values()), None
            ),
        },
        "problem_statement": framing.decision_json["business_problem_statement"] if framing else (
            problem.reasoning_text if problem else None
        ),
        "data_quality_findings": _business_findings(validation, cleaning),
        "actions_taken": _actions_taken(decisions),
        "model_recommendation": {
            "algorithm": recommendation.decision_json["top_choice"]["algorithm"],
            "rationale": recommendation.decision_json["top_choice"]["rationale"],
            "business_benefits": recommendation.decision_json["top_choice"].get("business_benefits", []),
        } if recommendation else None,
        "performance_summary": evaluation.decision_json.get("business_metrics") if evaluation else {},
        "business_insights": _business_insights(recommendation, validation),
        "recommendations": _recommendations(recommendation, pipeline_run),
        # Populated live at read time by routers/pipeline.py::get_report() from
        # PredictionLog rows — always [] in the stored snapshot, since predictions are
        # normally made via the Playground after the run (and this report) already exist.
        "predictions": [],
    }


def export_pdf(report: dict, out_path: Path) -> Path:
    doc = SimpleDocTemplate(str(out_path), pagesize=letter)
    styles = getSampleStyleSheet()
    story = [Paragraph("Business Report", styles["Title"]), Spacer(1, 16)]

    def section(title: str, body: str | list[str]):
        story.append(Paragraph(title, styles["Heading2"]))
        text = "<br/>".join(f"• {b}" for b in body) if isinstance(body, list) else body
        story.append(Paragraph((text or "Not available yet.").replace("\n", "<br/>"), styles["BodyText"]))
        story.append(Spacer(1, 12))

    # Business-language sections, in the requested order — this is the document a
    # non-technical stakeholder is meant to read; the technical appendix follows.
    exec_summary = report.get("executive_summary") or {}
    section(
        "Executive Summary",
        f"<b>Business problem:</b> {exec_summary.get('business_problem') or 'Not yet determined.'}<br/>"
        f"<b>Dataset:</b> {exec_summary.get('dataset_overview') or ''}<br/>"
        f"<b>ML type:</b> {exec_summary.get('ml_type') or 'Pending.'}<br/>"
        f"<b>Target variable:</b> {exec_summary.get('target_variable') or 'None (unsupervised).'}<br/>"
        f"<b>Recommended model:</b> {exec_summary.get('recommended_model') or 'Pending.'}<br/>"
        f"<b>Performance:</b> {exec_summary.get('performance_summary') or 'Pending.'}",
    )
    section("Problem Statement", report.get("problem_statement") or "Not available.")
    section("Data Quality Findings", report.get("data_quality_findings") or "Not available.")
    section("Actions Taken", report.get("actions_taken") or [])
    model_rec = report.get("model_recommendation")
    section(
        "Model Recommendation",
        f"<b>{model_rec['algorithm']}</b> — {model_rec['rationale']}" if model_rec else "Not yet available.",
    )
    section("Performance Summary", list((report.get("performance_summary") or {}).values()))
    predictions = report.get("predictions") or []
    section(
        "Predictions",
        [f"Input {p['input']} → predicted {p['prediction']} ({p['confidence']})" for p in predictions]
        if predictions else "No predictions have been made with this model yet.",
    )
    section("Business Insights", report.get("business_insights") or [])
    section("Recommendations", report.get("recommendations") or [])

    story.append(Paragraph("Technical Appendix", styles["Heading1"]))
    story.append(Spacer(1, 8))

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

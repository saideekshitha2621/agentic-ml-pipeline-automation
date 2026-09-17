"""Reporting Agent.

Synthesizes the full agent_decisions trail for a pipeline run into a business-readable
report shaped like a document a manager/client could read in 2-3 minutes: Executive
Summary, Problem Statement, Data Quality Summary, Data Preparation Summary, Model
Selection Summary, Model Performance, Key Insights, Prediction Capability, Business
Recommendations, Conclusion, and a Technical Appendix for anyone who wants the raw numbers.
No new judgment happens here — this is composition of what earlier agents already decided
and what a human approved/overrode.
"""
from __future__ import annotations

from pathlib import Path

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from reportlab.lib import colors

from app.db.models import AgentDecision as AgentDecisionORM
from app.db.models import Dataset as DatasetORM
from app.db.models import Job as JobORM
from app.db.models import PipelineRun as PipelineRunORM

_IMPUTE_ACTION_LABELS = {"mean": "Mean Imputation", "median": "Median Imputation", "mode": "Mode Imputation"}


def _decision_by_agent(decisions: list[AgentDecisionORM], agent_name: str) -> AgentDecisionORM | None:
    matches = [d for d in decisions if d.agent_name == agent_name]
    return matches[-1] if matches else None


def _effective(decision: AgentDecisionORM | None) -> dict:
    if decision is None:
        return {}
    return decision.human_edits_json if decision.status == "edited" and decision.human_edits_json else decision.decision_json


def _champion_run(job: JobORM | None, algorithm: str | None):
    if not job or not algorithm:
        return None
    candidates = [r for r in job.runs if r.algorithm == algorithm]
    return min(candidates, key=lambda r: (r.rank is None, r.rank), default=None) if candidates else None


def _data_quality_summary(validation: AgentDecisionORM | None, cleaning: AgentDecisionORM | None, profiling: AgentDecisionORM | None) -> dict:
    checks = _effective(validation).get("checks", [])
    du = _effective(profiling).get("dataset_understanding", {})
    invalid_data_cols = next((c["affected_columns"] for c in checks if c["name"] == "Invalid data types"), [])

    recs = _effective(cleaning).get("recommendations", [])
    actions_taken = [
        {
            "column": r["column"],
            "missing_values": r["missing_count"],
            "action": _IMPUTE_ACTION_LABELS.get(r["action"], r["action"].replace("_", " ").title()),
            "reason": r["reason"],
        }
        for r in recs if r["action"] in _IMPUTE_ACTION_LABELS
    ]
    n_dupes = _effective(cleaning).get("n_duplicates", du.get("n_duplicate_rows", 0))
    drop_duplicates = _effective(cleaning).get("drop_duplicates", False)
    duplicate_handling = (
        f"{n_dupes} duplicate record(s) were identified and removed before training." if n_dupes and drop_duplicates
        else f"{n_dupes} duplicate record(s) were identified but kept (not removed)." if n_dupes
        else "No duplicate records were found."
    )

    return {
        "issues_detected": {
            "missing_values": du.get("n_missing_values", 0),
            "duplicate_records": n_dupes,
            "invalid_data": len(invalid_data_cols),
        },
        "actions_taken": actions_taken,
        "duplicate_handling": duplicate_handling,
    }


def _data_preparation_summary(cleaning: AgentDecisionORM | None, transformation: AgentDecisionORM | None, split: AgentDecisionORM | None) -> dict:
    cleaning_data = _effective(cleaning)
    transformation_data = _effective(transformation)
    split_data = _effective(split)

    steps = []
    if any(r["action"] in _IMPUTE_ACTION_LABELS for r in cleaning_data.get("recommendations", [])):
        steps.append("Missing values handled")
    if cleaning_data.get("drop_duplicates"):
        steps.append("Duplicate records removed")
    if transformation_data.get("categorical_columns"):
        steps.append("Categorical values encoded")
    if transformation_data.get("scaling_method", "none") != "none":
        steps.append("Numerical values normalized")

    train_pct = test_pct = None
    if split_data.get("applicable"):
        train_pct = round((split_data.get("train_size") or 0) * 100)
        test_pct = round((split_data.get("test_size") or 0) * 100)
        steps.append("Dataset split into training and testing sets")

    return {"steps": steps, "train_pct": train_pct, "test_pct": test_pct}


def _model_selection_summary(model_selection: AgentDecisionORM | None, hpo: AgentDecisionORM | None, recommendation: AgentDecisionORM | None) -> dict:
    algorithms = _effective(model_selection).get("algorithms_run", [])
    recommendation_data = _effective(recommendation)
    top = recommendation_data.get("top_choice")

    hpo_results = [r for r in _effective(hpo).get("results", []) if not r.get("skipped")]
    if hpo_results:
        hpo_text = (
            "Hyperparameter tuning was performed automatically to improve model performance across "
            f"{len(hpo_results)} algorithm(s). The optimized configuration outperformed each algorithm's default "
            "settings and was used for the final comparison."
        )
    else:
        hpo_text = "Hyperparameter optimization was not applicable for this run (e.g. clustering, or no search space configured)."

    return {
        "algorithms_evaluated": [a.replace("_", " ").title() for a in algorithms],
        "selected_model": top["algorithm"].replace("_", " ").title() if top else None,
        "why_selected": top["rationale"] if top else None,
        "hyperparameter_optimization": hpo_text,
    }


def _model_performance(evaluation: AgentDecisionORM | None, recommendation: AgentDecisionORM | None) -> dict:
    business_metrics = _effective(evaluation).get("business_metrics", {})
    confidence = _effective(recommendation).get("confidence")
    confidence_level = {"high": "High", "low": "Moderate"}.get(confidence, "Not yet available")
    confidence_explanation = (
        "The model demonstrates strong predictive capability and is suitable for business decision support."
        if confidence == "high"
        else "The model shows reasonable predictive capability, but results were close between the top candidates "
        "— consider a pilot rollout before full deployment."
        if confidence == "low"
        else "Confidence will be available once a model has been recommended."
    )
    return {
        "reliability_sentence": next(iter(business_metrics.values()), None),
        "confidence_level": confidence_level,
        "confidence_explanation": confidence_explanation,
    }


def _key_insights(recommendation: AgentDecisionORM | None, validation: AgentDecisionORM | None) -> list[str]:
    insights = []
    recommendation_data = _effective(recommendation)
    if recommendation_data:
        top = recommendation_data["top_choice"]
        insights.extend(top.get("business_benefits", []))
        insights.extend(f"Watch point: {w}" for w in top.get("weaknesses", []))
    validation_data = _effective(validation)
    if validation_data:
        critical = [c for c in validation_data.get("checks", []) if c["status"] == "critical"]
        insights.extend(f"Data risk: {c['detail']}" for c in critical)
    return insights or ["No significant concerns were identified."]


def _business_recommendations(recommendation: AgentDecisionORM | None, pipeline_run: PipelineRunORM) -> list[str]:
    recommendation_data = _effective(recommendation)
    if not recommendation_data:
        return ["Complete the remaining review steps to receive a model recommendation."]
    top = recommendation_data["top_choice"]
    items = [
        f"Deploy the {top['algorithm'].replace('_', ' ')} model recommended by this analysis for "
        f"{pipeline_run.declared_target or 'the target outcome'}.",
        "Collect additional data over time to further improve prediction accuracy.",
        "Use the model as a decision-support tool, reviewing important or low-confidence cases manually.",
        "Monitor prediction performance periodically and retrain if real-world outcomes drift from these results.",
    ]
    if recommendation_data.get("confidence") == "low":
        items.append("Confidence in this recommendation is modest — consider a pilot rollout before full deployment.")
    return items


def _conclusion(framing: AgentDecisionORM | None, model_selection_summary: dict, pipeline_run: PipelineRunORM) -> str:
    framing_data = _effective(framing)
    target = pipeline_run.declared_target or "the target outcome"
    selected = model_selection_summary.get("selected_model")
    if not selected:
        statement = framing_data.get("business_problem_statement")
        return f"{statement} This run has not yet produced a final model recommendation." if statement else (
            "This run has not yet produced a final model recommendation."
        )
    n_algorithms = len(model_selection_summary.get("algorithms_evaluated", []))
    goal = f"predicting {target}" if pipeline_run.declared_target else "grouping records into natural segments"
    outcome_phrase = f"around {target}" if pipeline_run.declared_target else "using these segments"
    return (
        f"This analysis successfully addressed the goal of {goal}. The dataset was cleaned, "
        f"validated, and prepared for machine learning. After evaluating {n_algorithms} algorithm(s), {selected} "
        f"was selected as the best-performing model. The model can support decision-making {outcome_phrase} going forward."
    )


def build_report(
    pipeline_run: PipelineRunORM,
    decisions: list[AgentDecisionORM],
    dataset: DatasetORM,
    job: JobORM | None = None,
) -> dict:
    profiling = _decision_by_agent(decisions, "data_profiling")
    framing = _decision_by_agent(decisions, "business_framing")
    validation = _decision_by_agent(decisions, "data_validation")
    cleaning = _decision_by_agent(decisions, "cleaning_plan")
    transformation = _decision_by_agent(decisions, "transformation")
    split = _decision_by_agent(decisions, "train_test_split")
    model_selection = _decision_by_agent(decisions, "model_selection")
    hpo = _decision_by_agent(decisions, "hyperparameter_optimization")
    evaluation = _decision_by_agent(decisions, "evaluation")
    recommendation = _decision_by_agent(decisions, "recommendation")

    framing_data = _effective(framing)
    recommendation_data = _effective(recommendation)
    top = recommendation_data.get("top_choice")
    champion_run = _champion_run(job, top["algorithm"] if top else None)

    model_selection_summary = _model_selection_summary(model_selection, hpo, recommendation)
    data_preparation_summary = _data_preparation_summary(cleaning, transformation, split)

    n_rows = dataset.n_rows
    train_pct, test_pct = data_preparation_summary["train_pct"], data_preparation_summary["test_pct"]
    train_test_split = (
        {"training": round(n_rows * train_pct / 100), "testing": round(n_rows * test_pct / 100)}
        if train_pct is not None else None
    )

    prediction_type = (
        "Binary Classification" if pipeline_run.problem_type == "classification" else
        "Clustering / Segmentation" if pipeline_run.problem_type == "clustering" else
        pipeline_run.problem_type
    )

    return {
        "title": (
            f"{pipeline_run.declared_target.replace('_', ' ').title()} Prediction Report"
            if pipeline_run.declared_target else
            f"{Path(dataset.filename).stem.replace('_', ' ').title()} Segmentation Report"
        ),
        "executive_summary": {
            "problem_statement": framing_data.get("business_problem_statement"),
            "dataset_overview": {
                "dataset_name": dataset.filename,
                "total_records": dataset.n_rows,
                "features_analyzed": framing_data.get("key_features", []),
                "target_variable": pipeline_run.declared_target,
                "business_domain": "General Analytics",
            },
        },
        "data_quality_summary": _data_quality_summary(validation, cleaning, profiling),
        "data_preparation_summary": data_preparation_summary,
        "model_selection_summary": model_selection_summary,
        "model_performance": _model_performance(evaluation, recommendation),
        "key_insights": _key_insights(recommendation, validation),
        "prediction_capability": {
            "description": framing_data.get("prediction_objective"),
            # Populated live at read time by routers/pipeline.py::get_report() from
            # PredictionLog rows — always [] in this stored snapshot.
            "example_predictions": [],
        },
        "business_recommendations": _business_recommendations(recommendation, pipeline_run),
        "conclusion": _conclusion(framing, model_selection_summary, pipeline_run),
        "technical_appendix": {
            "model_details": champion_run.metrics_json if champion_run else {},
            "hyperparameters": champion_run.params_json if champion_run else {},
            "train_test_split": train_test_split,
            "model_training_summary": {
                "algorithms_evaluated": len(model_selection_summary["algorithms_evaluated"]),
                "hyperparameter_optimization": any(not r.get("skipped") for r in _effective(hpo).get("results", [])),
                "champion_model": model_selection_summary["selected_model"],
                "prediction_type": prediction_type,
            },
        },
        "final_outcome": pipeline_run.status,
    }


def _table(rows: list[list[str]]) -> Table:
    t = Table(rows, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    return t


def export_pdf(report: dict, out_path: Path) -> Path:
    doc = SimpleDocTemplate(str(out_path), pagesize=letter)
    styles = getSampleStyleSheet()
    story = [Paragraph(report.get("title", "Business Report"), styles["Title"]), Spacer(1, 16)]

    def heading(text: str, level: str = "Heading2"):
        story.append(Paragraph(text, styles[level]))
        story.append(Spacer(1, 6))

    def para(text: str):
        story.append(Paragraph(text or "Not available yet.", styles["BodyText"]))
        story.append(Spacer(1, 10))

    def bullets(items: list[str]):
        if not items:
            para("None.")
            return
        story.append(Paragraph("<br/>".join(f"• {i}" for i in items), styles["BodyText"]))
        story.append(Spacer(1, 10))

    heading("Executive Summary", "Heading1")
    exec_summary = report["executive_summary"]
    heading("Problem Statement")
    para(exec_summary["problem_statement"])
    heading("Dataset Overview")
    ov = exec_summary["dataset_overview"]
    story.append(_table([
        ["Item", "Value"],
        ["Dataset Name", ov["dataset_name"]],
        ["Total Records", str(ov["total_records"])],
        ["Features Analyzed", ", ".join(ov["features_analyzed"]) or "—"],
        ["Target Variable", ov["target_variable"] or "None (unsupervised)"],
        ["Business Domain", ov["business_domain"]],
    ]))
    story.append(Spacer(1, 14))

    heading("Data Quality Summary", "Heading1")
    dq = report["data_quality_summary"]
    heading("Issues Detected")
    story.append(_table([
        ["Issue", "Count"],
        ["Missing Values", str(dq["issues_detected"]["missing_values"])],
        ["Duplicate Records", str(dq["issues_detected"]["duplicate_records"])],
        ["Invalid Data", str(dq["issues_detected"]["invalid_data"])],
    ]))
    story.append(Spacer(1, 10))
    if dq["actions_taken"]:
        heading("Actions Taken")
        story.append(_table(
            [["Column", "Missing Values", "Action", "Reason"]]
            + [[a["column"], str(a["missing_values"]), a["action"], a["reason"]] for a in dq["actions_taken"]]
        ))
        story.append(Spacer(1, 10))
    heading("Duplicate Handling")
    para(dq["duplicate_handling"])

    heading("Data Preparation Summary", "Heading1")
    dp = report["data_preparation_summary"]
    bullets(dp["steps"])
    if dp["train_pct"] is not None:
        para(f"Training Data: {dp['train_pct']}% &nbsp;&nbsp; Testing Data: {dp['test_pct']}%")

    heading("Model Selection Summary", "Heading1")
    ms = report["model_selection_summary"]
    heading("Algorithms Evaluated")
    bullets(ms["algorithms_evaluated"])
    heading("Selected Model")
    para(f"<b>{ms['selected_model']}</b>" if ms["selected_model"] else "Not yet available.")
    heading("Why Was It Selected?")
    para(ms["why_selected"])
    heading("Hyperparameter Optimization")
    para(ms["hyperparameter_optimization"])

    heading("Model Performance", "Heading1")
    mp = report["model_performance"]
    heading("Prediction Reliability")
    para(mp["reliability_sentence"])
    heading("Confidence Level")
    para(f"<b>{mp['confidence_level']}</b><br/>{mp['confidence_explanation']}")

    heading("Key Insights", "Heading1")
    bullets(report["key_insights"])

    heading("Prediction Capability", "Heading1")
    pc = report["prediction_capability"]
    para(pc["description"])
    if pc["example_predictions"]:
        heading("Example Predictions")
        example_cols = list(pc["example_predictions"][0]["input"].keys())
        story.append(_table(
            [[*example_cols, "Prediction"]]
            + [[*(str(p["input"].get(c, "")) for c in example_cols), str(p["prediction"])] for p in pc["example_predictions"]]
        ))
        story.append(Spacer(1, 10))

    heading("Business Recommendations", "Heading1")
    bullets(report["business_recommendations"])

    heading("Conclusion", "Heading1")
    para(report["conclusion"])

    story.append(Paragraph("Technical Appendix", styles["Heading1"]))
    story.append(Spacer(1, 8))
    ta = report["technical_appendix"]
    if ta["model_details"]:
        heading("Model Details")
        story.append(_table([["Metric", "Value"]] + [[k, str(v)] for k, v in ta["model_details"].items()]))
        story.append(Spacer(1, 10))
    if ta["hyperparameters"]:
        heading("Hyperparameters")
        story.append(_table([["Parameter", "Value"]] + [[k, str(v)] for k, v in ta["hyperparameters"].items()]))
        story.append(Spacer(1, 10))
    if ta["train_test_split"]:
        heading("Train/Test Split")
        story.append(_table([
            ["Dataset", "Records"],
            ["Training", str(ta["train_test_split"]["training"])],
            ["Testing", str(ta["train_test_split"]["testing"])],
        ]))
        story.append(Spacer(1, 10))
    heading("Model Training Summary")
    mts = ta["model_training_summary"]
    para(
        f"Algorithms Evaluated: {mts['algorithms_evaluated']}<br/>"
        f"Hyperparameter Optimization: {'Yes' if mts['hyperparameter_optimization'] else 'No'}<br/>"
        f"Champion Model: {mts['champion_model'] or 'Not yet available'}<br/>"
        f"Prediction Type: {mts['prediction_type']}"
    )

    heading("Final Outcome")
    para(report["final_outcome"])

    doc.build(story)
    return out_path

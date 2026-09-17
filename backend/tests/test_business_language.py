from __future__ import annotations

from app.agents import business_framing_agent
from app.services import business_language_service


def test_metric_sentence_accuracy_matches_requirement_example():
    assert business_language_service.metric_sentence("accuracy", 0.92, "classification") == (
        "92 out of 100 predictions are expected to be correct."
    )


def test_metric_sentence_unknown_metric_returns_empty_string():
    assert business_language_service.metric_sentence("some_unknown_metric", 0.5) == ""


def test_metric_sentence_none_value_returns_empty_string():
    assert business_language_service.metric_sentence("accuracy", None) == ""


def test_ml_type_business_label_avoids_jargon():
    label = business_language_service.ml_type_business_label("classification")
    assert "classification" not in label.lower()


def test_business_action_for_prediction_bands_by_confidence():
    assert "act on this prediction directly" in business_language_service.business_action_for_prediction(0.95)
    assert "verify manually" in business_language_service.business_action_for_prediction(0.3)


def test_business_framing_agent_produces_problem_statement_with_target():
    framing = business_framing_agent.frame("sales.csv", 500, "classification", "will_buy")
    assert "will_buy" in framing["business_problem_statement"]
    assert framing["target_variable"] == "will_buy"
    assert "classification" not in framing["ml_type_business_label"].lower()


def test_business_framing_agent_clustering_has_no_target():
    framing = business_framing_agent.frame("customers.csv", 500, "clustering", None)
    assert framing["target_variable"] is None
    assert "group" in framing["business_problem_statement"].lower()

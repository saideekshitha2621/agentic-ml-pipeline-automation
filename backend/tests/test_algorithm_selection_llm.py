from __future__ import annotations

import json

import pandas as pd
import pytest

from app.agents import algorithm_selection_llm, algorithm_shortlist_agent, data_profiling_agent


def _df(n=50):
    return pd.DataFrame(
        {
            "customer_id": [f"C{i}" for i in range(n)],
            "age": [20 + (i % 40) for i in range(n)],
            "income": [30000.0 + i * 137.5 for i in range(n)],
            "region": (["north", "south", "east", "west"] * (n // 4 + 1))[:n],
            "churn": (["yes", "no"] * (n // 2))[:n],
        }
    )


@pytest.fixture
def profile():
    df = _df()
    return data_profiling_agent.analyze(df, {"data_quality_score": 90})


@pytest.fixture
def validation():
    return {
        "checks": [
            {"name": "Data leakage risk", "status": "ok", "detail": "none", "affected_columns": []},
            {"name": "Missing values", "status": "ok", "detail": "none", "affected_columns": []},
        ],
        "overall_status": "ok",
    }


def _valid_llm_json(algorithms: list[str], confidence: str = "high") -> str:
    return json.dumps(
        {
            "shortlist": [
                {"algorithm": a, "recommended": True, "rationale": f"Good fit: {a}.", "risks": ["none notable"]}
                for a in algorithms
            ],
            "selected_algorithms": algorithms,
            "confidence": confidence,
        }
    )


def test_successful_llm_recommendation_used_when_valid(monkeypatch, profile, validation):
    monkeypatch.setattr("app.services.llm_service.is_configured", lambda: True)
    monkeypatch.setattr(
        "app.services.llm_service._active_config",
        lambda: {"provider": "anthropic", "api_key": "x", "model": "m"},
    )
    monkeypatch.setattr(
        "app.services.llm_service.call",
        lambda *a, **k: _valid_llm_json(["random_forest", "logistic_regression"]),
    )

    result = algorithm_selection_llm.recommend(profile, validation, "classification", _df(), "churn")

    assert result["source"] == "llm"
    assert result["fallback_reason"] is None
    assert set(result["selected_algorithms"]) == {"random_forest", "logistic_regression"}
    assert "tool_outputs_used" in result and len(result["tool_outputs_used"]) > 0


def test_invalid_json_falls_back(monkeypatch, profile, validation):
    monkeypatch.setattr("app.services.llm_service.is_configured", lambda: True)
    monkeypatch.setattr(
        "app.services.llm_service._active_config",
        lambda: {"provider": "anthropic", "api_key": "x", "model": "m"},
    )
    monkeypatch.setattr("app.services.llm_service.call", lambda *a, **k: "this is not json at all {{{")

    result = algorithm_selection_llm.recommend(profile, validation, "classification", _df(), "churn")

    assert result["source"] == "fallback"
    assert result["fallback_reason"] == "invalid_json"
    expected = algorithm_shortlist_agent.recommend(profile, "classification")
    assert result["selected_algorithms"] == expected["selected_algorithms"]


def test_empty_response_falls_back(monkeypatch, profile, validation):
    monkeypatch.setattr("app.services.llm_service.is_configured", lambda: True)
    monkeypatch.setattr(
        "app.services.llm_service._active_config",
        lambda: {"provider": "anthropic", "api_key": "x", "model": "m"},
    )
    monkeypatch.setattr("app.services.llm_service.call", lambda *a, **k: "")

    result = algorithm_selection_llm.recommend(profile, validation, "classification", _df(), "churn")

    assert result["source"] == "fallback"
    assert result["fallback_reason"] == "empty_or_malformed_response"


def test_unsupported_algorithm_rejected_falls_back(monkeypatch, profile, validation):
    monkeypatch.setattr("app.services.llm_service.is_configured", lambda: True)
    monkeypatch.setattr(
        "app.services.llm_service._active_config",
        lambda: {"provider": "anthropic", "api_key": "x", "model": "m"},
    )
    monkeypatch.setattr(
        "app.services.llm_service.call",
        lambda *a, **k: _valid_llm_json(["xgboost_turbo", "made_up_algorithm"]),
    )

    result = algorithm_selection_llm.recommend(profile, validation, "classification", _df(), "churn")

    assert result["source"] == "fallback"
    assert result["fallback_reason"] == "no_valid_algorithms_in_response"


def test_api_failure_falls_back(monkeypatch, profile, validation):
    monkeypatch.setattr("app.services.llm_service.is_configured", lambda: True)
    monkeypatch.setattr(
        "app.services.llm_service._active_config",
        lambda: {"provider": "anthropic", "api_key": "x", "model": "m"},
    )

    def _raise(*a, **k):
        raise RuntimeError("provider timed out")

    monkeypatch.setattr("app.services.llm_service.call", _raise)

    result = algorithm_selection_llm.recommend(profile, validation, "classification", _df(), "churn")

    assert result["source"] == "fallback"
    assert result["fallback_reason"].startswith("llm_api_error")


def test_low_confidence_response_falls_back(monkeypatch, profile, validation):
    monkeypatch.setattr("app.services.llm_service.is_configured", lambda: True)
    monkeypatch.setattr(
        "app.services.llm_service._active_config",
        lambda: {"provider": "anthropic", "api_key": "x", "model": "m"},
    )
    monkeypatch.setattr(
        "app.services.llm_service.call",
        lambda *a, **k: _valid_llm_json(["random_forest"], confidence="low"),
    )

    result = algorithm_selection_llm.recommend(profile, validation, "classification", _df(), "churn")

    assert result["source"] == "fallback"
    assert result["fallback_reason"] == "low_confidence_response"


def test_no_provider_configured_uses_deterministic_fallback(monkeypatch, profile, validation):
    monkeypatch.setattr("app.services.llm_service.is_configured", lambda: False)

    def _should_not_be_called(*a, **k):
        raise AssertionError("llm_service.call should never be invoked when no provider is configured")

    monkeypatch.setattr("app.services.llm_service.call", _should_not_be_called)

    result = algorithm_selection_llm.recommend(profile, validation, "classification", _df(), "churn")

    assert result["source"] == "fallback"
    assert result["fallback_reason"] == "llm_not_configured"
    expected = algorithm_shortlist_agent.recommend(profile, "classification")
    assert result["shortlist"] == expected["shortlist"]
    assert result["selected_algorithms"] == expected["selected_algorithms"]

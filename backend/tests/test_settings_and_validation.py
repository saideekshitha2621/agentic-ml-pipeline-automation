from __future__ import annotations

import pandas as pd

from app.agents import data_profiling_agent, data_validation_agent
from app.services import error_translation_service, llm_service
from app.services.preprocessing_service import MissingValueValidationError


def _df(n=20):
    return pd.DataFrame({"age": list(range(n)), "region": (["N", "S"] * n)[:n], "target": (["yes", "no"] * n)[:n]})


def test_validate_flags_too_few_rows():
    df = _df(n=3)
    profile = data_profiling_agent.analyze(df, {})
    result = data_validation_agent.validate(df, profile, target_column="target", problem_type="classification")
    check = next(c for c in result["checks"] if c["name"] == "Dataset size")
    assert check["status"] == "critical"


def test_validate_flags_empty_dataset():
    df = pd.DataFrame({"age": [], "target": []})
    profile = data_profiling_agent.analyze(df, {})
    result = data_validation_agent.validate(df, profile, target_column="target", problem_type="classification")
    check = next(c for c in result["checks"] if c["name"] == "Dataset size")
    assert check["status"] == "critical"


def test_validate_flags_missing_target_column():
    df = _df()
    profile = data_profiling_agent.analyze(df, {})
    result = data_validation_agent.validate(df, profile, target_column="does_not_exist", problem_type="classification")
    check = next(c for c in result["checks"] if c["name"] == "Target column validity")
    assert check["status"] == "critical"


def test_validate_flags_single_class_target_for_classification():
    df = _df()
    df["target"] = "only_one_value"
    profile = data_profiling_agent.analyze(df, {})
    result = data_validation_agent.validate(df, profile, target_column="target", problem_type="classification")
    check = next(c for c in result["checks"] if c["name"] == "Target column validity")
    assert check["status"] == "critical"


def test_validate_passes_valid_target():
    df = _df()
    profile = data_profiling_agent.analyze(df, {})
    result = data_validation_agent.validate(df, profile, target_column="target", problem_type="classification")
    check = next(c for c in result["checks"] if c["name"] == "Target column validity")
    assert check["status"] == "ok"


def test_error_translation_passes_through_missing_value_error():
    exc = MissingValueValidationError({"age": 5})
    assert "age" in error_translation_service.to_business_message(exc)


def test_error_translation_rewords_known_sklearn_message():
    exc = ValueError("Input contains NaN")
    message = error_translation_service.to_business_message(exc)
    assert "stack" not in message.lower()
    assert "missing values" in message.lower()


def test_error_translation_generic_fallback_has_no_raw_exception_repr():
    exc = RuntimeError("some obscure internal detail")
    message = error_translation_service.to_business_message(exc)
    assert "RuntimeError" in message
    assert "some obscure internal detail" not in message


def test_llm_service_env_fallback(monkeypatch):
    monkeypatch.setattr("app.services.llm_service.ANTHROPIC_API_KEY", None)
    monkeypatch.setattr("app.services.llm_service.GEMINI_API_KEY", "env-gemini-key")
    monkeypatch.setattr("app.services.llm_service.OPENAI_API_KEY", None)

    config = llm_service._active_config()
    assert config["provider"] == "gemini"
    assert config["api_key"] == "env-gemini-key"


def test_llm_service_no_provider_configured_falls_back_to_template(monkeypatch):
    monkeypatch.setattr("app.services.llm_service.ANTHROPIC_API_KEY", None)
    monkeypatch.setattr("app.services.llm_service.GEMINI_API_KEY", None)
    monkeypatch.setattr("app.services.llm_service.OPENAI_API_KEY", None)

    assert llm_service.is_configured() is False
    assert llm_service.explain("test", {}, fallback="the template text") == "the template text"

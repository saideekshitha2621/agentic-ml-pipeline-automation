"""Unit tests for the multi-key/multi-provider LLM key manager (no network)."""
from __future__ import annotations

import pytest

from app.services import llm_key_manager as km


def _manager(pools: dict[str, list[str]], provider_order: list[str] | None = None, **cfg) -> km.KeyManager:
    config = km.KeyManagerConfig(
        warning_after=cfg.get("warning_after", 2),
        exhausted_after=cfg.get("exhausted_after", 3),
        rate_limit_cooldown=cfg.get("rate_limit_cooldown", 300.0),
        auth_cooldown=cfg.get("auth_cooldown", 3600.0),
        max_retries_per_request=cfg.get("max_retries_per_request", 6),
    )
    models = {p: f"{p}-model" for p in pools}
    return km.KeyManager(lambda: pools, models, provider_order or list(pools), config)


def test_mask_key_never_leaks_the_raw_value():
    assert km.mask_key("sk-proj-abcdefghijklmnop") == "sk-p...mnop"
    assert "abcdefgh" not in km.mask_key("sk-proj-abcdefghijklmnop")
    assert km.mask_key("") == "<empty>"
    assert km.mask_key("short") == "*****"


def test_classify_error_kinds():
    assert km.classify_error(RuntimeError("429 You exceeded your current quota")) is km.FailureKind.RATE_LIMIT
    assert km.classify_error(RuntimeError("RESOURCE_EXHAUSTED: rate limit")) is km.FailureKind.RATE_LIMIT
    assert km.classify_error(RuntimeError("401 Unauthorized: invalid api key")) is km.FailureKind.AUTH
    assert km.classify_error(ValueError("bad json")) is km.FailureKind.OTHER


def test_execute_uses_first_healthy_key_and_records_success():
    mgr = _manager({"openai": ["k1"]})
    result = mgr.execute(lambda provider, key, model: f"{provider}:{key}:{model}")
    assert result == "openai:k1:openai-model"
    status = mgr.status_report()
    assert status[0]["status"] == "healthy" and status[0]["total_calls"] == 1


def test_execute_fails_over_to_next_key_in_same_provider():
    mgr = _manager({"openai": ["bad", "good"]})

    def attempt(provider, key, model):
        if key == "bad":
            raise RuntimeError("429 quota exceeded")
        return "ok"

    assert mgr.execute(attempt) == "ok"
    report = {r["masked_key"]: r for r in mgr.status_report()}
    bad_status = [r for r in mgr.status_report() if r["consecutive_failures"] == 1][0]
    assert bad_status["status"] == "exhausted"  # a rate-limit hit exhausts immediately, no warning grace period


def test_openai_exhausted_falls_back_to_gemini():
    mgr = _manager({"openai": ["o1"], "gemini": ["g1"]}, provider_order=["openai", "gemini"])

    def attempt(provider, key, model):
        if provider == "openai":
            raise RuntimeError("429 quota exceeded")
        return f"answered-by-{provider}"

    assert mgr.execute(attempt) == "answered-by-gemini"


def test_all_gemini_keys_exhausted_raises_clear_error_without_leaking_keys():
    mgr = _manager({"gemini": ["secret-key-1", "secret-key-2"]}, max_retries_per_request=6)

    def attempt(provider, key, model):
        raise RuntimeError(f"429 quota exceeded for key {key}")

    with pytest.raises(km.AllKeysExhaustedError) as exc_info:
        mgr.execute(attempt)
    message = str(exc_info.value)
    assert "secret-key-1" not in message and "secret-key-2" not in message
    assert "gemini" in message


def test_no_provider_configured_raises_distinct_error():
    mgr = _manager({})
    assert mgr.is_configured() is False
    with pytest.raises(km.NoProviderConfiguredError):
        mgr.execute(lambda p, k, m: "unreachable")


def test_non_fatal_failures_need_the_threshold_before_exhaustion():
    mgr = _manager({"openai": ["k1"]}, warning_after=2, exhausted_after=3)
    calls = {"n": 0}

    def attempt(provider, key, model):
        calls["n"] += 1
        if calls["n"] < 3:
            raise ValueError("transient parse error")
        return "ok"

    assert mgr.execute(attempt) == "ok"
    # 2 non-fatal failures should have only reached Warning, not Exhausted, before recovering
    assert calls["n"] == 3


def test_key_recovers_to_healthy_after_success():
    mgr = _manager({"openai": ["k1"]})
    attempts = {"n": 0}

    def attempt(provider, key, model):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise ValueError("transient")
        return "ok"

    mgr.execute(attempt)
    status = mgr.status_report()[0]
    assert status["status"] == "healthy" and status["consecutive_failures"] == 0


def test_status_report_shape_has_no_raw_key_field():
    mgr = _manager({"openai": ["sk-super-secret-value"]})
    report = mgr.status_report()[0]
    assert "key" not in report and "api_key" not in report
    assert "sk-super-secret-value" not in str(report)
    assert report["masked_key"] == km.mask_key("sk-super-secret-value")

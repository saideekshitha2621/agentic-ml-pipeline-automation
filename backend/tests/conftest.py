"""Test-wide safety: the app loads backend/.env, so without this every test that reaches an
LLM call (narratives, tool loops) would hit the real provider and burn the developer's API
quota. Blanking the key constants keeps the real `_active_config` logic under test while
guaranteeing no live calls; tests that need an LLM install a scripted fake
(see `fake_llm` in test_agentic_loops) or set the constants themselves."""
import pytest

from app.services import llm_service


@pytest.fixture(autouse=True)
def _no_live_llm(monkeypatch):
    for name in ("ANTHROPIC_API_KEY", "GEMINI_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.setattr(llm_service, name, None)
    llm_service.reset_breaker()
    yield
    llm_service.reset_breaker()

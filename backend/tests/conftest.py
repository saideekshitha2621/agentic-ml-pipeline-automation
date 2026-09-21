"""Test-wide safety.

* The app loads backend/.env, so without this every test that reaches an LLM call (narratives,
  tool loops) would hit the real provider and burn the developer's API quota. Blanking the key
  constants keeps the real `_active_config` logic under test while guaranteeing no live calls;
  tests that need an LLM install a scripted fake (see `fake_llm` in test_agentic_loops) or set
  the constants themselves.
* Background tasks run inline so HTTP-level tests see the effect of a request immediately, and
  API-key auth is off unless a test switches it on.
"""
import pytest

from app.core import auth
from app.services import llm_service, task_queue_service


@pytest.fixture(autouse=True)
def _isolated_environment(monkeypatch):
    for name in ("ANTHROPIC_API_KEY", "GEMINI_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.setattr(llm_service, name, None)
    monkeypatch.setattr(task_queue_service, "BACKEND", "inline")
    monkeypatch.setattr(auth, "API_KEYS", set())
    llm_service.reset_breaker()
    yield
    llm_service.reset_breaker()

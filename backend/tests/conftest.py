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
    for name in ("ANTHROPIC_API_KEY", "GEMINI_API_KEY", "OPENAI_API_KEY", "GROQ_API_KEY"):
        monkeypatch.setattr(llm_service, name, None)
        prefix = name.rsplit("_API_KEY", 1)[0]
        monkeypatch.delenv(f"{prefix}_API_KEYS", raising=False)
        for i in range(1, 21):  # llm_key_manager's numbered multi-key variant (non-contiguous scan, 1..20)
            monkeypatch.delenv(f"{prefix}_API_KEY_{i}", raising=False)
    # The Governance Proxy (enabled by GOVERNANCE_BASE_URL + GOVERNANCE_KEY in .env) is the first
    # provider in the order and would receive every "LLM" call, ahead of any scripted fake.
    monkeypatch.setattr(llm_service, "GOVERNANCE_ENABLED", False)
    monkeypatch.setattr(llm_service.key_manager, "_provider_order", list(llm_service._DIRECT_PROVIDER_ORDER))
    monkeypatch.setattr(task_queue_service, "BACKEND", "inline")
    monkeypatch.setattr(auth, "API_KEYS", set())
    llm_service.reset_breaker()
    llm_service.key_manager.reset()
    yield
    llm_service.reset_breaker()
    llm_service.key_manager.reset()

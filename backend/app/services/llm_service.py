"""Thin wrapper around the Anthropic Messages API, used for two things:

1. `explain()` — turn a stage's structured decision payload into a business-language
   narrative sentence or two. Every new agent calls this instead of hand-writing prose so
   the whole pipeline reads consistently. Falls back to a plain template render when no
   `ANTHROPIC_API_KEY` is configured (or the call fails) so nothing in the pipeline ever
   hard-depends on having a key — narratives are just less polished without one.
2. `chat()` — the Conversational Q&A endpoint. Builds a grounding context out of a
   pipeline run's own stored artifacts (profile, every AgentDecision, champion metrics,
   feature importances) and answers a free-form question against it.
"""
from __future__ import annotations

from app.core.config import ANTHROPIC_API_KEY, ANTHROPIC_MODEL

_client = None
_client_init_failed = False


def _get_client():
    global _client, _client_init_failed
    if _client is not None or _client_init_failed:
        return _client
    if not ANTHROPIC_API_KEY:
        _client_init_failed = True
        return None
    try:
        import anthropic

        _client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    except Exception:
        _client_init_failed = True
        _client = None
    return _client


def is_configured() -> bool:
    return _get_client() is not None


def explain(kind: str, context: dict, fallback: str) -> str:
    """`fallback` is the deterministic template string the caller already built —
    used verbatim when no LLM is configured, and as the safety net if the call fails."""
    client = _get_client()
    if client is None:
        return fallback
    try:
        response = client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=220,
            system=(
                "You are an AutoML pipeline agent explaining a technical decision to a "
                "business stakeholder in 1-3 plain-English sentences. No jargon, no markdown, "
                "no preamble like 'Sure' or 'Here is'. Be concrete: cite the numbers given."
            ),
            messages=[
                {
                    "role": "user",
                    "content": f"Stage: {kind}\nStructured data: {context}\n\nExplain this decision.",
                }
            ],
        )
        text = "".join(block.text for block in response.content if hasattr(block, "text")).strip()
        return text or fallback
    except Exception:
        return fallback


def chat(question: str, grounding_context: str, history: list[dict]) -> str:
    client = _get_client()
    if client is None:
        return (
            "Conversational Q&A needs an ANTHROPIC_API_KEY configured on the backend. "
            "Once set, I can answer questions grounded in this run's dataset profile, "
            "agent decisions, and model results."
        )
    try:
        messages = [{"role": h["role"], "content": h["content"]} for h in history]
        messages.append({"role": "user", "content": question})
        response = client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=600,
            system=(
                "You are an assistant embedded in an explainable AutoML pipeline. Answer the "
                "user's question about their dataset, preprocessing, models, or predictions "
                "using ONLY the grounding context below — it is the actual recorded state of "
                "this pipeline run. If the context doesn't cover the question, say so rather "
                "than guessing. Be concise and business-friendly.\n\n"
                f"=== Pipeline run context ===\n{grounding_context}"
            ),
            messages=messages,
        )
        return "".join(block.text for block in response.content if hasattr(block, "text")).strip()
    except Exception as exc:  # noqa: BLE001
        return f"Sorry, the chat model call failed: {exc}"

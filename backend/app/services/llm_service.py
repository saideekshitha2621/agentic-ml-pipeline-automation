"""Multi-provider LLM wrapper (Anthropic/Gemini/OpenAI), used for two things:

1. `explain()` — turn a stage's structured decision payload into a business-language
   narrative sentence or two. Falls back to a plain template render when no provider is
   configured (or the call fails) so nothing in the pipeline ever hard-depends on an LLM —
   narratives are just less polished without one.
2. `chat()` — the Conversational Q&A endpoint. Builds a grounding context out of a
   pipeline run's own stored artifacts and answers a free-form question against it.

Which provider/key is active is resolved per-call via `llm_config_service.get_active_config`
(DB Settings override, else env vars, else None) — see that module for exactly where keys
can be configured. Each provider's SDK is imported lazily inside a try/except so a package
that isn't installed just makes that provider unavailable rather than crashing the app.
"""
from __future__ import annotations

from sqlalchemy.orm import Session

from app.services import llm_config_service

_SYSTEM_EXPLAIN = (
    "You are an AutoML pipeline agent explaining a technical decision to a business "
    "stakeholder in 1-3 plain-English sentences. No jargon, no markdown, no preamble like "
    "'Sure' or 'Here is'. Be concrete: cite the numbers given."
)


def _call_anthropic(api_key: str, model: str, system: str, messages: list[dict], max_tokens: int) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(model=model, max_tokens=max_tokens, system=system, messages=messages)
    return "".join(block.text for block in response.content if hasattr(block, "text")).strip()


def _call_openai(api_key: str, model: str, system: str, messages: list[dict], max_tokens: int) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model, max_tokens=max_tokens,
        messages=[{"role": "system", "content": system}, *messages],
    )
    return (response.choices[0].message.content or "").strip()


def _call_gemini(api_key: str, model: str, system: str, messages: list[dict], max_tokens: int) -> str:
    import google.generativeai as genai

    genai.configure(api_key=api_key)
    gemini_model = genai.GenerativeModel(model, system_instruction=system)
    history = [{"role": "model" if m["role"] == "assistant" else "user", "parts": [m["content"]]} for m in messages[:-1]]
    chat = gemini_model.start_chat(history=history)
    response = chat.send_message(
        messages[-1]["content"], generation_config={"max_output_tokens": max_tokens}
    )
    return (response.text or "").strip()


_DISPATCH = {"anthropic": _call_anthropic, "openai": _call_openai, "gemini": _call_gemini}


def call(provider: str, api_key: str, model: str, system: str, messages: list[dict], max_tokens: int = 400) -> str:
    """Raises on failure — callers decide whether to fall back or surface the error
    (the Settings page's Test Connection wants the real error; explain()/chat() below
    want a silent fallback)."""
    fn = _DISPATCH.get(provider)
    if fn is None:
        raise ValueError(f"Unknown LLM provider '{provider}'.")
    return fn(api_key, model, system, messages, max_tokens)


def is_configured(db: Session) -> bool:
    return llm_config_service.get_active_config(db) is not None


def explain(kind: str, context: dict, fallback: str, db: Session | None = None) -> str:
    """`fallback` is the deterministic template string the caller already built — used
    verbatim when no LLM is configured, and as the safety net if the call fails."""
    config = llm_config_service.get_active_config(db) if db is not None else None
    if config is None:
        return fallback
    try:
        text = call(
            config["provider"], config["api_key"], config["model"], _SYSTEM_EXPLAIN,
            [{"role": "user", "content": f"Stage: {kind}\nStructured data: {context}\n\nExplain this decision."}],
            max_tokens=220,
        )
        return text or fallback
    except Exception:
        return fallback


def chat(question: str, grounding_context: str, history: list[dict], db: Session | None = None) -> str:
    config = llm_config_service.get_active_config(db) if db is not None else None
    if config is None:
        return (
            "Conversational Q&A needs an LLM provider configured — set one up on the "
            "Settings page (or an ANTHROPIC_API_KEY/GEMINI_API_KEY/OPENAI_API_KEY env var). "
            "Once set, I can answer questions grounded in this run's dataset profile, agent "
            "decisions, and model results."
        )
    try:
        system = (
            "You are an assistant embedded in an explainable AutoML pipeline. Answer the "
            "user's question about their dataset, preprocessing, models, or predictions "
            "using ONLY the grounding context below — it is the actual recorded state of "
            "this pipeline run. If the context doesn't cover the question, say so rather "
            "than guessing. Be concise and business-friendly.\n\n"
            f"=== Pipeline run context ===\n{grounding_context}"
        )
        messages = [*history, {"role": "user", "content": question}]
        return call(config["provider"], config["api_key"], config["model"], system, messages, max_tokens=600)
    except Exception as exc:  # noqa: BLE001
        return f"Sorry, the chat model call failed: {exc}"

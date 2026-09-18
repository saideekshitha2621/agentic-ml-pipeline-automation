"""Multi-provider LLM wrapper (Anthropic/Gemini/OpenAI), used for two things:

1. `explain()` — turn a stage's structured decision payload into a business-language
   narrative sentence or two. Falls back to a plain template render when no provider is
   configured (or the call fails) so nothing in the pipeline ever hard-depends on an LLM —
   narratives are just less polished without one.
2. `chat()` — the Conversational Q&A endpoint. Builds a grounding context out of a
   pipeline run's own stored artifacts and answers a free-form question against it.

Provider/key configuration is env-var only — set exactly one of ANTHROPIC_API_KEY,
GEMINI_API_KEY, or OPENAI_API_KEY (see app/core/config.py, checked in that order). Each
provider's SDK is imported lazily inside a try/except so a package that isn't installed
just makes that provider unavailable rather than crashing the app.
"""
from __future__ import annotations

from app.core.config import ANTHROPIC_API_KEY, ANTHROPIC_MODEL, GEMINI_API_KEY, OPENAI_API_KEY

_SYSTEM_EXPLAIN = (
    "You are an AutoML pipeline agent explaining a technical decision to a business "
    "stakeholder in 1-3 plain-English sentences. No jargon, no markdown, no preamble like "
    "'Sure' or 'Here is'. Be concrete: cite the numbers given."
)

_DEFAULT_MODELS = {"anthropic": ANTHROPIC_MODEL, "gemini": "gemini-3.6-flash", "openai": "gpt-4o-mini"}


def _active_config() -> dict | None:
    for provider, api_key in (("anthropic", ANTHROPIC_API_KEY), ("gemini", GEMINI_API_KEY), ("openai", OPENAI_API_KEY)):
        if api_key:
            return {"provider": provider, "api_key": api_key, "model": _DEFAULT_MODELS[provider]}
    return None


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
    # Gemini's newer models spend part of max_output_tokens on internal reasoning before
    # the visible answer — finish_reason 1 is STOP (a complete answer); anything else
    # (2 = MAX_TOKENS, hit before finishing) means what came back, if anything, is a
    # mid-sentence fragment. Raise instead of returning it, so the caller's fallback
    # template is used rather than silently showing garbled, truncated text.
    finish_reason = response.candidates[0].finish_reason if response.candidates else None
    if finish_reason is not None and int(finish_reason) != 1:
        raise RuntimeError(f"Gemini response did not finish cleanly (finish_reason={finish_reason}); discarding partial output.")
    return (response.text or "").strip()


_DISPATCH = {"anthropic": _call_anthropic, "openai": _call_openai, "gemini": _call_gemini}


def call(provider: str, api_key: str, model: str, system: str, messages: list[dict], max_tokens: int = 400) -> str:
    fn = _DISPATCH.get(provider)
    if fn is None:
        raise ValueError(f"Unknown LLM provider '{provider}'.")
    return fn(api_key, model, system, messages, max_tokens)


def is_configured() -> bool:
    return _active_config() is not None


def explain(kind: str, context: dict, fallback: str) -> str:
    """`fallback` is the deterministic template string the caller already built — used
    verbatim when no LLM is configured, and as the safety net if the call fails."""
    config = _active_config()
    if config is None:
        return fallback
    try:
        text = call(
            config["provider"], config["api_key"], config["model"], _SYSTEM_EXPLAIN,
            [{"role": "user", "content": f"Stage: {kind}\nStructured data: {context}\n\nExplain this decision."}],
            # Generous budget — some providers' newer models spend part of this on internal
            # reasoning before the visible answer, so a tight limit risks truncation.
            max_tokens=1024,
        )
        return text or fallback
    except Exception:
        return fallback


def chat(question: str, grounding_context: str, history: list[dict]) -> str:
    config = _active_config()
    if config is None:
        return (
            "Conversational Q&A needs an LLM provider configured — set ANTHROPIC_API_KEY, "
            "GEMINI_API_KEY, or OPENAI_API_KEY in the backend's environment. Once set, I can "
            "answer questions grounded in this run's dataset profile, agent decisions, and "
            "model results."
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
        return call(config["provider"], config["api_key"], config["model"], system, messages, max_tokens=1200)
    except Exception as exc:  # noqa: BLE001
        return f"Sorry, the chat model call failed: {exc}"

"""Multi-provider LLM wrapper (Anthropic/Gemini/OpenAI/Groq), used for two things:

1. `explain()` — turn a stage's structured decision payload into a business-language
   narrative sentence or two. Falls back to a plain template render when no provider is
   configured (or the call fails) so nothing in the pipeline ever hard-depends on an LLM —
   narratives are just less polished without one.
2. `chat()` — the Conversational Q&A endpoint. Builds a grounding context out of a
   pipeline run's own stored artifacts and answers a free-form question against it.

Provider/key configuration is env-var driven. The simple case is still exactly one of
ANTHROPIC_API_KEY, GEMINI_API_KEY, or OPENAI_API_KEY (see app/core/config.py). For
multiple keys per provider — so a quota/rate-limit hit on one key fails over to the next
instead of stalling the pipeline — also set `<PROVIDER>_API_KEY_1`, `_2`, `_3`, ... or a
comma-separated `<PROVIDER>_API_KEYS`; any combination of these is pooled together and
deduplicated. All of that pooling, health tracking (Healthy/Warning/Exhausted per key),
and provider-level fallback (e.g. every OpenAI key exhausted -> try Gemini; every Gemini
key exhausted -> a clear error) lives in `llm_key_manager.py` — this module just supplies
the env-reading glue and the per-provider transport functions below. Each provider's SDK
is imported lazily inside a try/except so a package that isn't installed just makes that
provider unavailable rather than crashing the app.
"""
from __future__ import annotations

import os

from app.core.config import ANTHROPIC_API_KEY, ANTHROPIC_MODEL, GEMINI_API_KEY, GROQ_API_KEY, OPENAI_API_KEY
from app.services import llm_key_manager as _km

_SYSTEM_EXPLAIN = (
    "You are an AutoML pipeline agent explaining a technical decision to a business "
    "stakeholder in 1-3 plain-English sentences. No jargon, no markdown, no preamble like "
    "'Sure' or 'Here is'. Be concrete: cite the numbers given."
)

_DEFAULT_MODELS = {
    "anthropic": ANTHROPIC_MODEL, "gemini": "gemini-3.6-flash", "openai": "gpt-4o-mini",
    # openai/gpt-oss-120b (Groq's previous default) intermittently emits a tool call named
    # "JSON" that was never declared once it's done reasoning and wants to hand back its
    # final answer — Groq's API rejects the whole response with a 400 before it ever reaches
    # _execute_tool, breaking every agent that uses run_tool_loop. openai/gpt-oss-20b instead
    # loops without ever emitting a final answer. qwen/qwen3.8-27b has reliable native
    # function-calling on Groq and returns clean JSON — verified against this account's
    # actual available model list (`client.models.list()`), which does not include the
    # llama-3.x models many Groq docs/examples assume.
    "groq": os.environ.get("GROQ_MODEL", "qwen/qwen3.8-27b"),
}

# Anthropic first (unchanged from the original single-key precedence), then OpenAI, Gemini,
# Groq — override with LLM_PROVIDER_ORDER="groq,anthropic,openai,gemini" (comma-separated) if
# desired, e.g. to try a fast/free Groq key before the others.
_env_order = os.environ.get("LLM_PROVIDER_ORDER")
_PROVIDER_ORDER = [p.strip() for p in _env_order.split(",") if p.strip()] if _env_order else ["anthropic", "openai", "gemini", "groq"]


def _pooled_keys(prefix: str, primary: str | None) -> list[str]:
    """One provider's full key pool: the legacy single env var (kept as a module global so
    tests can monkeypatch it directly) plus any `_1`/`_2`/... or comma-separated `_KEYS`
    variants read fresh from the environment, deduplicated, order preserved."""
    keys: list[str] = []
    seen: set[str] = set()

    def _add(value: str | None) -> None:
        value = (value or "").strip()
        if value and value not in seen:
            seen.add(value)
            keys.append(value)

    _add(primary)
    for part in (os.environ.get(f"{prefix}_API_KEYS") or "").split(","):
        _add(part)
    # Non-contiguous on purpose: a plain <PREFIX>_API_KEY already covers "key #1", so it's
    # natural to add a second key as _2 without ever setting a _1. Scan a fixed range
    # instead of stopping at the first gap.
    for i in range(1, 21):
        _add(os.environ.get(f"{prefix}_API_KEY_{i}"))
    return keys


def _resolve_all_keys() -> dict[str, list[str]]:
    # Read the *current* module globals (not captured at import time) so tests that
    # monkeypatch ANTHROPIC_API_KEY/GEMINI_API_KEY/OPENAI_API_KEY keep working unchanged.
    return {
        "anthropic": _pooled_keys("ANTHROPIC", ANTHROPIC_API_KEY),
        "openai": _pooled_keys("OPENAI", OPENAI_API_KEY),
        "gemini": _pooled_keys("GEMINI", GEMINI_API_KEY),
        "groq": _pooled_keys("GROQ", GROQ_API_KEY),
    }


key_manager = _km.KeyManager(_resolve_all_keys, _DEFAULT_MODELS, _PROVIDER_ORDER)


def _active_config() -> dict | None:
    """Single best-next candidate (provider/api_key/model), respecting key health and
    round-robin rotation. Kept as the stable, single-key-shaped entry point some callers
    (and tests) use directly; `key_manager.execute()` below is what gives multi-attempt
    retry-with-failover to the higher-level helpers in this module."""
    picked = key_manager.next_candidate()
    if picked is None:
        return None
    provider, record = picked
    return {"provider": provider, "api_key": record.key, "model": key_manager.model_for(provider)}


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


_GROQ_BASE_URL = "https://api.groq.com/openai/v1"


def _call_groq(api_key: str, model: str, system: str, messages: list[dict], max_tokens: int) -> str:
    # Groq's API is OpenAI-compatible (same request/response shape) — the `openai` package's
    # client just needs to be pointed at Groq's base_url instead of OpenAI's.
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=_GROQ_BASE_URL)
    response = client.chat.completions.create(
        model=model, max_tokens=max_tokens,
        messages=[{"role": "system", "content": system}, *messages],
    )
    choice = response.choices[0]
    text = (choice.message.content or "").strip()
    # Reasoning models (e.g. openai/gpt-oss-*) can spend the entire max_tokens budget on the
    # internal `reasoning` field and return empty `content` with finish_reason="length" — the
    # same failure mode already handled for Gemini below. Raise instead of returning empty
    # text, so the caller's deterministic fallback is used rather than silently showing nothing.
    if not text and choice.finish_reason == "length":
        raise RuntimeError("Groq response spent the entire token budget on reasoning with no visible answer; discarding empty output.")
    return text


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


_DISPATCH = {"anthropic": _call_anthropic, "openai": _call_openai, "gemini": _call_gemini, "groq": _call_groq}


# --- Rate-limit circuit breaker ----------------------------------------------------------
# Free/low-tier keys have small daily quotas (e.g. Gemini free tier: 20 requests/day/model),
# and every agent already has a deterministic fallback. After a quota/rate-limit error, stop
# calling the provider for BREAKER_SECONDS instead of paying a failing round-trip per stage.
import time as _time

BREAKER_SECONDS = 300
_breaker_until = 0.0
_stats = {"calls": 0, "errors": 0, "breaker_trips": 0, "skipped_while_open": 0}


def _is_rate_limited(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(k in text for k in ("429", "quota", "rate limit", "rate_limit", "resource_exhausted", "resource exhausted"))


def breaker_open() -> bool:
    return _time.monotonic() < _breaker_until


def _trip_breaker() -> None:
    global _breaker_until
    _breaker_until = _time.monotonic() + BREAKER_SECONDS
    _stats["breaker_trips"] += 1


def reset_breaker() -> None:
    global _breaker_until
    _breaker_until = 0.0


def usage_stats() -> dict:
    """`keys` is the per-key health board (provider, masked key id, Healthy/Warning/
    Exhausted status, counts) — safe to expose on an internal monitoring endpoint since no
    raw key ever appears in it."""
    return {**_stats, "breaker_open": breaker_open(), "keys": key_manager.status_report()}


def _guarded(fn, *args, **kwargs):
    if breaker_open():
        _stats["skipped_while_open"] += 1
        raise RuntimeError("LLM temporarily disabled after a rate-limit/quota error; using deterministic fallback.")
    _stats["calls"] += 1
    try:
        return fn(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001
        _stats["errors"] += 1
        if _is_rate_limited(exc):
            _trip_breaker()
        raise


def call(provider: str, api_key: str, model: str, system: str, messages: list[dict], max_tokens: int = 400) -> str:
    fn = _DISPATCH.get(provider)
    if fn is None:
        raise ValueError(f"Unknown LLM provider '{provider}'.")
    return _guarded(fn, api_key, model, system, messages, max_tokens)


def _dispatch_call(provider: str, api_key: str, model: str, system: str, messages: list[dict], max_tokens: int) -> str:
    """Bare provider call used by `key_manager.execute()` — no `_guarded`/global breaker
    here; per-key health and cooldown is the key manager's job, so one key's rate limit
    doesn't block a *different* key or provider from being tried immediately after."""
    fn = _DISPATCH.get(provider)
    if fn is None:
        raise ValueError(f"Unknown LLM provider '{provider}'.")
    return fn(api_key, model, system, messages, max_tokens)


def complete(system: str, user: str, max_tokens: int = 1500) -> str:
    """Completion with automatic key/provider failover; raises ToolLoopUnavailable when
    nothing is configured at all (callers fall back deterministically), or
    `llm_key_manager.AllKeysExhaustedError` (a RuntimeError) when every configured key is
    currently exhausted — both are safe to surface, neither ever contains a raw key."""
    try:
        return key_manager.execute(
            lambda provider, api_key, model: _dispatch_call(provider, api_key, model, system, [{"role": "user", "content": user}], max_tokens)
        )
    except _km.NoProviderConfiguredError as exc:
        raise ToolLoopUnavailable(str(exc)) from exc


def is_configured() -> bool:
    return key_manager.is_configured()


def explain(kind: str, context: dict, fallback: str) -> str:
    """`fallback` is the deterministic template string the caller already built — used
    verbatim when no LLM is configured, and as the safety net if every configured key
    fails (rate-limited, exhausted, or otherwise)."""
    if not key_manager.is_configured():
        return fallback
    try:
        text = key_manager.execute(
            lambda provider, api_key, model: _dispatch_call(
                provider, api_key, model, _SYSTEM_EXPLAIN,
                [{"role": "user", "content": f"Stage: {kind}\nStructured data: {context}\n\nExplain this decision."}],
                # Generous budget — some providers' newer models spend part of this on internal
                # reasoning before the visible answer, so a tight limit risks truncation.
                1024,
            )
        )
        return text or fallback
    except Exception:
        return fallback


def chat(question: str, grounding_context: str, history: list[dict]) -> str:
    if not key_manager.is_configured():
        return (
            "Conversational Q&A needs an LLM provider configured — set ANTHROPIC_API_KEY, "
            "GEMINI_API_KEY, OPENAI_API_KEY, or GROQ_API_KEY (or their _1/_2/... numbered / "
            "_KEYS comma-separated multi-key variants) in the backend's environment. Once "
            "set, I can answer questions grounded in this run's dataset profile, agent "
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
        return key_manager.execute(lambda provider, api_key, model: _dispatch_call(provider, api_key, model, system, messages, 1200))
    except Exception as exc:  # noqa: BLE001 — AllKeysExhaustedError's message is already key-safe
        return f"Sorry, the chat model call failed: {exc}"


# --- Tool-calling loop (Phase 3) -------------------------------------------------------
# `call()` above is single-shot. Agents that should *investigate* (inspect a column, probe
# for leakage, then decide) use `run_tool_loop`: the model may request tools, we execute
# them locally (read-only Python functions) and feed the results back until it answers.
# Anthropic, OpenAI and Gemini all have native function-calling adapters; a provider without
# one raises `ToolLoopUnavailable`, which callers treat as "use the deterministic fallback".
import json as _json
from dataclasses import dataclass, field
from typing import Any, Callable


class ToolLoopUnavailable(RuntimeError):
    """No provider configured, or the configured provider has no tool-use adapter."""


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict  # JSON-schema object describing the arguments
    fn: Callable[..., Any]


@dataclass
class ToolLoopResult:
    text: str
    calls: list[dict] = field(default_factory=list)  # [{"tool", "args", "ok"}] audit trace


MAX_TOOL_RESULT_CHARS = 4000


class _AnthropicSession:
    def __init__(self, config: dict, system: str, tools: list[Tool], max_tokens: int):
        import anthropic

        self._client = anthropic.Anthropic(api_key=config["api_key"])
        self._model, self._system, self._max_tokens = config["model"], system, max_tokens
        self._tools = [{"name": t.name, "description": t.description, "input_schema": t.parameters} for t in tools]
        self._messages: list[dict] = []

    def send_user(self, text: str) -> None:
        self._messages.append({"role": "user", "content": text})

    def next(self) -> tuple[str, list[tuple[str, str, dict]]]:
        response = self._client.messages.create(
            model=self._model, max_tokens=self._max_tokens, system=self._system,
            messages=self._messages, tools=self._tools,
        )
        self._messages.append({"role": "assistant", "content": response.content})
        text = "".join(b.text for b in response.content if getattr(b, "type", "") == "text").strip()
        calls = [(b.id, b.name, dict(b.input or {})) for b in response.content if getattr(b, "type", "") == "tool_use"]
        return text, calls

    def send_results(self, results: list[tuple[str, str]]) -> None:
        self._messages.append({
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": cid, "content": out} for cid, out in results],
        })


class _OpenAISession:
    def __init__(self, config: dict, system: str, tools: list[Tool], max_tokens: int):
        from openai import OpenAI

        self._client = OpenAI(api_key=config["api_key"])
        self._model, self._max_tokens = config["model"], max_tokens
        self._tools = [
            {"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.parameters}}
            for t in tools
        ]
        self._messages: list[dict] = [{"role": "system", "content": system}]

    def send_user(self, text: str) -> None:
        self._messages.append({"role": "user", "content": text})

    def next(self) -> tuple[str, list[tuple[str, str, dict]]]:
        response = self._client.chat.completions.create(
            model=self._model, max_tokens=self._max_tokens, messages=self._messages, tools=self._tools,
        )
        message = response.choices[0].message
        entry: dict = {"role": "assistant", "content": message.content or ""}
        calls = []
        if message.tool_calls:
            entry["tool_calls"] = [
                {"id": c.id, "type": "function", "function": {"name": c.function.name, "arguments": c.function.arguments}}
                for c in message.tool_calls
            ]
            for c in message.tool_calls:
                try:
                    args = _json.loads(c.function.arguments or "{}")
                except _json.JSONDecodeError:
                    args = {}
                calls.append((c.id, c.function.name, args))
        self._messages.append(entry)
        return (message.content or "").strip(), calls

    def send_results(self, results: list[tuple[str, str]]) -> None:
        for cid, out in results:
            self._messages.append({"role": "tool", "tool_call_id": cid, "content": out})


class _GroqSession(_OpenAISession):
    """Identical wire protocol to `_OpenAISession` — Groq's Chat Completions + tool-calling
    API is OpenAI-compatible — just constructed against Groq's base_url."""

    def __init__(self, config: dict, system: str, tools: list[Tool], max_tokens: int):
        from openai import OpenAI

        self._client = OpenAI(api_key=config["api_key"], base_url=_GROQ_BASE_URL)
        self._model, self._max_tokens = config["model"], max_tokens
        self._tools = [
            {"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.parameters}}
            for t in tools
        ]
        self._messages: list[dict] = [{"role": "system", "content": system}]


class _GeminiSession:
    """Gemini function calling via google.generativeai. Gemini matches tool results to calls
    by function *name* (no call ids), so ids here are synthetic and mapped back to names."""

    def __init__(self, config: dict, system: str, tools: list[Tool], max_tokens: int):
        import google.generativeai as genai

        genai.configure(api_key=config["api_key"])
        declarations = []
        for t in tools:
            decl = {"name": t.name, "description": t.description}
            if t.parameters.get("properties"):  # Gemini rejects an object schema with no properties
                decl["parameters"] = _gemini_schema(t.parameters)
            declarations.append(decl)
        model = genai.GenerativeModel(
            config["model"], system_instruction=system, tools=[{"function_declarations": declarations}]
        )
        self._genai = genai
        self._chat = model.start_chat()
        self._max_tokens = max_tokens
        self._pending: str | object | None = None
        self._names: dict[str, str] = {}

    def send_user(self, text: str) -> None:
        self._pending = text

    def next(self) -> tuple[str, list[tuple[str, str, dict]]]:
        response = self._chat.send_message(self._pending, generation_config={"max_output_tokens": self._max_tokens})
        parts = response.candidates[0].content.parts if response.candidates else []
        calls, text_parts = [], []
        for i, part in enumerate(parts):
            fc = getattr(part, "function_call", None)
            if fc is not None and getattr(fc, "name", ""):
                call_id = f"{fc.name}-{len(self._names)}-{i}"
                self._names[call_id] = fc.name
                calls.append((call_id, fc.name, {k: v for k, v in dict(fc.args).items()}))
            elif getattr(part, "text", ""):
                text_parts.append(part.text)
        if not calls:
            finish_reason = response.candidates[0].finish_reason if response.candidates else None
            if finish_reason is not None and int(finish_reason) != 1:  # 1 == STOP; see _call_gemini
                raise RuntimeError(f"Gemini response did not finish cleanly (finish_reason={finish_reason}).")
        return "".join(text_parts).strip(), calls

    def send_results(self, results: list[tuple[str, str]]) -> None:
        proto = self._genai.protos
        self._pending = proto.Content(parts=[
            proto.Part(function_response=proto.FunctionResponse(name=self._names[cid], response={"result": out}))
            for cid, out in results
        ])


def _gemini_schema(schema: dict) -> dict:
    """JSON-schema -> Gemini schema (upper-case type names, recursively)."""
    out = {k: v for k, v in schema.items() if k in ("description", "required", "enum")}
    out["type"] = str(schema.get("type", "object")).upper()
    if "properties" in schema:
        out["properties"] = {k: _gemini_schema(v) for k, v in schema["properties"].items()}
    if "items" in schema:
        out["items"] = _gemini_schema(schema["items"])
    return out


_TOOL_SESSIONS = {"anthropic": _AnthropicSession, "openai": _OpenAISession, "gemini": _GeminiSession, "groq": _GroqSession}


def _execute_tool(tool: Tool | None, args: dict) -> tuple[str, bool]:
    if tool is None:
        return "error: unknown tool", False
    try:
        out = _json.dumps(tool.fn(**args), default=str)
        return out[:MAX_TOOL_RESULT_CHARS], True
    except Exception as exc:  # noqa: BLE001 — surfaced to the model so it can recover
        return f"error: {exc}", False


def _run_tool_loop_once(provider: str, api_key: str, model: str, system: str, user: str, tools: list[Tool], max_steps: int, max_tokens: int) -> ToolLoopResult:
    session_cls = _TOOL_SESSIONS.get(provider)
    if session_cls is None:
        # Only reachable for a future provider registered without a tool-use adapter — the
        # three built-in providers all have one. Treated as a normal failed attempt so
        # key_manager.execute() just moves on to the next candidate provider/key.
        raise RuntimeError(f"provider '{provider}' has no tool-use adapter")
    by_name = {t.name: t for t in tools}
    session = session_cls({"api_key": api_key, "model": model}, system, tools, max_tokens)
    session.send_user(user)
    trace: list[dict] = []
    for _ in range(max_steps):
        text, calls = session.next()
        if not calls:
            return ToolLoopResult(text=text, calls=trace)
        results = []
        for call_id, name, args in calls:
            output, ok = _execute_tool(by_name.get(name), args)
            trace.append({"tool": name, "args": args, "ok": ok})
            results.append((call_id, output))
        session.send_results(results)
    raise RuntimeError(f"tool loop exceeded {max_steps} steps without a final answer")


def run_tool_loop(
    system: str, user: str, tools: list[Tool], *, max_steps: int = 6, max_tokens: int = 1500
) -> ToolLoopResult:
    """Multi-step tool-use conversation with automatic key/provider failover. A failure
    (including a quota/rate-limit hit mid-conversation) restarts the whole tool loop fresh
    against the next available key rather than resuming — a tool-use conversation is tied
    to one provider's session/client, so it can't be handed off mid-flight."""
    try:
        return key_manager.execute(
            lambda provider, api_key, model: _run_tool_loop_once(provider, api_key, model, system, user, tools, max_steps, max_tokens)
        )
    except _km.NoProviderConfiguredError as exc:
        raise ToolLoopUnavailable(str(exc)) from exc


def parse_json_object(text: str) -> dict | None:
    """Best-effort: a JSON object from model text, tolerating ```json fences and prose."""
    if not text:
        return None
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        parsed = _json.loads(cleaned[start : end + 1])
    except _json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None

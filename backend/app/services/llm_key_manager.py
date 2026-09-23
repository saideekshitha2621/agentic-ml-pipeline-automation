"""Multi-key, multi-provider LLM key manager.

Generalizes "one API key per provider" into a pool of keys per provider, with automatic
failover on quota/rate-limit/auth errors, per-key health tracking (Healthy/Warning/
Exhausted), and provider-level fallback (e.g. every OpenAI key exhausted -> try Gemini;
every Gemini key exhausted -> a clear error naming what was tried).

Provider-agnostic by design: the caller supplies a `key_resolver` (how to find this
provider's keys — env vars, a secrets manager, whatever) and a `models` map. Adding a new
provider is just adding it to both and to the provider order — nothing in this module
hardcodes "openai"/"gemini".

Not thread-pool-safe against true concurrency (a `Lock` would be needed for that), but
FastAPI's request handlers here are single-threaded-per-event-loop-tick around this state,
matching the rest of the codebase's in-memory state (e.g. llm_service's own breaker).
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

logger = logging.getLogger("llm_key_manager")


class KeyStatus(str, Enum):
    HEALTHY = "healthy"
    WARNING = "warning"
    EXHAUSTED = "exhausted"


class FailureKind(str, Enum):
    RATE_LIMIT = "rate_limit"  # quota/429/rate-limit — temporary, key cools down
    AUTH = "auth"               # invalid/revoked key — long cooldown, unlikely to self-heal
    OTHER = "other"              # transient/network/parse/etc — counts toward warning/exhaustion


class NoProviderConfiguredError(RuntimeError):
    """No API key is set for any provider this manager knows about."""


class AllKeysExhaustedError(RuntimeError):
    """At least one provider is configured, but every key currently usable is exhausted."""


def mask_key(key: str) -> str:
    """Never log or store a usable key — keeps just enough to tell keys apart in logs."""
    if not key:
        return "<empty>"
    if len(key) <= 10:
        return "*" * len(key)
    return f"{key[:4]}...{key[-4:]}"


_RATE_LIMIT_MARKERS = ("429", "quota", "rate limit", "rate_limit", "resource_exhausted", "resource exhausted", "too many requests")
_AUTH_MARKERS = ("401", "403", "invalid api key", "invalid_api_key", "unauthorized", "authentication", "permission denied", "api key not valid", "incorrect api key")


def classify_error(exc: Exception) -> FailureKind:
    text = str(exc).lower()
    if any(marker in text for marker in _RATE_LIMIT_MARKERS):
        return FailureKind.RATE_LIMIT
    if any(marker in text for marker in _AUTH_MARKERS):
        return FailureKind.AUTH
    return FailureKind.OTHER


@dataclass
class KeyRecord:
    provider: str
    index: int
    key: str
    status: KeyStatus = KeyStatus.HEALTHY
    consecutive_failures: int = 0
    total_calls: int = 0
    total_errors: int = 0
    last_error: str | None = None
    last_used_at: float | None = None
    cooldown_until: float = 0.0  # time.monotonic() timestamp

    @property
    def label(self) -> str:
        return f"{self.provider}#{self.index} ({mask_key(self.key)})"

    def is_available(self, now: float) -> bool:
        return not (self.status == KeyStatus.EXHAUSTED and now < self.cooldown_until)

    def to_status_dict(self, now: float) -> dict:
        cooldown_remaining = round(self.cooldown_until - now, 1) if self.status == KeyStatus.EXHAUSTED else 0.0
        return {
            "provider": self.provider,
            "key_id": f"{self.provider}#{self.index}",
            "masked_key": mask_key(self.key),
            "status": self.status.value,
            "consecutive_failures": self.consecutive_failures,
            "total_calls": self.total_calls,
            "total_errors": self.total_errors,
            "last_error": self.last_error,
            "cooldown_remaining_seconds": max(0.0, cooldown_remaining),
        }


@dataclass
class KeyManagerConfig:
    """Every threshold is env-configurable so ops can tune failover behavior without a
    code change; defaults are conservative enough to work out of the box."""

    warning_after: int = 2           # consecutive non-fatal failures before a key is flagged Warning
    exhausted_after: int = 4         # consecutive non-fatal failures before a key is pulled from rotation
    rate_limit_cooldown: float = 300.0   # seconds a rate-limited/quota-exhausted key sits out
    auth_cooldown: float = 3600.0        # seconds an invalid-key sits out before being retried
    max_retries_per_request: int = 6     # hard cap on attempts across all keys/providers for one logical call

    @classmethod
    def from_env(cls) -> "KeyManagerConfig":
        def _int(name: str, default: int) -> int:
            try:
                return int(os.environ.get(name, default))
            except (TypeError, ValueError):
                return default

        def _float(name: str, default: float) -> float:
            try:
                return float(os.environ.get(name, default))
            except (TypeError, ValueError):
                return default

        return cls(
            warning_after=_int("LLM_KEY_WARNING_THRESHOLD", 2),
            exhausted_after=_int("LLM_KEY_EXHAUSTED_THRESHOLD", 4),
            rate_limit_cooldown=_float("LLM_KEY_COOLDOWN_SECONDS", 300.0),
            auth_cooldown=_float("LLM_KEY_AUTH_COOLDOWN_SECONDS", 3600.0),
            max_retries_per_request=_int("LLM_MAX_RETRIES_PER_REQUEST", 6),
        )


class KeyManager:
    def __init__(
        self,
        key_resolver: Callable[[], dict[str, list[str]]],
        models: dict[str, str],
        provider_order: list[str],
        config: KeyManagerConfig | None = None,
    ):
        self._key_resolver = key_resolver
        self._models = dict(models)
        self._provider_order = list(provider_order)
        self._config = config or KeyManagerConfig.from_env()
        self._records: dict[tuple[str, str], KeyRecord] = {}
        self._rr: dict[str, int] = {}
        self._static_pools: dict[str, list[str]] = {}  # test/manual registrations, merged over the resolver

    # --- pool resolution -----------------------------------------------------------------

    def _resolve_pools(self) -> dict[str, list[str]]:
        pools: dict[str, list[str]] = {p: list(keys) for p, keys in (self._key_resolver() or {}).items()}
        for provider, keys in self._static_pools.items():
            existing = pools.setdefault(provider, [])
            for key in keys:
                if key not in existing:
                    existing.append(key)
        return pools

    def _records_for(self, provider: str, pools: dict[str, list[str]]) -> list[KeyRecord]:
        records = []
        for i, key in enumerate(pools.get(provider) or [], start=1):
            record = self._records.get((provider, key))
            if record is None:
                record = KeyRecord(provider=provider, index=i, key=key)
                self._records[(provider, key)] = record
            else:
                record.index = i
            records.append(record)
        return records

    # --- introspection ---------------------------------------------------------------------

    def configured_providers(self) -> list[str]:
        pools = self._resolve_pools()
        return [p for p in self._provider_order if pools.get(p)]

    def is_configured(self) -> bool:
        return bool(self.configured_providers())

    def model_for(self, provider: str) -> str:
        return self._models.get(provider, "")

    def status_report(self) -> list[dict]:
        pools = self._resolve_pools()
        now = time.monotonic()
        return [
            record.to_status_dict(now)
            for provider in self._provider_order
            for record in self._records_for(provider, pools)
        ]

    # --- selection -------------------------------------------------------------------------

    def next_candidate(self, providers: list[str] | None = None) -> tuple[str, KeyRecord] | None:
        pools = self._resolve_pools()
        now = time.monotonic()
        for provider in (providers or self._provider_order):
            records = self._records_for(provider, pools)
            if not records:
                continue
            start = self._rr.get(provider, 0) % len(records)
            ordered = records[start:] + records[:start]  # round-robin so load spreads across healthy keys
            for record in ordered:
                if record.is_available(now):
                    return provider, record
        return None

    # --- health bookkeeping ------------------------------------------------------------------

    def _scrub(self, text: str) -> str:
        """Strips any known raw key out of a string before it's logged or stored — a
        provider's own error text sometimes echoes the key it rejected."""
        for record in self._records.values():
            if record.key and record.key in text:
                text = text.replace(record.key, mask_key(record.key))
        return text

    def record_success(self, record: KeyRecord) -> None:
        record.total_calls += 1
        record.last_used_at = time.time()
        if record.status != KeyStatus.HEALTHY:
            logger.info("llm_key_manager: %s recovered -> healthy", record.label)
        record.consecutive_failures = 0
        record.status = KeyStatus.HEALTHY
        record.cooldown_until = 0.0
        self._rr[record.provider] = self._rr.get(record.provider, 0) + 1

    def record_failure(self, record: KeyRecord, exc: Exception) -> FailureKind:
        kind = classify_error(exc)
        record.total_calls += 1
        record.total_errors += 1
        record.consecutive_failures += 1
        record.last_error = self._scrub(str(exc))[:300]
        record.last_used_at = time.time()
        now = time.monotonic()

        if kind is FailureKind.RATE_LIMIT:
            record.status = KeyStatus.EXHAUSTED
            record.cooldown_until = now + self._config.rate_limit_cooldown
            logger.warning("llm_key_manager: %s hit a quota/rate limit -> exhausted for %ss", record.label, self._config.rate_limit_cooldown)
        elif kind is FailureKind.AUTH:
            record.status = KeyStatus.EXHAUSTED
            record.cooldown_until = now + self._config.auth_cooldown
            logger.error("llm_key_manager: %s failed authentication -> exhausted for %ss (check this key)", record.label, self._config.auth_cooldown)
        elif record.consecutive_failures >= self._config.exhausted_after:
            record.status = KeyStatus.EXHAUSTED
            record.cooldown_until = now + self._config.rate_limit_cooldown
            logger.warning("llm_key_manager: %s failed %s times in a row -> exhausted", record.label, record.consecutive_failures)
        elif record.consecutive_failures >= self._config.warning_after:
            record.status = KeyStatus.WARNING
            logger.info("llm_key_manager: %s degraded -> warning (%s consecutive failures)", record.label, record.consecutive_failures)
        self._rr[record.provider] = self._rr.get(record.provider, 0) + 1
        return kind

    def reset(self) -> None:
        for record in self._records.values():
            record.status = KeyStatus.HEALTHY
            record.consecutive_failures = 0
            record.cooldown_until = 0.0
        self._rr.clear()

    # --- test/manual registration (also how a future provider can be wired without env vars) --

    def configure_test_provider(self, provider: str, keys: list[str], model: str = "test-model") -> None:
        self._static_pools[provider] = list(keys)
        self._models.setdefault(provider, model)
        if provider not in self._provider_order:
            self._provider_order.append(provider)

    def clear_test_provider(self, provider: str) -> None:
        self._static_pools.pop(provider, None)
        for key in [k for k in self._records if k[0] == provider]:
            del self._records[key]

    # --- execution -----------------------------------------------------------------------------

    def execute(self, attempt: Callable[[str, str, str], str]) -> str:
        """Runs `attempt(provider, api_key, model)` against candidate keys in priority
        order (provider order, then round-robin within a provider), recording health and
        retrying with a different key — same provider first, then the next provider in
        line — until one succeeds, the retry budget is spent, or every eligible key is
        exhausted. Never raises with a raw key in the message."""
        if not self.is_configured():
            raise NoProviderConfiguredError("No LLM provider is configured (set an API key for at least one provider).")

        attempts = 0
        last_provider: str | None = None
        last_error_text: str | None = None
        tried_providers: set[str] = set()
        while attempts < self._config.max_retries_per_request:
            picked = self.next_candidate()
            if picked is None:
                break
            provider, record = picked
            if last_provider is not None and provider != last_provider:
                logger.info("llm_key_manager: switching provider %s -> %s (previous provider's keys unavailable)", last_provider, provider)
            last_provider = provider
            tried_providers.add(provider)
            attempts += 1
            try:
                result = attempt(provider, record.key, self.model_for(provider))
            except Exception as exc:  # noqa: BLE001 — classified below, never re-raised verbatim with a key in it
                kind = self.record_failure(record, exc)
                last_error_text = record.last_error
                logger.info("llm_key_manager: %s attempt failed (%s) -> trying next available key", record.label, kind.value)
                continue
            self.record_success(record)
            return result

        configured = self.configured_providers()
        if not configured:
            raise NoProviderConfiguredError("No LLM provider is configured (set an API key for at least one provider).")
        detail = f" Last error: {last_error_text}" if last_error_text else ""
        raise AllKeysExhaustedError(
            f"All available API keys for provider(s) [{', '.join(sorted(tried_providers) or configured)}] are "
            f"currently exhausted or failing.{detail}"
        )

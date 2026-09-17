"""Resolves which LLM provider/key/model is active right now: the runtime-editable
`AppSettings` DB row (set via the Settings page) takes precedence over the deployment-time
environment variables, so a user can override without restarting the server. If neither is
set, callers get `None` and fall back to template-only behavior — see `llm_service.py`.

Where API keys can be configured (documented here since this is the one place that reads
both sources):
  - Environment variables (deployment default, read in `app/core/config.py`):
      ANTHROPIC_API_KEY, GEMINI_API_KEY, OPENAI_API_KEY
  - Settings page -> `POST /api/v1/settings/llm` (see `routers/settings.py`) -> persisted in
    the `app_settings` table's `llm_provider`/`llm_api_key`/`llm_model` columns.
"""
from __future__ import annotations

from sqlalchemy.orm import Session

from app.core.config import ANTHROPIC_API_KEY, ANTHROPIC_MODEL, GEMINI_API_KEY, OPENAI_API_KEY
from app.db.models import AppSettings as AppSettingsORM

DEFAULT_MODELS = {
    "anthropic": ANTHROPIC_MODEL,
    "gemini": "gemini-2.0-flash",
    "openai": "gpt-4o-mini",
}

_ENV_PROVIDERS = [
    ("anthropic", ANTHROPIC_API_KEY),
    ("gemini", GEMINI_API_KEY),
    ("openai", OPENAI_API_KEY),
]


def _get_row(db: Session) -> AppSettingsORM | None:
    return db.get(AppSettingsORM, "default")


def get_active_config(db: Session) -> dict | None:
    """Returns {"provider", "api_key", "model", "source"} or None if nothing is configured
    anywhere. `source` is "database" or "environment", surfaced by GET /settings/llm."""
    row = _get_row(db)
    if row and row.llm_provider and row.llm_api_key:
        return {
            "provider": row.llm_provider,
            "api_key": row.llm_api_key,
            "model": row.llm_model or DEFAULT_MODELS.get(row.llm_provider),
            "source": "database",
        }
    for provider, api_key in _ENV_PROVIDERS:
        if api_key:
            return {"provider": provider, "api_key": api_key, "model": DEFAULT_MODELS[provider], "source": "environment"}
    return None


def save_config(db: Session, provider: str, api_key: str, model: str | None = None) -> AppSettingsORM:
    row = _get_row(db)
    if row is None:
        row = AppSettingsORM(id="default")
        db.add(row)
    row.llm_provider = provider
    row.llm_api_key = api_key
    row.llm_model = model
    db.commit()
    db.refresh(row)
    return row


def clear_config(db: Session) -> None:
    row = _get_row(db)
    if row:
        row.llm_provider = None
        row.llm_api_key = None
        row.llm_model = None
        db.commit()


def mask(api_key: str | None) -> str | None:
    if not api_key:
        return None
    if len(api_key) <= 4:
        return "*" * len(api_key)
    return f"{'*' * (len(api_key) - 4)}{api_key[-4:]}"


def status(db: Session) -> dict:
    config = get_active_config(db)
    if not config:
        return {"provider": None, "model": None, "masked_api_key": None, "configured": False, "source": "none"}
    return {
        "provider": config["provider"],
        "model": config["model"],
        "masked_api_key": mask(config["api_key"]),
        "configured": True,
        "source": config["source"],
    }

"""Settings page backend — lets a user pick an LLM provider, paste an API key, test it,
and save it, all at runtime (no server restart / env var edit needed). See
`services/llm_config_service.py` for exactly how this interacts with the env-var fallback.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.services import llm_config_service, llm_service

router = APIRouter(prefix="/api/v1/settings", tags=["settings"])

_VALID_PROVIDERS = {"anthropic", "gemini", "openai"}


class LLMSettingsRequest(BaseModel):
    provider: str
    api_key: str
    model: str | None = None


@router.get("/llm")
def get_llm_settings(db: Session = Depends(get_db)):
    return llm_config_service.status(db)


@router.post("/llm")
def save_llm_settings(body: LLMSettingsRequest, db: Session = Depends(get_db)):
    if body.provider not in _VALID_PROVIDERS:
        return {"error": f"Unknown provider '{body.provider}'. Choose one of {sorted(_VALID_PROVIDERS)}."}
    llm_config_service.save_config(db, body.provider, body.api_key, body.model)
    return llm_config_service.status(db)


@router.post("/llm/test")
def test_llm_settings(body: LLMSettingsRequest, db: Session = Depends(get_db)):
    if body.provider not in _VALID_PROVIDERS:
        return {"success": False, "message": f"Unknown provider '{body.provider}'. Choose one of {sorted(_VALID_PROVIDERS)}."}
    model = body.model or llm_config_service.DEFAULT_MODELS[body.provider]
    try:
        llm_service.call(
            body.provider, body.api_key, model,
            system="Reply with exactly one word: OK.",
            messages=[{"role": "user", "content": "Connection test."}],
            max_tokens=10,
        )
        return {"success": True, "message": f"Connected to {body.provider} successfully."}
    except Exception as exc:  # noqa: BLE001
        return {"success": False, "message": f"Connection failed: {exc}"}


@router.delete("/llm")
def clear_llm_settings(db: Session = Depends(get_db)):
    llm_config_service.clear_config(db)
    return llm_config_service.status(db)

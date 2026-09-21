"""API-key authentication (Phase 5).

Set `API_KEYS` (comma-separated) in the backend environment to protect every `/api/v1/*` route
except the health check. With it unset the API is open, exactly as before — so local development
needs no change. Clients send the key as an `X-API-Key` header; a `?api_key=` query parameter is
also accepted because report downloads are plain browser links that cannot set headers (prefer
the header everywhere else — query strings can end up in server logs).

Scope note: this is shared-secret access control, not user accounts. It does not distinguish
users, so `reviewed_by` on approvals is still self-reported. Roles/SSO are a separate piece of work.
"""
from __future__ import annotations

import hmac
import os

from fastapi import Header, HTTPException, Query


def _load_keys() -> set[str]:
    return {k.strip() for k in os.environ.get("API_KEYS", "").split(",") if k.strip()}


API_KEYS: set[str] = _load_keys()


def auth_enabled() -> bool:
    return bool(API_KEYS)


def require_api_key(x_api_key: str | None = Header(default=None), api_key: str | None = Query(default=None)) -> None:
    if not API_KEYS:
        return
    supplied = x_api_key or api_key
    if not supplied or not any(hmac.compare_digest(supplied, k) for k in API_KEYS):
        raise HTTPException(status_code=401, detail="Missing or invalid API key.")

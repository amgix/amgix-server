"""API key authentication for the Amgix HTTP API (Qdrant-style keys and headers)."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from enum import IntEnum
from typing import Mapping, Optional

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

class KeyRole(IntEnum):
    SEARCH = 1
    READ = 2
    ADMIN = 3


_PUBLIC_GET_PATHS = frozenset(
    {
        "/v1/version",
        "/v1/health/check",
        "/v1/health/ready",
    }
)

_ENV_KEYS: tuple[tuple[str, KeyRole], ...] = (
    ("AMGIX_SEARCH_KEY", KeyRole.SEARCH),
    ("AMGIX_ALT_SEARCH_KEY", KeyRole.SEARCH),
    ("AMGIX_READ_KEY", KeyRole.READ),
    ("AMGIX_ALT_READ_KEY", KeyRole.READ),
    ("AMGIX_ALT_API_KEY", KeyRole.ADMIN),
    ("AMGIX_API_KEY", KeyRole.ADMIN),
)

_ROLE_LABELS = {
    KeyRole.SEARCH: "search",
    KeyRole.READ: "read",
    KeyRole.ADMIN: "admin",
}


@dataclass(frozen=True)
class ApiKeyAuthConfig:
    need_auth: bool
    key_roles: Mapping[str, KeyRole]


def load_api_key_auth_config() -> ApiKeyAuthConfig:
    """Load API keys from the environment once at process startup."""
    key_roles: dict[str, KeyRole] = {}
    for env_name, role in _ENV_KEYS:
        raw = os.getenv(env_name)
        if not raw:
            continue
        key = raw.strip()
        if not key:
            continue
        existing = key_roles.get(key)
        if existing is None or role > existing:
            key_roles[key] = role
    return ApiKeyAuthConfig(need_auth=bool(key_roles), key_roles=key_roles)


API_KEY_AUTH = load_api_key_auth_config()


def log_auth_startup(logger: logging.Logger) -> None:
    if not API_KEY_AUTH.need_auth:
        logger.info("API key auth disabled (no AMGIX_*_KEY env vars set)")
        return
    counts = {role: 0 for role in KeyRole}
    for role in API_KEY_AUTH.key_roles.values():
        counts[role] += 1
    logger.info(
        "API key auth enabled (%d key(s): %d admin, %d read, %d search)",
        len(API_KEY_AUTH.key_roles),
        counts[KeyRole.ADMIN],
        counts[KeyRole.READ],
        counts[KeyRole.SEARCH],
    )


def extract_api_key(request: Request) -> Optional[str]:
    api_key_header = request.headers.get("api-key")
    if api_key_header is not None:
        stripped = api_key_header.strip()
        if stripped:
            return stripped
        return None

    authorization = request.headers.get("authorization")
    if authorization is None:
        return None

    if authorization.startswith("Bearer "):
        token = authorization[7:].strip()
        return token or None

    stripped = authorization.strip()
    return stripped or None


def _role_label(role: KeyRole) -> str:
    return _ROLE_LABELS[role]


def _role_phrase(role: KeyRole) -> str:
    label = _role_label(role)
    article = "an" if label[0] in "aeiou" else "a"
    return f"{article} {label}"


def _authentication_failed_detail(needed: KeyRole) -> str:
    return (
        f"Authentication failed. This endpoint requires at least {_role_phrase(needed)} API key. "
        f"Pass it via the `api-key` header or `Authorization: Bearer <key>`."
    )


def _auth_error_response(detail: str) -> JSONResponse:
    return JSONResponse(
        status_code=401,
        content={"detail": detail},
        headers={"WWW-Authenticate": 'Bearer realm="amgix", charset="UTF-8"'},
    )


def required_role(method: str, path: str) -> Optional[KeyRole]:
    """Return the minimum key role for a /v1 route, or None if public / out of scope."""
    if not path.startswith("/v1"):
        return None

    if method == "GET" and path in _PUBLIC_GET_PATHS:
        return None

    parts = [segment for segment in path.split("/") if segment]
    if len(parts) < 2 or parts[0] != "v1":
        return None

    if method == "POST" and _is_search_path(parts):
        return KeyRole.SEARCH

    if _is_admin_route(method, parts):
        return KeyRole.ADMIN

    # All other /v1 routes are read-only (including POST fetch).
    return KeyRole.READ


def _is_search_path(parts: list[str]) -> bool:
    return (
        len(parts) >= 4
        and parts[1] == "collections"
        and parts[-1] == "search"
    )


def _is_admin_route(method: str, parts: list[str]) -> bool:
    if parts[1] != "collections":
        return False

    if method == "DELETE":
        if len(parts) == 3:
            return True
        if "documents" in parts:
            return True
        return parts[-1] == "queue"

    if method != "POST":
        return False

    if len(parts) == 3:
        return True
    if parts[-1] == "empty":
        return True
    if parts[-1] == "bulk":
        return True
    if len(parts) >= 5 and parts[-2] == "documents" and parts[-1] == "sync":
        return True
    if len(parts) == 4 and parts[-1] == "documents":
        return True
    return False


class ApiKeyAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        if not API_KEY_AUTH.need_auth:
            return await call_next(request)

        needed = required_role(request.method, request.url.path)
        if needed is None:
            return await call_next(request)

        presented = extract_api_key(request)
        have = API_KEY_AUTH.key_roles.get(presented) if presented else None
        if have is None or have < needed:
            return _auth_error_response(_authentication_failed_detail(needed))

        return await call_next(request)

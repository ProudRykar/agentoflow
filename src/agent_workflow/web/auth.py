from __future__ import annotations

import hmac
import os
import secrets
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from fastapi import HTTPException, Request, WebSocket
from fastapi.responses import JSONResponse


TOKEN_ENV = "AGENTOFLOW_API_TOKEN"

TOKEN_HEADER = "x-agentoflow-token"

AUTH_QUERY = "token"


@dataclass(slots=True, frozen=True)
class AuthSettings:
    """Bearer-token gate for the control plane.

    Disabled unless a token is configured, so local development
    needs no setup. The token is read from the environment and
    never stored in the repository.
    """

    token: str = ""

    @property
    def enabled(self) -> bool:
        return bool(self.token)

    @classmethod
    def from_env(cls) -> "AuthSettings":
        token = os.getenv(TOKEN_ENV, "").strip()

        return cls(token=token)

    def matches(self, candidate: str) -> bool:
        if not self.enabled:
            return True

        if not candidate:
            return False

        return hmac.compare_digest(
            candidate,
            self.token,
        )


def _presented_token(request: Request) -> str:
    header = request.headers.get(TOKEN_HEADER)

    if header:
        return header.strip()

    authorization = request.headers.get("authorization", "")

    if authorization.lower().startswith("bearer "):
        return authorization[7:].strip()

    return ""


def require_token(
    settings: AuthSettings,
) -> Callable[[Request], Awaitable[None]]:
    """Dependency guarding every REST route."""

    if not settings.enabled:
        async def allow(_: Request) -> None:
            return None

        return allow

    async def guard(request: Request) -> None:
        if settings.matches(_presented_token(request)):
            return

        raise HTTPException(
            status_code=401,
            detail="Missing or invalid API token",
        )

    return guard


async def guard_websocket(
    websocket: WebSocket,
    settings: AuthSettings,
) -> bool:
    """Authorise a WebSocket handshake.

    Browsers cannot set headers on a WebSocket, so the token may
    also arrive as a query parameter.
    """

    if not settings.enabled:
        return True

    candidate = websocket.query_params.get(
        AUTH_QUERY,
        "",
    ) or websocket.headers.get(TOKEN_HEADER, "")

    if settings.matches(candidate.strip()):
        return True

    await websocket.close(
        code=4001,
        reason="Unauthorized",
    )

    return False


def generate_token() -> str:
    """Helper for operators: ``python -m agent_workflow.web.auth``."""

    return secrets.token_urlsafe(32)


def main() -> None:
    print(generate_token())


if __name__ == "__main__":
    main()

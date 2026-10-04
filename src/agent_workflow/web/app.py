from __future__ import annotations

import os
import secrets
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from agent_workflow.core.application.session import SessionManager
from agent_workflow.web.api import (
    mcp,
    plugins,
    sessions,
    settings,
    skills,
    skills_crud,
    system,
    tools,
)
from agent_workflow.web.auth import (
    AuthSettings,
    require_token,
)
from agent_workflow.web.websocket import agent as ws_agent


API_PREFIX = "/api"


@asynccontextmanager
async def lifespan(app: FastAPI):
    if getattr(app.state, "session_manager", None) is None:
        app.state.session_manager = SessionManager()

    try:
        yield
    finally:
        manager = getattr(
            app.state,
            "session_manager",
            None,
        )

        if manager is not None:
            await manager.close_all()


def cors_origins() -> list[str]:
    raw = os.getenv("AGENTOFLOW_CORS_ORIGINS")

    if not raw:
        return [
            "http://localhost:5173",
            "http://127.0.0.1:5173",
        ]

    return [
        origin.strip()
        for origin in raw.split(",")
        if origin.strip()
    ]


def create_app(
    session_manager: SessionManager | None = None,
) -> FastAPI:
    """Build the ASGI application.

    ``session_manager`` is injectable so tests and embedders can
    supply an isolated manager instead of the process-wide one.
    """

    app = FastAPI(
        title="Agentoflow",
        description="LLM agent workflow runtime",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.state.auth = AuthSettings.from_env()

    if session_manager is not None:
        app.state.session_manager = session_manager

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins(),
        allow_credentials=True,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["*"],
    )

    guard = Depends(require_token(app.state.auth))

    app.include_router(
        sessions.router,
        prefix=f"{API_PREFIX}/sessions",
        tags=["sessions"],
        dependencies=[guard],
    )
    app.include_router(
        plugins.router,
        prefix=f"{API_PREFIX}/plugins",
        tags=["plugins"],
        dependencies=[guard],
    )
    app.include_router(
        system.router,
        prefix=f"{API_PREFIX}/system",
        tags=["system"],
        dependencies=[guard],
    )
    app.include_router(
        tools.router,
        prefix=f"{API_PREFIX}/tools",
        tags=["tools"],
        dependencies=[guard],
    )
    # Registered before the session-scoped skills router so the
    # literal /skills/library paths are not captured as a
    # session id.
    app.include_router(
        skills_crud.router,
        prefix=f"{API_PREFIX}/skills",
        tags=["skills"],
        dependencies=[guard],
    )
    app.include_router(
        skills.router,
        prefix=f"{API_PREFIX}/skills",
        tags=["skills"],
        dependencies=[guard],
    )
    app.include_router(
        settings.router,
        prefix=f"{API_PREFIX}/settings",
        tags=["settings"],
        dependencies=[guard],
    )
    app.include_router(
        mcp.router,
        prefix=f"{API_PREFIX}/mcp",
        tags=["mcp"],
        dependencies=[guard],
    )

    app.include_router(
        ws_agent.router,
        prefix="/ws",
        tags=["websocket"],
    )

    _mount_frontend(app)

    return app


def _mount_frontend(app: FastAPI) -> None:
    """Serve the built SPA when it exists.

    Only a production build directory is served; in development
    Vite serves the UI and proxies /api and /ws here.
    """

    dist = Path(__file__).resolve().parents[3] / "frontend" / "dist"

    if not dist.is_dir():
        return

    from fastapi.staticfiles import StaticFiles

    app.mount(
        "/",
        StaticFiles(
            directory=str(dist),
            html=True,
        ),
        name="frontend",
    )


def main() -> None:
    import uvicorn

    host = os.getenv("AGENTOFLOW_HOST", "127.0.0.1")
    port = int(os.getenv("AGENTOFLOW_PORT", "8000"))

    auth = AuthSettings.from_env()

    if auth.enabled:
        print(
            "[agentoflow] authentication ENABLED "
            "(bearer token required)"
        )
    else:
        print(
            "[agentoflow] authentication DISABLED - "
            "bind to 127.0.0.1 or set AGENTOFLOW_API_TOKEN "
            "before exposing this service"
        )

    if host not in ("127.0.0.1", "localhost", "::1"):
        print(
            f"[agentoflow] WARNING: listening on {host} "
            "is reachable from the network"
        )

    uvicorn.run(
        "agent_workflow.web.app:app",
        host=host,
        port=port,
        reload=False,
    )


app = create_app()

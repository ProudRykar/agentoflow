from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from agent_workflow.core.infrastructure.config import (
    ConfigLoader,
    MCPConfig,
    MCPServerConfigEntry,
)
from agent_workflow.web.api.settings import service_of
from agent_workflow.web.api.tools import _session

from agent_workflow.web.schemas import (
    MCPServerInfo,
    MCPToolInfo,
    MCPResponse,
)
from agent_workflow.core.application.settings_service import CONFIG_FILE

router = APIRouter()

@router.get("/{session_id}", response_model=MCPResponse)
async def list_mcp(
    session_id: str,
    request: Request,
) -> MCPResponse:
    """Connected MCP servers and the tools they contribute."""

    session = await _session(request, session_id)

    manager = session.runtime.mcp_manager

    if manager is None or not manager.enabled:
        return MCPResponse(enabled=False)

    servers: list[MCPServerInfo] = []

    for server in manager.servers():
        tools = [
            MCPToolInfo(
                qualified_name=info.qualified_name,
                remote_name=info.remote_name,
                description=info.description,
                parameters=list(info.parameters),
                required=list(info.required),
                requires_approval=info.requires_approval,
                permissions=sorted(info.permissions),
            )
            for info in manager.tools()
            if info.server == server.name
        ]

        servers.append(
            MCPServerInfo(
                name=server.name,
                transport=server.transport,
                command=server.command,
                url=server.url,
                args=list(server.args),
                enabled=server.enabled,
                state=server.state.value,
                error=server.error,
                server_version=server.server_version,
                protocol_version=server.protocol_version,
                instructions=server.instructions,
                warnings=list(server.warnings),
                tools=tools,
                connected_at=server.connected_at,
                startup_seconds=round(
                    server.duration_seconds,
                    3,
                ),
            )
        )

    return MCPResponse(
        enabled=True,
        servers=servers,
        permissions=sorted(manager.granted_permissions()),
        total_tools=len(manager.tools()),
        connected_servers=sum(
            1
            for server in servers
            if server.state == "connected"
        ),
    )


# ======================================================================
# Runtime control
# ======================================================================


class MCPActionResponse(BaseModel):
    session_id: str
    server: str
    action: str
    state: str
    total_tools: int


def _manager_or_400(session):
    manager = session.runtime.mcp_manager

    if manager is None or not manager.enabled:
        raise HTTPException(
            status_code=409,
            detail="MCP is disabled in the configuration",
        )

    return manager


def _require_known(manager, name: str) -> None:
    if not manager.known(name):
        raise HTTPException(
            status_code=404,
            detail=f"Unknown MCP server: {name}",
        )


# ======================================================================
# Server registry
# ======================================================================


class MCPServerWrite(BaseModel):
    """A server definition created or edited from the UI."""

    name: str
    transport: str = "stdio"
    command: str = ""
    args: list[str] = []
    url: str = ""
    # None means "keep whatever is stored". The browser never sees
    # these values, so an omitted field must not be read as a
    # request to wipe them; send {} to clear deliberately.
    env: dict[str, str] | None = None
    headers: dict[str, str] | None = None
    enabled: bool = True


class MCPEnabledWrite(BaseModel):
    enabled: bool


def _config_path(request: Request) -> Path:
    return service_of(request).config_path


def _persist(
    request: Request,
    mcp_table: dict,
) -> MCPConfig:
    """Write the ``[mcp]`` table back and re-parse it.

    The loader is the single source of truth for validation, so the
    reloaded config is exactly what a fresh session would build.
    """

    service = service_of(request)
    document = service.read(CONFIG_FILE)

    data = dict(document.data)
    data["mcp"] = mcp_table

    service.write(CONFIG_FILE, data)

    return ConfigLoader().load(_config_path(request)).mcp


def _mcp_table(
    config: MCPConfig | None,
    *,
    enabled: bool,
) -> dict:
    """Render the current config back into a TOML-ready table."""

    table: dict = {"enabled": enabled}

    if config is None:
        return table

    table["require_approval"] = config.require_approval

    if config.default_permissions:
        table["default_permissions"] = list(
            config.default_permissions
        )

    servers: dict[str, dict] = {}

    for entry in config.servers:
        item: dict = {"transport": entry.transport}

        if entry.command:
            item["command"] = entry.command

        if entry.url:
            item["url"] = entry.url

        if entry.args:
            item["args"] = list(entry.args)

        if entry.env:
            item["env"] = dict(entry.env)

        if entry.headers:
            item["headers"] = dict(entry.headers)

        if not entry.enabled:
            item["enabled"] = False

        servers[entry.name] = item

    if servers:
        table["servers"] = servers

    return table


def _current_mcp(request: Request) -> MCPConfig | None:
    path = _config_path(request)

    if not path.exists():
        return None

    return ConfigLoader().load(path).mcp


def current_entry(
    config: MCPConfig | None,
    name: str,
) -> MCPServerConfigEntry:
    """The stored definition for ``name``, or an empty one."""

    empty = MCPServerConfigEntry(name=name)

    if config is None:
        return empty

    for entry in config.servers:
        if entry.name == name:
            return entry

    return empty


def _validate(payload: MCPServerWrite) -> None:
    name = payload.name.strip()

    if not name:
        raise HTTPException(
            status_code=422,
            detail="Server name is required",
        )

    if any(character in name for character in " \t\n"):
        raise HTTPException(
            status_code=422,
            detail="Server name cannot contain spaces",
        )

    if payload.transport not in ("stdio", "http"):
        raise HTTPException(
            status_code=422,
            detail="Transport must be stdio or http",
        )

    if payload.transport == "stdio" and not payload.command.strip():
        raise HTTPException(
            status_code=422,
            detail="A stdio server needs a command",
        )

    if payload.transport == "http" and not payload.url.strip():
        raise HTTPException(
            status_code=422,
            detail="An http server needs a URL",
        )


@router.post("/{session_id}/servers", response_model=MCPResponse)
async def upsert_server(
    session_id: str,
    payload: MCPServerWrite,
    request: Request,
) -> MCPResponse:
    """Create or update a server and connect it in this session.

    Persisting through the settings service keeps validation, secret
    redaction and the config backup in one place, so the UI never
    has to hand-write TOML to add a server.
    """

    _validate(payload)

    session = await _session(request, session_id)

    current = _current_mcp(request)
    table = _mcp_table(current, enabled=True)

    servers = dict(table.get("servers") or {})
    previous = current_entry(current, payload.name.strip())

    item: dict = {"transport": payload.transport}

    if payload.command.strip():
        item["command"] = payload.command.strip()

    if payload.url.strip():
        item["url"] = payload.url.strip()

    if payload.args:
        item["args"] = list(payload.args)

    env = payload.env if payload.env is not None else previous.env
    headers = (
        payload.headers
        if payload.headers is not None
        else previous.headers
    )

    if env:
        item["env"] = dict(env)

    if headers:
        item["headers"] = dict(headers)

    if not payload.enabled:
        item["enabled"] = False

    servers[payload.name.strip()] = item
    table["servers"] = servers

    config = _persist(request, table)

    manager = session.runtime.mcp_manager

    if manager is not None:
        await manager.apply_config(config)

    session.refresh_permissions()

    return await list_mcp(session_id, request)


@router.delete("/{session_id}/servers/{name}")
async def delete_server(
    session_id: str,
    name: str,
    request: Request,
) -> MCPResponse:
    """Remove a server from the config and disconnect it."""

    session = await _session(request, session_id)

    current = _current_mcp(request)

    if current is None:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown MCP server: {name}",
        )

    table = _mcp_table(current, enabled=current.enabled)
    servers = dict(table.get("servers") or {})

    if name not in servers:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown MCP server: {name}",
        )

    del servers[name]

    if servers:
        table["servers"] = servers
    else:
        table.pop("servers", None)

    config = _persist(request, table)

    manager = session.runtime.mcp_manager

    if manager is not None:
        await manager.apply_config(config)

    session.refresh_permissions()

    return await list_mcp(session_id, request)


@router.post("/{session_id}/enabled", response_model=MCPResponse)
async def set_enabled(
    session_id: str,
    payload: MCPEnabledWrite,
    request: Request,
) -> MCPResponse:
    """Turn MCP on or off for every future session."""

    session = await _session(request, session_id)

    current = _current_mcp(request)
    table = _mcp_table(current, enabled=payload.enabled)

    if not payload.enabled:
        table.pop("servers", None)

    config = _persist(request, table)

    manager = session.runtime.mcp_manager

    if manager is not None:
        await manager.apply_config(config)

    session.refresh_permissions()

    return await list_mcp(session_id, request)


@router.post(
    "/{session_id}/servers/{name}/{action}",
    response_model=MCPActionResponse,
)
async def server_action(
    session_id: str,
    name: str,
    action: str,
    request: Request,
) -> MCPActionResponse:
    """Connect, disconnect or reload one MCP server.

    Tools contributed by a server are added to or removed from the
    live registry, and the session's permissions are re-derived so
    the agent never keeps a grant the runtime no longer offers.
    """

    if action not in ("connect", "disconnect", "reload"):
        raise HTTPException(
            status_code=404,
            detail="Unknown action",
        )

    session = await _session(request, session_id)

    manager = _manager_or_400(session)

    _require_known(manager, name)

    if action == "disconnect":
        await manager.disconnect(name)
    else:
        # A server that refuses to start does not raise: the
        # manager records the failure and the response reports the
        # resulting state, so the UI can show the server's own
        # diagnostics instead of a generic gateway error.
        await manager.connect(name)

    # Reconnecting re-registers the server's tools, which would
    # resurrect anything the operator switched off. Re-applying the
    # disabled set here keeps a reconnect from undoing that choice.
    toggle = getattr(session.runtime, "tool_toggle", None)

    if toggle is not None:
        toggle.observe()
        toggle.apply()

    session.refresh_permissions()

    info = next(
        (
            server
            for server in manager.servers()
            if server.name == name
        ),
        None,
    )

    return MCPActionResponse(
        session_id=session_id,
        server=name,
        action=action,
        state=info.state.value if info is not None else "unknown",
        total_tools=len(manager.tools()),
    )


@router.post("/{session_id}/reload")
async def reload_all(
    session_id: str,
    request: Request,
) -> dict:
    """Reconnect every server, picking up configuration changes."""

    session = await _session(request, session_id)

    manager = _manager_or_400(session)

    for name in manager.server_names():
        try:
            await manager.reload(name)
        except Exception:
            # Failures are already recorded per server.
            pass

    session.refresh_permissions()

    return {
        "session_id": session_id,
        "reloaded": len(manager.server_names()),
        "total_tools": len(manager.tools()),
    }

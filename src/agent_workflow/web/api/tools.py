from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request

import logging

from agent_workflow.core.application.session import SessionManager
from agent_workflow.core.application.settings_service import (
    CONFIG_FILE,
    SettingsService,
)
from agent_workflow.core.infrastructure.paths import (
    AgentWorkflowPaths,
)
from agent_workflow.core.entities.models.tool_definition_builder import (
    ToolDefinitionBuilder,
)
from agent_workflow.web.schemas import (
    GlobalStatsResponse,
    GlobalToolUsage,
    ToolInfo,
    ToolParameter,
    ToolsResponse,
    ToolsSummary,
    ToolToggleResponse,
    ToolToggleWrite,
    ToolUsageStats,
)


logger = logging.getLogger(__name__)


router = APIRouter()


def manager_of(request: Request) -> SessionManager:
    """Resolve the session manager for this app.

    An explicit ``is None`` check matters: SessionManager defines
    ``__len__``, so an empty manager is falsy and a truthiness
    fallback would silently build a second manager bound to the
    real home directory.
    """

    manager = getattr(
        request.app.state,
        "session_manager",
        None,
    )

    if manager is None:
        manager = SessionManager()
        request.app.state.session_manager = manager

    return manager


async def _session(request: Request, session_id: str):
    session = await manager_of(request).get_session(session_id)

    if session is None:
        raise HTTPException(
            status_code=404,
            detail="Session not found",
        )

    return session


# ======================================================================
# Tool inventory
# ======================================================================


@router.get("/{session_id}", response_model=ToolsResponse)
async def list_tools(
    session_id: str,
    request: Request,
) -> ToolsResponse:
    """Every tool the agent can call, with live usage statistics."""

    session = await _session(request, session_id)

    registry = session.agent.registry
    plugins = session.runtime.plugin_manager
    mcp = session.runtime.mcp_manager

    plugin_of: dict[str, str] = {}

    if plugins is not None:
        for plugin in plugins.registry.list():
            context = plugins.get_plugin_context(plugin.name)

            if context is None:
                continue

            for tool_name in context.registered_tools:
                plugin_of[tool_name] = plugin.name

    builder = ToolDefinitionBuilder()

    stats = session.stats
    permissions = (
        session.context.permissions
        | session.context.approved_permissions
    )

    toggle = session.runtime.tool_toggle

    if toggle is not None:
        toggle.observe()

    disabled = (
        toggle.disabled_names() if toggle is not None else ()
    )

    tools: list[ToolInfo] = []

    for tool in registry.all():
        definition = builder.build(tool)

        usage = stats.get(tool.name)

        missing = sorted(
            tool.policy.permissions - permissions
        )

        tools.append(
            ToolInfo(
                name=tool.name,
                description=definition.description,
                source=(
                    "mcp"
                    if mcp is not None and mcp.owns(tool.name)
                    else (
                        "plugin"
                        if tool.name in plugin_of
                        else "builtin"
                    )
                ),
                permissions=sorted(tool.policy.permissions),
                missing_permissions=missing,
                requires_approval=tool.policy.requires_approval,
                timeout=tool.policy.timeout,
                max_output_size=tool.policy.max_output_size,
                parameters=_parameters(definition.input_schema),
                required_parameters=_required(
                    definition.input_schema
                ),
                mcp_server=plugin_of.get(tool.name),
                stats=_stats(usage),
            )
        )

    tools.sort(key=lambda info: info.name)

    if toggle is not None:
        # A disabled tool is absent from the registry, so it has to
        # be listed from what the session still retains. Without
        # this there would be no way to switch one back on from the
        # UI.
        visible = {tool.name for tool in registry.all()}

        for name in disabled:
            if name in visible:
                continue

            hidden = toggle.known_tools_by_name().get(name)

            if hidden is None:
                continue

            definition = builder.build(hidden)

            tools.append(
                ToolInfo(
                    name=hidden.name,
                    description=definition.description,
                    source=_source_of(
                        hidden.name,
                        mcp,
                        plugin_of,
                    ),
                    enabled=False,
                    permissions=sorted(
                        hidden.policy.permissions
                    ),
                    missing_permissions=sorted(
                        hidden.policy.permissions - permissions
                    ),
                    requires_approval=(
                        hidden.policy.requires_approval
                    ),
                    timeout=hidden.policy.timeout,
                    max_output_size=(
                        hidden.policy.max_output_size
                    ),
                    parameters=_parameters(
                        definition.input_schema
                    ),
                    required_parameters=_required(
                        definition.input_schema
                    ),
                    mcp_server=plugin_of.get(hidden.name),
                )
            )

        tools.sort(key=lambda item: item.name)

    return ToolsResponse(
        tools=tools,
        # Only what the model may actually call. A disabled tool is
        # listed for the operator but is not registered.
        disabled=list(disabled),
        summary=ToolsSummary(
            registered_tools=sum(
                1 for item in tools if item.enabled
            ),
            **stats.summary(),  # type: ignore[arg-type]
        ),
    )


def _source_of(
    name: str,
    mcp: Any,
    plugin_of: dict[str, str],
) -> str:
    if mcp is not None and mcp.owns(name):
        return "mcp"

    if name in plugin_of:
        return "plugin"

    return "builtin"


def _parameters(
    schema: dict[str, Any],
) -> list[ToolParameter]:
    properties = schema.get("properties")

    if not isinstance(properties, dict):
        return []

    required = set(_required(schema))

    result: list[ToolParameter] = []

    for name, definition in properties.items():
        definition = (
            definition if isinstance(definition, dict) else {}
        )

        result.append(
            ToolParameter(
                name=str(name),
                type=str(
                    definition.get("type", "any") or "any"
                ),
                required=str(name) in required,
                description=str(
                    definition.get("description", "")
                ),
            )
        )

    return result


def _required(schema: dict[str, Any]) -> list[str]:
    required = schema.get("required")

    if not isinstance(required, list):
        return []

    return [str(item) for item in required]


def _stats(usage: Any) -> ToolUsageStats:
    if usage is None:
        return ToolUsageStats()

    return ToolUsageStats(
        calls=usage.calls,
        running=usage.running,
        successes=usage.successes,
        errors=usage.errors,
        error_rate=usage.error_rate,
        average_duration=usage.average_duration,
        last_error_code=usage.last_error_code,
        error_codes=dict(usage.error_codes),
    )


# ======================================================================
# Historical statistics
# ======================================================================


@router.post(
    "/{session_id}/tools/{name}",
    response_model=ToolToggleResponse,
)
async def set_tool_enabled(
    session_id: str,
    name: str,
    payload: ToolToggleWrite,
    request: Request,
) -> ToolToggleResponse:
    """Switch one tool off or on for this and future sessions.

    Disabling removes the tool from the live registry, so the model
    cannot call it for the rest of the run. The choice is written to
    ``[tools] disabled`` so it survives a restart.
    """

    session = await _session(request, session_id)

    toggle = session.runtime.tool_toggle

    if toggle is None:
        raise HTTPException(
            status_code=409,
            detail=(
                "This session has no tool toggle; it was built "
                "without one."
            ),
        )

    if payload.enabled:
        known = toggle.enable(name)
    else:
        known = toggle.disable(name)

    if not known:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown tool '{name}'",
        )

    session.refresh_permissions()

    _persist_toggle(request, toggle)

    return ToolToggleResponse(
        session_id=session_id,
        tool=name,
        enabled=payload.enabled,
        known=True,
        disabled=list(toggle.disabled_names()),
        visible_tools=len(session.agent.registry.all()),
    )


def _settings_of(request: Request) -> SettingsService:
    """The app's settings service, cached on app state."""

    cached = getattr(
        request.app.state,
        "settings_service",
        None,
    )

    if isinstance(cached, SettingsService):
        return cached

    service = SettingsService(AgentWorkflowPaths())

    request.app.state.settings_service = service

    return service


def _persist_toggle(
    request: Request,
    toggle: Any,
) -> None:
    """Write the current disabled set back to config.toml.

    Best effort: the in-memory switch has already happened, so a
    settings failure must not fail the request. Losing persistence is
    reported through the settings service's own error path rather
    than being swallowed silently.
    """

    try:
        service = _settings_of(request)
    except Exception:
        logger.warning(
            "Tool toggle applied but the settings service is "
            "unavailable, so the change was not persisted",
            exc_info=True,
        )
        return

    try:
        document = service.read(CONFIG_FILE)
    except Exception:
        logger.warning(
            "Could not read %s to persist the tool toggle",
            CONFIG_FILE,
            exc_info=True,
        )
        return

    data = dict(document.data)
    tools = dict(data.get("tools") or {})

    tools["disabled"] = list(toggle.disabled_names())

    data["tools"] = tools

    try:
        service.write(CONFIG_FILE, data)
    except Exception:
        logger.warning(
            "Could not persist the tool toggle to %s",
            CONFIG_FILE,
            exc_info=True,
        )


@router.get("/history/aggregate", response_model=GlobalStatsResponse)
async def aggregate_history(
    request: Request,
    limit: int = 10,
) -> GlobalStatsResponse:
    """Tool usage aggregated across every session ever recorded.

    Backed by SQLite, so these totals survive a restart even
    though the live per-session counters do not.
    """

    store = manager_of(request).stats_store

    if store is None:
        return GlobalStatsResponse()

    bounded = max(1, min(int(limit), 100))

    totals = store.tool_totals()

    return GlobalStatsResponse(
        total_calls=store.total_tool_calls(),
        distinct_tools=len(totals),
        top_tools=[
            GlobalToolUsage(
                tool_name=record.tool_name,
                calls=record.calls,
                successes=record.successes,
                errors=record.errors,
                average_duration=record.average_duration,
                error_rate=record.error_rate,
            )
            for record in totals[:bounded]
        ],
    )

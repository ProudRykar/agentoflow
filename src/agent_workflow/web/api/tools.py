from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request

from agent_workflow.core.application.session import SessionManager
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
    ToolUsageStats,
)


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

    return ToolsResponse(
        tools=tools,
        summary=ToolsSummary(
            registered_tools=len(tools),
            **stats.summary(),  # type: ignore[arg-type]
        ),
    )


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

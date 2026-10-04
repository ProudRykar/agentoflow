from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from agent_workflow.core.application.session import SessionManager
from agent_workflow.web.schemas import PluginInfo


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


@router.get("/{session_id}", response_model=list[PluginInfo])
async def list_plugins(
    session_id: str,
    request: Request,
) -> list[PluginInfo]:
    session = await _session(request, session_id)

    plugin_manager = session.runtime.plugin_manager

    if plugin_manager is None:
        return []

    result: list[PluginInfo] = []

    for plugin in plugin_manager.registry.list():
        context = plugin_manager.get_plugin_context(
            plugin.name
        )

        result.append(
            PluginInfo(
                name=plugin.name,
                version=plugin.version,
                status=getattr(
                    plugin.state,
                    "value",
                    str(plugin.state),
                ),
                description=plugin.metadata.description,
                author=plugin.metadata.author,
                capabilities=list(
                    plugin.metadata.capabilities
                ),
                tools=sorted(context.registered_tools)
                if context
                else [],
                skills=sorted(context.registered_skills)
                if context
                else [],
                error=plugin.error,
            ),
        )

    result.sort(key=lambda item: item.name)

    return result

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request

from agent_workflow.core.application.session import SessionManager
from agent_workflow.web.schemas import SkillInfo


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


@router.get("/{session_id}", response_model=list[SkillInfo])
async def list_skills(
    session_id: str,
    request: Request,
) -> list[SkillInfo]:
    session = await _session(request, session_id)

    skill_manager = session.agent.skill_manager

    if skill_manager is None:
        return []

    available = await skill_manager.list_available_metadata()

    active = set(skill_manager.active_skills)
    loaded = set(skill_manager.loaded_skills)
    used = set(skill_manager.used_skills)

    result: list[SkillInfo] = []

    for entry in available:
        name = str(entry.get("name", ""))

        if not name:
            continue

        result.append(
            SkillInfo(
                name=name,
                description=str(
                    entry.get("description", "")
                ),
                version=str(entry.get("version", "")),
                active=name in active,
                loaded=name in loaded,
                used=name in used,
            ),
        )

    # Skills loaded by a plugin may not appear in the discovered
    # metadata; surface the registry so nothing is hidden.
    seen = {item.name for item in result}

    for skill in skill_manager.registry.list():
        name = skill.name

        if name in seen:
            continue

        result.append(
            SkillInfo(
                name=name,
                description=skill.description,
                version=skill.version,
                active=name in active,
                loaded=name in loaded,
                used=name in used,
            ),
        )

    result.sort(key=lambda item: item.name)

    return result

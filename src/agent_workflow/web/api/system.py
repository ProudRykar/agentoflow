from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from agent_workflow.web.schemas import SystemInfo


router = APIRouter()

VERSION = "0.1.0"


@router.get("/{session_id}", response_model=SystemInfo)
async def get_system_info(
    session_id: str,
    request: Request,
) -> SystemInfo:
    manager = getattr(
        request.app.state,
        "session_manager",
        None,
    )

    session = (
        await manager.get_session(session_id)
        if manager is not None
        else None
    )

    if session is None:
        raise HTTPException(
            status_code=404,
            detail="Session not found",
        )

    state = session.agent.orchestrator.state
    config = session.config

    return SystemInfo(
        version=VERSION,
        model=config.llm.model,
        working_directory=str(session.working_directory),
        max_iterations=getattr(
            session.agent,
            "_max_iterations",
            None,
        ),
        agent_phase=getattr(
            getattr(state, "phase", None),
            "value",
            None,
        ),
    )

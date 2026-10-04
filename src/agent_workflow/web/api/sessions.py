from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Request

from agent_workflow.core.application.session import (
    AgentSession,
    SessionManager,
)
from agent_workflow.web.schemas import (
    ApprovalDecisionResponse,
    ApprovalStateInfo,
    CancelResponse,
    MessageRequest,
    ResumeResponse,
    RunRequest,
    RunResponse,
    SessionCreateRequest,
    SessionCreateResponse,
    SessionInfo,
    SessionRenameRequest,
    SessionStateInfo,
)


router = APIRouter()


def manager_of(request: Request) -> SessionManager:
    manager = getattr(
        request.app.state,
        "session_manager",
        None,
    )

    if manager is None:
        manager = SessionManager()
        request.app.state.session_manager = manager

    return manager


async def require_session(
    manager: SessionManager,
    session_id: str,
) -> AgentSession:
    session = await manager.get_session(session_id)

    if session is None:
        raise HTTPException(
            status_code=404,
            detail="Session not found",
        )

    if session.state.value == "closed":
        raise HTTPException(
            status_code=409,
            detail="Session is closed",
        )

    return session


def to_info(session: AgentSession) -> SessionInfo:
    snapshot = session.snapshot()

    return SessionInfo(
        session_id=session.session_id,
        created_at=session.created_at,
        metadata=session.metadata,
        title=snapshot["title"],
        title_source=snapshot["title_source"],
        model=snapshot["model"],
        working_directory=snapshot["working_directory"],
        subscriber_count=session.events.subscriber_count,
        latest_seq=session.events.seq,
        session=SessionStateInfo(
            state=snapshot["state"],
            phase=snapshot["phase"],
            iteration=snapshot["iteration"],
            running=snapshot["running"],
            last_error=snapshot["last_error"],
        ),
    )


@router.post("", response_model=SessionCreateResponse)
async def create_session(
    request: Request,
    body: SessionCreateRequest | None = None,
) -> SessionCreateResponse:
    manager = manager_of(request)

    metadata = body.metadata if body else {}
    raw_directory = body.working_directory if body else None

    working_directory = resolve_working_directory(raw_directory)

    session = await manager.create_session(
        metadata=metadata,
        working_directory=working_directory,
    )

    return SessionCreateResponse(
        session_id=session.session_id,
        state=session.state.value,
    )


def resolve_working_directory(
    raw: str | None,
) -> Path | None:
    """Validate a requested sandbox root.

    The value becomes both the agent's working directory and the
    PathPolicy allow-list, so a bad path must fail loudly here
    rather than deep inside tool execution. ``None`` keeps the
    server default (the process working directory).
    """

    if raw is None or not raw.strip():
        return None

    path = Path(raw.strip()).expanduser()

    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"working_directory is not accessible: {path}",
        ) from exc

    if not resolved.is_dir():
        raise HTTPException(
            status_code=400,
            detail=f"working_directory is not a directory: {resolved}",
        )

    return resolved


def to_record_info(
    manager: SessionManager,
    record,
) -> SessionInfo:
    """Describe a recorded session that has no live runtime.

    Listing must not rehydrate every session: that would build an
    Agent, tool registry and plugin set per chat just to render the
    sidebar. The client fetches a session when it is opened, which
    is where the runtime is rebuilt.
    """

    store = manager.event_store

    latest_seq = 0

    if store is not None:
        try:
            latest_seq = store.latest_seq(record.session_id)
        except Exception:
            latest_seq = 0

    return SessionInfo(
        session_id=record.session_id,
        created_at=record.created_at,
        metadata=dict(record.metadata),
        title=record.title,
        title_source=record.title_source,
        model="",
        working_directory=record.working_directory,
        subscriber_count=0,
        latest_seq=latest_seq,
        session=SessionStateInfo(state="idle"),
    )


@router.get("", response_model=list[SessionInfo])
async def list_sessions(
    request: Request,
) -> list[SessionInfo]:
    manager = manager_of(request)

    live = {
        session.session_id: to_info(session)
        for session in manager.list_sessions()
    }

    listed: list[SessionInfo] = list(live.values())

    # Sessions restored from disk but not yet opened still belong in
    # the sidebar: losing them on restart is what makes a chat
    # history feel broken.
    for record in manager.list_records():
        if record.session_id in live:
            continue

        listed.append(to_record_info(manager, record))

    listed.sort(key=lambda item: item.created_at)

    return listed


@router.get("/{session_id}", response_model=SessionInfo)
async def get_session(
    session_id: str,
    request: Request,
) -> SessionInfo:
    manager = manager_of(request)

    session = await require_session(manager, session_id)

    return to_info(session)


@router.patch("/{session_id}/title")
async def rename_session(
    session_id: str,
    body: SessionRenameRequest,
    request: Request,
) -> dict:
    """Rename a session.

    A manual title is final: it is never overwritten by a later
    generated one.
    """

    manager = manager_of(request)

    await require_session(manager, session_id)

    try:
        title = await manager.rename_session(
            session_id,
            body.title,
        )
    except KeyError as exc:
        raise HTTPException(
            status_code=404,
            detail="Session not found",
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    return {"session_id": session_id, "title": title}


@router.delete("/{session_id}")
async def close_session(
    session_id: str,
    request: Request,
) -> dict:
    manager = manager_of(request)

    if not await manager.close_session(session_id):
        raise HTTPException(
            status_code=404,
            detail="Session not found",
        )

    return {"session_id": session_id, "status": "closed"}


@router.get(
    "/{session_id}/approval",
    response_model=ApprovalStateInfo | None,
)
async def get_approval(
    session_id: str,
    request: Request,
) -> ApprovalStateInfo | None:
    manager = manager_of(request)

    session = await require_session(manager, session_id)

    pending = session.approval.request

    if pending is None:
        return None

    return ApprovalStateInfo(
        approval_id=pending.approval_id,
        tool_name=pending.tool_name,
        permission=pending.permission,
        reason=pending.reason,
        arguments=dict(pending.arguments),
    )


@router.post(
    "/{session_id}/approvals/{approval_id}/allow",
    response_model=ApprovalDecisionResponse,
)
async def approval_allow(
    session_id: str,
    approval_id: str,
    request: Request,
) -> ApprovalDecisionResponse:
    return await _decide(
        session_id,
        approval_id,
        request,
        approved=True,
    )


@router.post(
    "/{session_id}/approvals/{approval_id}/deny",
    response_model=ApprovalDecisionResponse,
)
async def approval_deny(
    session_id: str,
    approval_id: str,
    request: Request,
) -> ApprovalDecisionResponse:
    return await _decide(
        session_id,
        approval_id,
        request,
        approved=False,
    )


async def _decide(
    session_id: str,
    approval_id: str,
    request: Request,
    approved: bool,
) -> ApprovalDecisionResponse:
    manager = manager_of(request)

    session = await require_session(manager, session_id)

    if not session.approval.active:
        raise HTTPException(
            status_code=409,
            detail="No approval is pending",
        )

    if session.approval.approval_id != approval_id:
        raise HTTPException(
            status_code=409,
            detail="Approval id does not match "
            "the pending request",
        )

    if approved:
        accepted = session.approval.allow(approval_id)
    else:
        accepted = session.approval.deny(approval_id)

    if not accepted:
        raise HTTPException(
            status_code=409,
            detail="Approval is no longer pending",
        )

    return ApprovalDecisionResponse(
        session_id=session_id,
        approval_id=approval_id,
        accepted=True,
    )


@router.post("/{session_id}/run", response_model=RunResponse)
async def start_run(
    session_id: str,
    body: RunRequest,
    request: Request,
) -> RunResponse:
    manager = manager_of(request)

    session = await require_session(manager, session_id)

    await _start(
        session,
        lambda: session.start_run(body.prompt.strip()),
    )

    return RunResponse(
        session_id=session_id,
        status="started",
        state=session.state.value,
    )


@router.post("/{session_id}/continue", response_model=RunResponse)
async def continue_run(
    session_id: str,
    body: RunRequest,
    request: Request,
) -> RunResponse:
    manager = manager_of(request)

    session = await require_session(manager, session_id)

    await _start(
        session,
        lambda: session.start_continue(body.prompt.strip()),
    )

    return RunResponse(
        session_id=session_id,
        status="started",
        state=session.state.value,
    )


@router.post("/{session_id}/resume", response_model=ResumeResponse)
async def resume_run(
    session_id: str,
    request: Request,
) -> ResumeResponse:
    manager = manager_of(request)

    session = await require_session(manager, session_id)

    await _start(session, session.start_resume)

    return ResumeResponse(
        session_id=session_id,
        status="started",
        state=session.state.value,
    )


async def _start(
    session: AgentSession,
    action: object,
) -> None:
    try:
        await action()  # type: ignore[operator]
    except RuntimeError as exc:
        raise HTTPException(
            status_code=409,
            detail=str(exc),
        ) from exc


@router.post("/{session_id}/messages", response_model=RunResponse)
async def post_message(
    session_id: str,
    body: MessageRequest,
    request: Request,
) -> RunResponse:
    """Submit a user turn.

    A single entry point for the chat UI: the caller sends text and
    the session decides whether it opens a conversation or continues
    the current one.
    """

    manager = manager_of(request)

    session = await require_session(manager, session_id)

    content = body.content.strip()

    if not content:
        raise HTTPException(
            status_code=422,
            detail="Message must not be empty",
        )

    if body.mode == "resume":
        await _start(session, session.start_resume)

        return RunResponse(
            session_id=session_id,
            status="started",
            state=session.state.value,
            mode=None,
        )

    if body.mode == "run":
        await _start(
            session,
            lambda: session.start_run(content),
        )

        return RunResponse(
            session_id=session_id,
            status="started",
            state=session.state.value,
            mode="run",
        )

    if body.mode == "continue":
        await _start(
            session,
            lambda: session.start_continue(content),
        )

        return RunResponse(
            session_id=session_id,
            status="started",
            state=session.state.value,
            mode="continue",
        )

    # auto: first turn opens the conversation, later turns continue.
    if session.has_history:
        await _start(
            session,
            lambda: session.start_continue(content),
        )

        mode = "continue"
    else:
        await _start(
            session,
            lambda: session.start_run(content),
        )

        mode = "run"

    return RunResponse(
        session_id=session_id,
        status="started",
        state=session.state.value,
        mode=mode,
    )


@router.post("/{session_id}/cancel", response_model=CancelResponse)
async def cancel_run(
    session_id: str,
    request: Request,
) -> CancelResponse:
    manager = manager_of(request)

    session = await require_session(manager, session_id)

    cancelled = await session.cancel()

    return CancelResponse(
        session_id=session_id,
        cancelled=cancelled,
        state=session.state.value,
    )

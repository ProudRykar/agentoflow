from __future__ import annotations

import asyncio
import json
from contextlib import suppress
from typing import Any

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

from agent_workflow.core.application.session import SessionManager
from agent_workflow.web.auth import AuthSettings, guard_websocket
from agent_workflow.web.serialization import dumps


router = APIRouter()

HEARTBEAT_SECONDS = 20.0

SEND_TIMEOUT_SECONDS = 10.0


def session_manager_of(
    websocket: WebSocket,
) -> SessionManager:
    """Resolve the manager from the serving app instance.

    Importing the module-level ``app`` would bind the router to a
    single application object, breaking both multiple apps and
    tests. The running app is always on the connection scope.
    """

    manager = getattr(
        websocket.app.state,
        "session_manager",
        None,
    )

    if manager is None:
        manager = SessionManager()
        websocket.app.state.session_manager = manager

    return manager


def auth_of(websocket: WebSocket) -> AuthSettings:
    settings = getattr(
        websocket.app.state,
        "auth",
        None,
    )

    if isinstance(settings, AuthSettings):
        return settings

    return AuthSettings()


@router.websocket("/sessions/{session_id}")
async def agent_socket(
    websocket: WebSocket,
    session_id: str,
    since: int = Query(
        default=-1,
        ge=-1,
    ),
) -> None:
    if not await guard_websocket(
        websocket,
        auth_of(websocket),
    ):
        return

    manager = session_manager_of(websocket)

    session = await manager.get_session(session_id)

    if session is None:
        await websocket.close(
            code=4004,
            reason="Session not found",
        )
        return

    await websocket.accept()

    bus = session.events

    # Replay first, then attach. Subscribing after the replay
    # closes the gap where an event could be published between
    # reading history and starting the live subscription.
    backlog = bus.replay_since(since)

    pending = asyncio.Queue()
    key = bus.subscribe(
        lambda seq, event: _enqueue(pending, seq, event),
    )

    try:
        for stored in backlog:
            await _send(
                websocket,
                session_id,
                stored.seq,
                stored.event,
            )

        oldest = bus.oldest_retained_seq()

        await _send_raw(
            websocket,
            _hello(
                session_id=session_id,
                latest=bus.seq,
                oldest=oldest,
                replayed=len(backlog),
                state=session.snapshot(),
            ),
        )

        reader = asyncio.create_task(
            _read_client(websocket),
        )

        try:
            while True:
                if reader.done():
                    # Client went away; stop forwarding.
                    break

                # Wait for the next event or for the peer to
                # disappear, whichever comes first. Racing the
                # reader means a disconnect is noticed at once
                # instead of after the heartbeat interval.
                getter = asyncio.create_task(pending.get())

                done, _ = await asyncio.wait(
                    {getter, reader},
                    timeout=HEARTBEAT_SECONDS,
                    return_when=asyncio.FIRST_COMPLETED,
                )

                if not done:
                    await _discard(getter)

                    if not await _heartbeat(websocket):
                        break

                    continue

                if reader in done:
                    await _discard(getter)
                    break

                seq, event = getter.result()

                if not await _send(
                    websocket,
                    session_id,
                    seq,
                    event,
                ):
                    break

        except WebSocketDisconnect:
            pass

        finally:
            reader.cancel()

            # The reader finishes by raising WebSocketDisconnect
            # when the client goes away, so both that and a
            # cancellation must be absorbed here: re-raising from
            # a finally block would escape the handler and be
            # logged as an ASGI error on every disconnect.
            with suppress(
                asyncio.CancelledError,
                WebSocketDisconnect,
            ):
                await reader

    finally:
        bus.unsubscribe(key)

        try:
            await websocket.close()
        except (RuntimeError, WebSocketDisconnect):
            # Already closed by the peer.
            pass


def _hello(
    session_id: str,
    latest: int,
    oldest: int,
    replayed: int,
    state: dict[str, Any],
) -> str:
    return json.dumps(
        {
            "type": "session.snapshot",
            "session_id": session_id,
            "seq": latest,
            "oldest_seq": oldest,
            "replayed": replayed,
            "data": state,
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )


async def _discard(task: asyncio.Task) -> None:
    """Cancel a pending getter and wait for it to settle."""

    task.cancel()

    with suppress(asyncio.CancelledError):
        await task


async def _enqueue(
    queue: asyncio.Queue,
    seq: int,
    event: Any,
) -> None:
    await queue.put((seq, event))


async def _send(
    websocket: WebSocket,
    session_id: str,
    seq: int,
    event: Any,
) -> bool:
    return await _send_raw(
        websocket,
        dumps(event, session_id=session_id, seq=seq),
    )


async def _heartbeat(websocket: WebSocket) -> bool:
    """Keep intermediaries from dropping an idle socket."""

    return await _send_raw(
        websocket,
        '{"type":"heartbeat"}',
    )


async def _send_raw(
    websocket: WebSocket,
    text: str,
) -> bool:
    if websocket.client_state is not WebSocketState.CONNECTED:
        return False

    try:
        await asyncio.wait_for(
            websocket.send_text(text),
            timeout=SEND_TIMEOUT_SECONDS,
        )
    except Exception:
        return False

    return True


async def _read_client(websocket: WebSocket) -> None:
    """Consume client frames so disconnects are detected."""

    while True:
        message = await websocket.receive_text()

        if message == "ping":
            try:
                await websocket.send_text("pong")
            except Exception:
                return

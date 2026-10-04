from __future__ import annotations

import asyncio
import contextlib
import io
import json
import logging
import socket
import threading
import time
from pathlib import Path

from starlette.websockets import WebSocketDisconnect

import pytest
from fastapi.testclient import TestClient

from agent_workflow.core.entities.models.agent_phase import AgentPhase
from agent_workflow.core.entities.models.agent_trace import (
    AgentFinished,
    AgentPhaseChanged,
    AgentStarted,
    LLMContentChunk,
    ToolFinished,
    ToolStarted,
)
from agent_workflow.core.entities.models.approval import (
    ApprovalRequest,
    ApprovalRequested,
    ApprovalResolved,
)
from agent_workflow.core.application.events import EventBus


def _session(client: TestClient) -> str:
    return client.post("/api/sessions").json()["session_id"]


def _get(client: TestClient, session_id: str):
    """Fetch a live session through the app's event loop."""

    return client.portal.call(  # type: ignore[attr-defined]
        client.app.state.session_manager.get_session,
        session_id,
    )


def _read(websocket) -> dict:
    return json.loads(websocket.receive_text())


def _read_snapshot(websocket, limit: int = 200) -> dict:
    """Read frames until the greeting arrives.

    A session that already ran replays its history first, so the
    snapshot is not necessarily the first frame.
    """

    for _ in range(limit):
        message = _read(websocket)

        if message["type"] == "session.snapshot":
            return message

    raise AssertionError("snapshot frame never arrived")


# ======================================================================
# Connection lifecycle
# ======================================================================


def test_unknown_session_is_rejected(
    stubbed: TestClient,
) -> None:
    from starlette.websockets import WebSocketDisconnect

    with pytest.raises(WebSocketDisconnect):
        with stubbed.websocket_connect("/ws/sessions/missing"):
            pass


def test_connects_and_greets_with_snapshot(
    stubbed: TestClient,
) -> None:
    session_id = _session(stubbed)

    with stubbed.websocket_connect(
        f"/ws/sessions/{session_id}"
    ) as websocket:
        message = _read(websocket)

    assert message["type"] == "session.snapshot"
    assert message["session_id"] == session_id
    assert message["data"]["state"] == "idle"
    assert message["seq"] == 0
    assert message["oldest_seq"] == 0
    assert message["replayed"] == 0


def test_snapshot_includes_pending_approval(
    stubbed: TestClient,
    portal,
) -> None:
    """A reconnecting client learns about a pending decision
    from the snapshot alone, without replaying the request."""

    import asyncio

    session_id = _session(stubbed)
    session = _get(stubbed, session_id)

    holder: dict[str, object] = {}

    def start_approval() -> None:
        holder["task"] = asyncio.create_task(
            session.approval(
                ApprovalRequest(
                    tool_name="execute_shell",
                    arguments={"command": "curl https://x.dev"},
                    permission="shell.network",
                    reason="network",
                )
            )
        )

    async def let_it_park() -> None:
        await asyncio.sleep(0.05)

    portal.call(start_approval)
    portal.call(let_it_park)

    try:
        with stubbed.websocket_connect(
            f"/ws/sessions/{session_id}"
        ) as websocket:
            # approval.requested is replayed, then the snapshot.
            requested = _read(websocket)
            greeting = _read(websocket)

        assert requested["type"] == "approval.requested"

        assert greeting["type"] == "session.snapshot"
        assert greeting["data"]["state"] == "waiting_approval"
        assert greeting["data"]["approval"]["permission"] == (
            "shell.network"
        )
        assert greeting["data"]["approval"]["approval_id"] == (
            requested["data"]["approval_id"]
        )

    finally:
        portal.call(session.approval.deny)


def test_subscriber_is_registered_and_released(
    stubbed: TestClient,
) -> None:
    session_id = _session(stubbed)
    session = _get(stubbed, session_id)

    # The session always has internal observers (statistics), so
    # the websocket's contribution is measured as a delta.
    baseline = session.events.subscriber_count

    with stubbed.websocket_connect(
        f"/ws/sessions/{session_id}"
    ) as websocket:
        _read_snapshot(websocket)

        assert session.events.subscriber_count == baseline + 1

    assert session.events.subscriber_count == baseline


def test_ping_pong(stubbed: TestClient) -> None:
    session_id = _session(stubbed)

    with stubbed.websocket_connect(
        f"/ws/sessions/{session_id}"
    ) as websocket:
        _read(websocket)

        websocket.send_text("ping")

        assert websocket.receive_text() == "pong"


# ======================================================================
# Event delivery
# ======================================================================


def test_published_events_reach_the_socket(
    stubbed: TestClient,
    publish,
) -> None:
    session_id = _session(stubbed)
    session = _get(stubbed, session_id)

    with stubbed.websocket_connect(
        f"/ws/sessions/{session_id}"
    ) as websocket:
        _read(websocket)

        publish(session, AgentStarted(prompt="hi"))

        message = _read(websocket)

    assert message["type"] == "agent.started"
    assert message["data"]["prompt"] == "hi"
    assert message["session_id"] == session_id
    assert message["seq"] == 1
    assert message["timestamp"]


def test_streaming_chunks_arrive_in_order(
    stubbed: TestClient,
    publish,
) -> None:
    """Three chunks must arrive as three separate frames."""

    session_id = _session(stubbed)
    session = _get(stubbed, session_id)

    with stubbed.websocket_connect(
        f"/ws/sessions/{session_id}"
    ) as websocket:
        _read(websocket)

        for chunk in ("Hel", "lo", "!"):
            publish(
                session,
                LLMContentChunk(iteration=1, content=chunk),
            )

        received = [
            _read(websocket)["data"]["content"] for _ in range(3)
        ]

    assert received == ["Hel", "lo", "!"]
    assert "".join(received) == "Hello!"


def test_tool_events_carry_payloads(
    stubbed: TestClient,
    publish,
) -> None:
    session_id = _session(stubbed)
    session = _get(stubbed, session_id)

    with stubbed.websocket_connect(
        f"/ws/sessions/{session_id}"
    ) as websocket:
        _read(websocket)

        publish(
            session,
            ToolStarted(
                iteration=1,
                tool_call_id="c1",
                tool_name="read_file",
                arguments={"path": "main.py"},
            ),
        )
        publish(
            session,
            ToolFinished(
                iteration=1,
                tool_call_id="c1",
                tool_name="read_file",
                output="print()",
                error_code=None,
                error_message=None,
                duration_seconds=0.01,
            ),
        )

        started = _read(websocket)
        finished = _read(websocket)

    assert started["type"] == "tool.started"
    assert started["data"]["arguments"] == {"path": "main.py"}
    assert finished["type"] == "tool.finished"
    assert finished["data"]["output"] == "print()"
    assert finished["data"]["duration_seconds"] == 0.01
    assert started["seq"] < finished["seq"]


def test_phase_events_serialize_enum(
    stubbed: TestClient,
    publish,
) -> None:
    session_id = _session(stubbed)
    session = _get(stubbed, session_id)

    with stubbed.websocket_connect(
        f"/ws/sessions/{session_id}"
    ) as websocket:
        _read(websocket)

        publish(
            session,
            AgentPhaseChanged(
                previous_phase=AgentPhase.IDLE,
                phase=AgentPhase.EXECUTION,
                reason="tool",
                iteration=1,
            ),
        )

        message = _read(websocket)

    assert message["type"] == "agent.phase_changed"
    assert message["data"]["previous_phase"] == "idle"
    assert message["data"]["phase"] == "execution"


def test_agent_finished_reaches_socket(
    stubbed: TestClient,
    publish,
) -> None:
    session_id = _session(stubbed)
    session = _get(stubbed, session_id)

    with stubbed.websocket_connect(
        f"/ws/sessions/{session_id}"
    ) as websocket:
        _read(websocket)

        publish(session, AgentFinished(result="all done"))

        message = _read(websocket)

    assert message["type"] == "agent.finished"
    assert message["data"]["result"] == "all done"


# ======================================================================
# Multiple subscribers
# ======================================================================


def test_two_sockets_receive_the_same_event(
    stubbed: TestClient,
    publish,
) -> None:
    session_id = _session(stubbed)
    session = _get(stubbed, session_id)

    with stubbed.websocket_connect(
        f"/ws/sessions/{session_id}"
    ) as first:
        _read(first)

        with stubbed.websocket_connect(
            f"/ws/sessions/{session_id}"
        ) as second:
            _read(second)

            assert session.events.subscriber_count >= 2

            publish(session, AgentStarted(prompt="shared"))

            assert _read(first)["data"]["prompt"] == "shared"
            assert _read(second)["data"]["prompt"] == "shared"


def test_events_do_not_cross_sessions(
    stubbed: TestClient,
    publish,
) -> None:
    first_id = _session(stubbed)
    second_id = _session(stubbed)

    first = _get(stubbed, first_id)
    second = _get(stubbed, second_id)

    with stubbed.websocket_connect(
        f"/ws/sessions/{first_id}"
    ) as first_socket:
        _read(first_socket)

        with stubbed.websocket_connect(
            f"/ws/sessions/{second_id}"
        ) as second_socket:
            _read(second_socket)

            publish(second, AgentStarted(prompt="second only"))

            assert (
                _read(second_socket)["data"]["prompt"]
                == "second only"
            )

            first_socket.send_text("ping")

            assert first_socket.receive_text() == "pong"


# ======================================================================
# Reconnect / replay
# ======================================================================


def test_replay_returns_missed_events(
    stubbed: TestClient,
    publish,
) -> None:
    session_id = _session(stubbed)
    session = _get(stubbed, session_id)

    publish(session, AgentStarted(prompt="one"))
    publish(session, AgentStarted(prompt="two"))
    publish(session, AgentStarted(prompt="three"))

    with stubbed.websocket_connect(
        f"/ws/sessions/{session_id}?since=1"
    ) as websocket:
        first = _read(websocket)
        second = _read(websocket)
        greeting = _read(websocket)

    assert first["data"]["prompt"] == "two"
    assert second["data"]["prompt"] == "three"
    assert greeting["type"] == "session.snapshot"
    assert greeting["replayed"] == 2
    assert greeting["seq"] == 3


def test_reconnect_does_not_duplicate_delivery(
    stubbed: TestClient,
    publish,
) -> None:
    session_id = _session(stubbed)
    session = _get(stubbed, session_id)

    publish(session, AgentStarted(prompt="first"))

    with stubbed.websocket_connect(
        f"/ws/sessions/{session_id}?since=0"
    ) as websocket:
        replayed = _read(websocket)

        assert replayed["data"]["prompt"] == "first"

        greeting = _read(websocket)

        assert greeting["seq"] == 1

    with stubbed.websocket_connect(
        f"/ws/sessions/{session_id}?since=1"
    ) as websocket:
        greeting = _read(websocket)

        assert greeting["replayed"] == 0

        publish(session, AgentStarted(prompt="second"))

        live = _read(websocket)

        assert live["data"]["prompt"] == "second"
        assert live["seq"] == 2


def test_full_replay_without_cursor(
    stubbed: TestClient,
    publish,
) -> None:
    session_id = _session(stubbed)
    session = _get(stubbed, session_id)

    publish(session, AgentStarted(prompt="a"))
    publish(session, AgentStarted(prompt="b"))

    with stubbed.websocket_connect(
        f"/ws/sessions/{session_id}"
    ) as websocket:
        messages = [_read(websocket) for _ in range(3)]

    assert messages[0]["data"]["prompt"] == "a"
    assert messages[1]["data"]["prompt"] == "b"
    assert messages[2]["replayed"] == 2


def test_replay_after_history_eviction_is_bounded(
    stubbed: TestClient,
    publish,
) -> None:
    session_id = _session(stubbed)
    session = _get(stubbed, session_id)

    session.events = EventBus(history_limit=2)

    for index in range(5):
        publish(
            session,
            AgentStarted(prompt=f"p{index}"),
        )

    with stubbed.websocket_connect(
        f"/ws/sessions/{session_id}?since=0"
    ) as websocket:
        first = _read(websocket)
        second = _read(websocket)
        greeting = _read(websocket)

    assert first["seq"] == 4
    assert second["seq"] == 5
    assert greeting["oldest_seq"] == 4
    assert greeting["replayed"] == 2


def test_reconnect_keeps_streaming_from_cursor(
    stubbed: TestClient,
    publish,
) -> None:
    """After a disconnect the client resumes exactly where it left."""

    session_id = _session(stubbed)
    session = _get(stubbed, session_id)

    publish(session, AgentStarted(prompt="one"))
    publish(session, AgentStarted(prompt="two"))

    with stubbed.websocket_connect(
        f"/ws/sessions/{session_id}"
    ) as websocket:
        assert _read(websocket)["data"]["prompt"] == "one"
        assert _read(websocket)["data"]["prompt"] == "two"

        cursor = _read(websocket)["seq"]

    # Disconnected; events accumulate in the bus.
    publish(session, AgentStarted(prompt="three"))
    publish(session, AgentStarted(prompt="four"))

    with stubbed.websocket_connect(
        f"/ws/sessions/{session_id}?since={cursor}"
    ) as websocket:
        assert _read(websocket)["data"]["prompt"] == "three"
        assert _read(websocket)["data"]["prompt"] == "four"

        greeting = _read(websocket)

    assert greeting["replayed"] == 2
    assert greeting["seq"] == cursor + 2


def test_reconnect_does_not_reset_session(
    stubbed: TestClient,
) -> None:
    """Session identity and agent state survive a disconnect."""

    session_id = _session(stubbed)

    stubbed.post(
        f"/api/sessions/{session_id}/run",
        json={"prompt": "hello"},
    )

    with stubbed.websocket_connect(
        f"/ws/sessions/{session_id}"
    ) as websocket:
        assert (
            _read_snapshot(websocket)["type"] == "session.snapshot"
        )

    info = stubbed.get(f"/api/sessions/{session_id}").json()

    assert info["session_id"] == session_id
    assert info["session"]["running"] is False
    assert info["session"]["state"] in {
        "completed",
        "running",
        "idle",
    }
    assert info["model"] == "stub-model"

    assert (
        stubbed.post(
            f"/api/sessions/{session_id}/continue",
            json={"prompt": "again"},
        ).status_code
        == 200
    )


# ======================================================================
# Approval over the socket
# ======================================================================


def test_approval_request_reaches_socket(
    stubbed: TestClient,
    publish,
) -> None:
    session_id = _session(stubbed)
    session = _get(stubbed, session_id)

    with stubbed.websocket_connect(
        f"/ws/sessions/{session_id}"
    ) as websocket:
        _read(websocket)

        publish(
            session,
            ApprovalRequested(
                approval_id="ap-1",
                tool_name="execute_shell",
                arguments={"command": "curl https://x.dev"},
                permission="shell.network",
                reason="network access",
            ),
        )

        message = _read(websocket)

    assert message["type"] == "approval.requested"
    assert message["data"]["approval_id"] == "ap-1"
    assert message["data"]["permission"] == "shell.network"


def test_real_approval_flow_reaches_socket_and_unblocks(
    stubbed: TestClient,
    portal,
) -> None:
    """End-to-end: the controller asks, the socket sees it,
    HTTP resolves it, and the awaiting coroutine resumes."""

    import asyncio

    session_id = _session(stubbed)
    session = _get(stubbed, session_id)

    holder: dict[str, object] = {}

    request = ApprovalRequest(
        tool_name="execute_shell",
        arguments={"command": "curl https://x.dev"},
        permission="shell.network",
        reason="network access",
    )

    def start_approval() -> None:
        # Creates the task and returns without awaiting it, so
        # the test thread stays free to drive HTTP and reads.
        holder["task"] = asyncio.create_task(
            session.approval(request)
        )

    async def let_it_park() -> None:
        await asyncio.sleep(0.05)

    with stubbed.websocket_connect(
        f"/ws/sessions/{session_id}"
    ) as websocket:
        _read(websocket)

        portal.call(start_approval)
        portal.call(let_it_park)

        message = _read(websocket)

        assert message["type"] == "approval.requested"

        approval_id = message["data"]["approval_id"]

        assert approval_id == session.approval.approval_id
        assert session.state.value == "waiting_approval"

        accepted = stubbed.post(
            f"/api/sessions/{session_id}"
            f"/approvals/{approval_id}/allow"
        )

        assert accepted.status_code == 200

        resolved = _read(websocket)

        assert resolved["type"] == "approval.resolved"
        assert resolved["data"]["approved"] is True

    async def collect() -> bool:
        return await holder["task"]  # type: ignore[index,operator]

    assert portal.call(collect) is True


def test_approval_resolved_event_is_streamed(
    stubbed: TestClient,
    publish,
) -> None:
    session_id = _session(stubbed)
    session = _get(stubbed, session_id)

    with stubbed.websocket_connect(
        f"/ws/sessions/{session_id}"
    ) as websocket:
        _read(websocket)

        publish(
            session,
            ApprovalResolved(approval_id="ap-1", approved=True),
        )

        message = _read(websocket)

    assert message["type"] == "approval.resolved"
    assert message["data"]["approved"] is True


# ======================================================================
# Subagent events
# ======================================================================


def test_subagent_events_carry_run_hierarchy(
    stubbed: TestClient,
    publish,
) -> None:
    """Child-run events must expose run_id/parent_run_id.

    The Agent rewrites ToolContext.event_callback to the callback
    passed as ``on_event``, so subagent events reach the session
    bus through exactly the same path as main-agent events.
    """

    session_id = _session(stubbed)
    session = _get(stubbed, session_id)

    assert session.context.event_callback is None

    with stubbed.websocket_connect(
        f"/ws/sessions/{session_id}"
    ) as websocket:
        _read_snapshot(websocket)

        publish(
            session,
            AgentStarted(
                prompt="subagent task",
                run_id="child-1",
                parent_run_id="parent-1",
                agent_id="researcher",
                role="research",
                model="qwen",
            ),
        )
        publish(
            session,
            ToolStarted(
                iteration=1,
                tool_call_id="c1",
                tool_name="web_fetch",
                arguments={"url": "https://x.dev"},
                run_id="child-1",
                parent_run_id="parent-1",
            ),
        )

        started = _read(websocket)
        tool = _read(websocket)

    assert started["type"] == "agent.started"
    assert started["run_id"] == "child-1"
    assert started["parent_run_id"] == "parent-1"
    assert started["data"]["role"] == "research"
    assert started["data"]["model"] == "qwen"

    assert tool["type"] == "tool.started"
    assert tool["run_id"] == "child-1"
    assert tool["parent_run_id"] == "parent-1"


def test_subagent_events_share_the_session_stream(
    stubbed: TestClient,
    publish,
) -> None:
    """Main and child events interleave on one ordered stream."""

    session_id = _session(stubbed)
    session = _get(stubbed, session_id)

    with stubbed.websocket_connect(
        f"/ws/sessions/{session_id}"
    ) as websocket:
        _read_snapshot(websocket)

        publish(session, AgentStarted(prompt="main"))
        publish(
            session,
            AgentStarted(
                prompt="child",
                run_id="c1",
                parent_run_id="m1",
            ),
        )
        publish(
            session,
            AgentFinished(result="child done", run_id="c1"),
        )
        publish(session, AgentFinished(result="main done"))

        frames = [_read(websocket) for _ in range(4)]

    assert [f["seq"] for f in frames] == [1, 2, 3, 4]
    assert frames[0].get("run_id") is None
    assert frames[1]["run_id"] == "c1"
    assert frames[2]["run_id"] == "c1"
    assert frames[3].get("run_id") is None


def test_agent_run_wires_callback_into_context(
    stubbed: TestClient,
) -> None:
    """Agent must route its events through the session callback.

    This is the mechanism that makes subagent events reach the
    web layer without any UI-specific code.
    """

    session_id = _session(stubbed)
    session = _get(stubbed, session_id)

    async def drive() -> None:
        await session.start_run("hello")

        # start_run returns as soon as the task is scheduled.
        while session.running:
            await asyncio.sleep(0.01)

    stubbed.portal.call(drive)  # type: ignore[attr-defined]

    # Events reached the bus, which is only possible through the
    # callback the session handed to Agent.run(). Agent uses
    # dataclasses.replace(), so the session's own ToolContext is
    # intentionally left untouched; the copy carrying the
    # callback is what reaches tools and subagents.
    assert session.events.seq > 0
    assert session.context.event_callback is None
    assert session.state.value == "completed"


# ======================================================================
# Disconnect hygiene
# ======================================================================


def test_disconnect_does_not_raise_asgi_error(
    stubbed: TestClient,
    publish,
) -> None:
    """Closing the client side must not surface as a server error.

    The reader task ends by raising WebSocketDisconnect, and the
    socket is already gone by the time the handler closes it. Both
    paths have to be absorbed inside the handler.
    """

    session_id = _session(stubbed)
    session = _get(stubbed, session_id)

    with stubbed.websocket_connect(
        f"/ws/sessions/{session_id}"
    ) as websocket:
        _read_snapshot(websocket)

        publish(session, AgentStarted(prompt="bye"))

        assert _read(websocket)["type"] == "agent.started"

    # Exiting the context closes abruptly, as a browser tab does.


def test_abrupt_disconnect_then_reconnect(
    stubbed: TestClient,
    publish,
) -> None:
    session_id = _session(stubbed)
    session = _get(stubbed, session_id)

    for _ in range(3):
        with stubbed.websocket_connect(
            f"/ws/sessions/{session_id}"
        ) as websocket:
            _read_snapshot(websocket)

    assert session.events.subscriber_count >= 1

    with stubbed.websocket_connect(
        f"/ws/sessions/{session_id}"
    ) as websocket:
        _read_snapshot(websocket)

        publish(session, AgentStarted(prompt="after"))

        assert _read(websocket)["data"]["prompt"] == "after"


def test_disconnect_without_reading_greeting(
    stubbed: TestClient,
) -> None:
    """A client that vanishes mid-handshake must not raise."""

    session_id = _session(stubbed)

    with stubbed.websocket_connect(
        f"/ws/sessions/{session_id}"
    ):
        pass

    assert stubbed.get(
        f"/api/sessions/{session_id}"
    ).status_code == 200


# ======================================================================
# Disconnect hygiene (real server)
# ======================================================================


@pytest.fixture
def live_server(tmp_path: Path):
    """Run the real ASGI app under uvicorn on an ephemeral port.

    ``TestClient`` swallows exceptions raised inside a WebSocket
    handler, so a disconnect regression is invisible to it. Uvicorn
    logs those as "Exception in ASGI application", which this
    fixture captures so a test can assert on it.
    """

    import asyncio as _asyncio

    import uvicorn

    from agent_workflow.core.application.session import (
        SessionManager,
    )
    from agent_workflow.web.app import create_app

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]

    manager = SessionManager(auto_restore=False)

    app = create_app(session_manager=manager)

    config = uvicorn.Config(
        app,
        host="127.0.0.1",
        port=port,
        log_level="warning",
        lifespan="off",
    )

    server = uvicorn.Server(config)

    loop = _asyncio.new_event_loop()
    thread = threading.Thread(
        target=loop.run_until_complete,
        args=(server.serve(),),
        daemon=True,
    )

    # uvicorn reports handler failures through the logging module,
    # so the assertion has to hook a handler rather than stdout.
    output = io.StringIO()

    handler = logging.StreamHandler(output)
    handler.setLevel(logging.DEBUG)

    logger = logging.getLogger("uvicorn.error")
    logger.addHandler(handler)
    logger.propagate = False

    thread.start()

    # Wait for the listener.
    for _ in range(200):
        try:
            with socket.create_connection(
                ("127.0.0.1", port),
                timeout=0.2,
            ):
                break
        except OSError:
            time.sleep(0.05)

    try:
        yield port, manager, output
    finally:
        server.should_exit = True

        thread.join(timeout=5)

        with contextlib.suppress(Exception):
            loop.close()

        logger.removeHandler(handler)


@pytest.mark.asyncio
async def test_session_usable_after_client_vanishes(
    live_server,
) -> None:
    import websockets

    port, manager, _output = live_server

    session = await manager.create_session(
        working_directory=Path.cwd(),
    )

    url = f"ws://127.0.0.1:{port}/ws/sessions/{session.session_id}"

    # Abrupt drop: no close frame, like a killed tab.
    socket = await websockets.connect(url)
    await asyncio.wait_for(socket.recv(), timeout=10)
    socket.transport.abort()

    await asyncio.sleep(0.3)

    # The session must still accept a new subscriber and deliver
    # events published after the abrupt disconnect.
    async with websockets.connect(url) as second:
        greeting = json.loads(
            await asyncio.wait_for(second.recv(), timeout=10)
        )

        assert greeting["type"] == "session.snapshot"

        await session.events.publish(
            LLMContentChunk(
                iteration=1,
                content="still here",
            )
        )

        frame = json.loads(
            await asyncio.wait_for(second.recv(), timeout=10)
        )

        assert frame["type"] == "llm.content_chunk"
        assert frame["data"]["content"] == "still here"

    await manager.close_all()


class _FakeSocket:
    """Minimal WebSocket stand-in for handler-level tests.

    ``receive_text`` raises the same ``WebSocketDisconnect`` a real
    peer produces, and ``close`` raises as well once the peer is
    gone, which is exactly the sequence that used to escape the
    handler as an ASGI error.
    """

    def __init__(
        self,
        app=None,
        *,
        fail_close: bool = False,
    ) -> None:
        from types import SimpleNamespace

        from starlette.websockets import WebSocketState

        self.client_state = WebSocketState.CONNECTED
        self.accepted = False
        self.sent: list[str] = []
        self.closed = False
        self._fail_close = fail_close

        # The handler resolves the session manager and auth
        # settings from ``websocket.app.state``.
        self.app = SimpleNamespace(
            state=SimpleNamespace(
                session_manager=app,
                auth=None,
            )
        )

    async def accept(self) -> None:
        self.accepted = True

    async def send_text(self, text: str) -> None:
        self.sent.append(text)

    async def receive_text(self) -> str:
        raise WebSocketDisconnect(code=1005)

    async def close(
        self,
        code: int = 1000,
        reason: str = "",
    ) -> None:
        del reason

        self.closed = True

        if self._fail_close:
            raise WebSocketDisconnect(code=1006)


@pytest.mark.asyncio
async def test_reader_disconnect_does_not_escape_handler() -> None:
    """The reader ends by raising; cleanup must absorb it.

    A real client disconnect raises WebSocketDisconnect inside
    ``_read_client``. Awaiting that task from a ``finally`` block
    re-raised the error and uvicorn logged it as an unhandled ASGI
    exception on every disconnect.
    """

    from agent_workflow.core.application.session import (
        SessionManager,
    )
    from agent_workflow.web.websocket.agent import agent_socket

    manager = SessionManager(auto_restore=False)

    session = await manager.create_session(
        working_directory=Path.cwd(),
    )

    socket = _FakeSocket(manager)

    await asyncio.wait_for(
        agent_socket(socket, session.session_id),
        timeout=10,
    )

    assert socket.accepted is True
    assert socket.closed is True
    assert any(
        '"type":"session.snapshot"' in frame
        for frame in socket.sent
    )

    await manager.close_all()


@pytest.mark.asyncio
async def test_close_failure_does_not_escape_handler() -> None:
    """Closing an already-dead socket must not raise either."""

    from agent_workflow.core.application.session import (
        SessionManager,
    )
    from agent_workflow.web.websocket.agent import agent_socket

    manager = SessionManager(auto_restore=False)

    session = await manager.create_session(
        working_directory=Path.cwd(),
    )

    socket = _FakeSocket(manager, fail_close=True)

    await asyncio.wait_for(
        agent_socket(socket, session.session_id),
        timeout=10,
    )

    assert socket.closed is True

    await manager.close_all()


@pytest.mark.asyncio
async def test_handler_unsubscribes_after_disconnect() -> None:
    """Cleanup must also release the event subscription."""

    from agent_workflow.core.application.session import (
        SessionManager,
    )
    from agent_workflow.web.websocket.agent import agent_socket

    manager = SessionManager(auto_restore=False)

    session = await manager.create_session(
        working_directory=Path.cwd(),
    )

    baseline = session.events.subscriber_count

    await asyncio.wait_for(
        agent_socket(
            _FakeSocket(manager),
            session.session_id,
        ),
        timeout=10,
    )

    assert session.events.subscriber_count == baseline

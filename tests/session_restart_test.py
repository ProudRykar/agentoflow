"""A restart must not lose the sidebar or the transcript.

The session registry survived a restart but the chat did not: the
transcript lived only in the event bus, and recorded sessions were
excluded from ``GET /api/sessions``. These tests pin the behaviour
end to end through a real ``SessionManager``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from agent_workflow.core.application.event_store import EventStore
from agent_workflow.core.application.session import SessionManager
from agent_workflow.core.application.session_store import SessionStore
from agent_workflow.core.entities.models.agent_trace import (
    AgentFinished,
    AgentStarted,
    LLMContentChunk,
    LLMResponded,
    ToolFinished,
    ToolStarted,
)
from conftest import session_manager_factory


def _manager(
    registry_path: Path,
    events_path: Path,
    state_path: Path | None = None,
) -> SessionManager:
    return session_manager_factory()(
        registry_path,
        events_path,
        state_path,
    )


async def _converse(session, prompt: str, answer: str) -> None:
    """Publish the event sequence one run produces."""

    bus = session.events

    await bus.publish(AgentStarted(prompt=prompt, run_id="r1"))
    await bus.publish(
        LLMResponded(
            iteration=1,
            content=answer,
            thinking=None,
            tool_call_count=1,
        )
    )
    await bus.publish(
        ToolStarted(
            iteration=1,
            tool_call_id="c1",
            tool_name="read_file",
            arguments={"path": "a.txt"},
        )
    )
    await bus.publish(
        ToolFinished(
            iteration=1,
            tool_call_id="c1",
            tool_name="read_file",
            output="contents",
            error_code=None,
            error_message=None,
        )
    )
    await bus.publish(
        LLMContentChunk(iteration=2, content=answer)
    )
    await bus.publish(AgentFinished(result=answer, run_id="r1"))


# ======================================================================
# Replay after a restart
# ======================================================================


async def test_transcript_survives_a_restart(
    stub_runtime,
    registry_path: Path,
    events_path: Path,
) -> None:
    first = _manager(registry_path, events_path)

    session = await first.create_session(
        metadata={"label": "demo"}
    )
    session_id = session.session_id

    await _converse(session, "what is 2+2?", "four")
    session.flush_events()

    await first.close_all()

    # A brand new manager, as after a process restart.
    second = _manager(registry_path, events_path)

    assert second.has_session(session_id)

    restored = await second.get_session(session_id)

    assert restored is not None
    assert restored.metadata == {"label": "demo"}

    replayed = restored.events.replay_since(0)

    # Everything the client needs to rebuild the chat, in order.
    assert [type(item.event).__name__ for item in replayed] == [
        "AgentStarted",
        "LLMResponded",
        "ToolStarted",
        "ToolFinished",
        "LLMContentChunk",
        "AgentFinished",
    ]
    assert replayed[0].event == AgentStarted(
        prompt="what is 2+2?",
        run_id="r1",
    )
    assert replayed[-1].event == AgentFinished(
        result="four",
        run_id="r1",
    )

    # Sequence numbers continue instead of restarting, so a
    # reconnecting client's cursor still means something.
    assert restored.events.seq == 6

    await second.close_all()


async def test_conversation_window_is_rebuilt(
    stub_runtime,
    registry_path: Path,
    events_path: Path,
) -> None:
    first = _manager(registry_path, events_path)

    session = await first.create_session()
    session_id = session.session_id

    await _converse(session, "first question", "first answer")
    session.flush_events()

    await first.close_all()

    second = _manager(registry_path, events_path)
    restored = await second.get_session(session_id)

    dialogue = restored.runtime.agent.context_manager.dialogue()

    assert [message["role"] for message in dialogue] == [
        "user",
        "assistant",
        "tool",
        "assistant",
    ]

    # Tool traffic survives the restart, so the model still has what
    # it read.
    assert dialogue[1]["tool_calls"][0]["id"] == "c1"
    assert dialogue[2]["tool_call_id"] == "c1"
    assert dialogue[2]["content"] == "contents"

    # The transcript is loaded into the live window, not appended to
    # a later prompt, so the next turn continues the conversation.
    assert restored.has_history is True

    await second.close_all()


async def test_two_turns_survive(
    stub_runtime,
    registry_path: Path,
    events_path: Path,
) -> None:
    first = _manager(registry_path, events_path)

    session = await first.create_session()
    session_id = session.session_id

    await _converse(session, "one", "first")
    session.flush_events()
    await _converse(session, "two", "second")
    session.flush_events()

    await first.close_all()

    second = _manager(registry_path, events_path)
    restored = await second.get_session(session_id)

    replayed = restored.events.replay_since(0)

    assert len(replayed) == 12
    assert replayed[-1].event.result == "second"

    await second.close_all()


async def test_session_titles_survive(
    stub_runtime,
    registry_path: Path,
    events_path: Path,
) -> None:
    first = _manager(registry_path, events_path)

    session = await first.create_session()
    session_id = session.session_id

    await first.rename_session(session_id, "My chat")

    await first.close_all()

    second = _manager(registry_path, events_path)
    restored = await second.get_session(session_id)

    assert restored.title == "My chat"
    assert restored.title_source == "manual"

    await second.close_all()


# ======================================================================
# Deleting a chat
# ======================================================================


async def test_deleting_a_session_drops_its_transcript(
    stub_runtime,
    registry_path: Path,
    events_path: Path,
) -> None:
    manager = _manager(registry_path, events_path)

    session = await manager.create_session()
    session_id = session.session_id

    await _converse(session, "bye", "later")
    session.flush_events()

    assert EventStore(events_path).count(session_id) == 6

    await manager.close_session(session_id)

    assert EventStore(events_path).count(session_id) == 0
    assert manager.has_session(session_id) is False


async def test_shutdown_keeps_the_transcript(
    stub_runtime,
    registry_path: Path,
    events_path: Path,
) -> None:
    manager = _manager(registry_path, events_path)

    session = await manager.create_session()
    session_id = session.session_id

    await _converse(session, "keep me", "sure")
    session.flush_events()

    await manager.close_all()

    assert EventStore(events_path).count(session_id) == 6


# ======================================================================
# Sessions without a store
# ======================================================================


async def test_manager_without_event_store_still_works(
    stub_runtime,
    registry_path: Path,
) -> None:
    manager = SessionManager(
        store=SessionStore(registry_path),
        auto_restore=True,
    )
    manager.set_event_store(None)

    session = await manager.create_session()

    await session.events.publish(AgentStarted(prompt="x"))

    # Nothing persisted, nothing raised: a session without durable
    # storage behaves exactly as it always did.
    assert session.events.replay_since(0)

    await manager.close_all()


# ======================================================================
# The listing endpoint
# ======================================================================


async def test_recorded_sessions_are_listed(
    stub_runtime,
    registry_path: Path,
    events_path: Path,
    client: TestClient,
) -> None:
    """Regression: GET /api/sessions returned live runtimes only, so
    the sidebar emptied on every restart."""

    first = _manager(registry_path, events_path)
    client.app.state.session_manager = first

    created = client.post("/api/sessions").json()["session_id"]

    await _converse(
        first._sessions[created],
        "remember this",
        "done",
    )
    first._sessions[created].flush_events()

    await first.close_all()

    # Fresh manager: nothing is live any more.
    second = _manager(registry_path, events_path)
    client.app.state.session_manager = second

    body = client.get("/api/sessions").json()

    listed = [item["session_id"] for item in body]

    assert created in listed

    [entry] = [item for item in body if item["session_id"] == created]

    assert entry["session"]["state"] == "idle"
    assert entry["subscriber_count"] == 0

    # The sequence counter survives too, so the client knows it has
    # a transcript to replay.
    assert entry["latest_seq"] == 6


async def test_opening_a_listed_session_rehydrates(
    stub_runtime,
    registry_path: Path,
    events_path: Path,
    client: TestClient,
) -> None:
    first = _manager(registry_path, events_path)
    client.app.state.session_manager = first

    created = client.post("/api/sessions").json()["session_id"]

    await _converse(
        first._sessions[created],
        "question",
        "answer",
    )
    first._sessions[created].flush_events()

    await first.close_all()

    second = _manager(registry_path, events_path)
    client.app.state.session_manager = second

    body = client.get(f"/api/sessions/{created}").json()

    assert body["session_id"] == created
    assert body["latest_seq"] == 6

    replayed = second._sessions[created].events.replay_since(0)

    assert replayed[0].event.prompt == "question"

    await second.close_all()
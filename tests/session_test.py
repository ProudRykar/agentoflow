from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from agent_workflow.core.application.events import EventBus
from agent_workflow.core.application.session import (
    SessionManager,
    SessionState,
)


@pytest.fixture
def manager(tmp_path: Path) -> SessionManager:
    """Isolated manager: no on-disk registry, no rehydration."""

    return SessionManager(auto_restore=False)


async def _dispose(manager: SessionManager) -> None:
    await manager.close_all()


@pytest.mark.asyncio
async def test_create_session(manager: SessionManager) -> None:
    session = await manager.create_session()

    try:
        assert session.session_id
        assert session.state is SessionState.IDLE
        assert session.agent is not None
        assert session.runtime.agent is session.agent
    finally:
        await _dispose(manager)


@pytest.mark.asyncio
async def test_get_session(manager: SessionManager) -> None:
    session = await manager.create_session()

    try:
        found = await manager.get_session(session.session_id)

        assert found is session
    finally:
        await _dispose(manager)


async def test_get_unknown_session(
    manager: SessionManager,
) -> None:
    assert await manager.get_session("missing") is None


@pytest.mark.asyncio
async def test_close_session(
    manager: SessionManager,
    tmp_path: Path,
) -> None:
    session = await manager.create_session(
        working_directory=tmp_path,
    )

    session_id = session.session_id

    assert await manager.close_session(session_id) is True
    assert await manager.get_session(session_id) is None
    assert session.state is SessionState.CLOSED


@pytest.mark.asyncio
async def test_close_unknown_session(
    manager: SessionManager,
) -> None:
    assert await manager.close_session("missing") is False


@pytest.mark.asyncio
async def test_list_sessions(
    manager: SessionManager,
    tmp_path: Path,
) -> None:
    first = await manager.create_session(
        working_directory=tmp_path,
    )
    second = await manager.create_session(
        working_directory=tmp_path,
    )

    try:
        sessions = manager.list_sessions()

        assert len(sessions) == 2
        assert first in sessions
        assert second in sessions
        assert len(manager) == 2
    finally:
        await _dispose(manager)


@pytest.mark.asyncio
async def test_sessions_are_independent(
    manager: SessionManager,
    tmp_path: Path,
) -> None:
    first = await manager.create_session(
        working_directory=tmp_path,
    )
    second = await manager.create_session(
        working_directory=tmp_path,
    )

    try:
        assert first.session_id != second.session_id
        assert first.agent is not second.agent
        assert first.events is not second.events
        assert (
            first.context.permissions
            is not second.context.permissions
        )
    finally:
        await _dispose(manager)


@pytest.mark.asyncio
async def test_events_do_not_leak_between_sessions(
    manager: SessionManager,
    tmp_path: Path,
) -> None:
    first = await manager.create_session(
        working_directory=tmp_path,
    )
    second = await manager.create_session(
        working_directory=tmp_path,
    )

    received: list[str] = []

    async def on_event(seq, event) -> None:
        received.append(type(event).__name__)

    first.events.subscribe(on_event)

    try:
        await second.events.publish(
            _agent_started("second")
        )

        assert received == []

        await first.events.publish(
            _agent_started("first")
        )

        assert received == ["AgentStarted"]
    finally:
        await _dispose(manager)


@pytest.mark.asyncio
async def test_metadata_is_stored(
    manager: SessionManager,
    tmp_path: Path,
) -> None:
    session = await manager.create_session(
        metadata={"project": "demo"},
        working_directory=tmp_path,
    )

    try:
        assert session.metadata == {"project": "demo"}
    finally:
        await _dispose(manager)


@pytest.mark.asyncio
async def test_snapshot_reports_runtime_state(
    manager: SessionManager,
    tmp_path: Path,
) -> None:
    session = await manager.create_session(
        working_directory=tmp_path,
    )

    try:
        snapshot = session.snapshot()

        assert snapshot["session_id"] == session.session_id
        assert snapshot["state"] == "idle"
        assert snapshot["running"] is False
        assert snapshot["working_directory"] == str(
            tmp_path.resolve()
        )
        assert snapshot["approval"] is None
        assert snapshot["model"]
    finally:
        await _dispose(manager)


@pytest.mark.asyncio
async def test_close_all_closes_everything(
    manager: SessionManager,
    tmp_path: Path,
) -> None:
    await manager.create_session(working_directory=tmp_path)
    await manager.create_session(working_directory=tmp_path)

    await manager.close_all()

    assert len(manager) == 0


def _agent_started(prompt: str):
    from agent_workflow.core.entities.models.agent_trace import (
        AgentStarted,
    )

    return AgentStarted(prompt=prompt)

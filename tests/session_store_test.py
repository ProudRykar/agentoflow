from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_workflow.core.application.session import SessionManager
from agent_workflow.core.application.session_store import (
    SessionRecord,
    SessionStore,
)


def _record(session_id: str = "s1") -> SessionRecord:
    return SessionRecord(
        session_id=session_id,
        working_directory="/tmp",
        created_at=1000.0,
        metadata={"project": "demo"},
    )


def test_missing_file_loads_empty(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions.json")

    assert store.load() == []


def test_round_trip(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions.json")

    store.save([_record()])

    [loaded] = store.load()

    assert loaded.session_id == "s1"
    assert loaded.working_directory == "/tmp"
    assert loaded.created_at == 1000.0
    assert loaded.metadata == {"project": "demo"}


def test_upsert_replaces_existing(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions.json")

    store.upsert(_record())
    store.upsert(
        SessionRecord(
            session_id="s1",
            working_directory="/var",
            created_at=2000.0,
            metadata={},
        )
    )

    records = store.load()

    assert len(records) == 1
    assert records[0].working_directory == "/var"
    assert records[0].created_at == 2000.0


def test_remove_drops_record(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions.json")

    store.upsert(_record("a"))
    store.upsert(_record("b"))

    store.remove("a")

    assert [r.session_id for r in store.load()] == ["b"]


def test_remove_unknown_is_noop(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "sessions.json")

    store.upsert(_record("a"))
    store.remove("zzz")

    assert len(store.load()) == 1


def test_corrupt_file_is_tolerated(tmp_path: Path) -> None:
    path = tmp_path / "sessions.json"
    path.write_text("{not json", encoding="utf-8")

    assert SessionStore(path).load() == []


def test_non_list_payload_is_tolerated(tmp_path: Path) -> None:
    path = tmp_path / "sessions.json"
    path.write_text('{"session_id": "s1"}', encoding="utf-8")

    assert SessionStore(path).load() == []


def test_entries_without_id_are_skipped(
    tmp_path: Path,
) -> None:
    path = tmp_path / "sessions.json"
    path.write_text(
        json.dumps([{"working_directory": "/tmp"}, {"session_id": "ok"}]),
        encoding="utf-8",
    )

    [loaded] = SessionStore(path).load()

    assert loaded.session_id == "ok"
    assert loaded.working_directory == ""


def test_write_is_atomic_no_temp_left(
    tmp_path: Path,
) -> None:
    store = SessionStore(tmp_path / "sessions.json")

    store.save([_record()])

    leftovers = [
        item.name
        for item in tmp_path.iterdir()
        if item.name != "sessions.json"
    ]

    assert leftovers == []


def test_creates_parent_directory(tmp_path: Path) -> None:
    store = SessionStore(tmp_path / "nested" / "deep" / "sessions.json")

    store.save([_record()])

    assert store.load()[0].session_id == "s1"


@pytest.mark.asyncio
async def test_manager_restores_recorded_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A recorded session is rebuilt on first access."""

    from agent_workflow.core.application import session as session_module

    sandbox = tmp_path / "work"
    sandbox.mkdir()

    store = SessionStore(tmp_path / "sessions.json")

    store.save(
        [
            SessionRecord(
                session_id="persisted",
                working_directory=str(sandbox),
                created_at=123.0,
                metadata={"project": "demo"},
            )
        ]
    )

    original = session_module.create_runtime

    async def fake(
        approval_handler,
        working_directory=None,
        stats_store=None,
    ):
        return await original(
            approval_handler,
            working_directory,
            stats_store,
        )

    monkeypatch.setattr(session_module, "create_runtime", fake)

    manager = SessionManager(store=store)

    assert manager.has_session("persisted")
    assert manager.list_sessions() == []

    session = await manager.get_session("persisted")

    assert session is not None
    assert session.session_id == "persisted"
    assert session.metadata == {"project": "demo"}
    assert session.created_at == 123.0
    assert session.working_directory == sandbox

    # A second access returns the same live object.
    assert await manager.get_session("persisted") is session

    await manager.close_all()


@pytest.mark.asyncio
async def test_manager_forgets_record_without_restore_flag(
    tmp_path: Path,
) -> None:
    store = SessionStore(tmp_path / "sessions.json")

    store.save([_record("persisted")])

    manager = SessionManager(
        store=store,
        auto_restore=False,
    )

    assert manager.has_session("persisted") is False


@pytest.mark.asyncio
async def test_manager_drops_record_with_missing_directory(
    tmp_path: Path,
) -> None:
    store = SessionStore(tmp_path / "sessions.json")

    store.save(
        [
            SessionRecord(
                session_id="stale",
                working_directory=str(tmp_path / "gone"),
                created_at=1.0,
                metadata={},
            )
        ]
    )

    manager = SessionManager(store=store)

    session = await manager.get_session("stale")

    # Falls back to the default working directory rather than
    # serving a session rooted in a deleted path.
    assert session is not None
    assert session.working_directory != tmp_path / "gone"

    await manager.close_all()


@pytest.mark.asyncio
async def test_close_all_keeps_registry(
    tmp_path: Path,
) -> None:
    store = SessionStore(tmp_path / "sessions.json")

    manager = SessionManager(store=store)

    session = await manager.create_session(
        metadata={"a": 1},
        working_directory=tmp_path,
    )

    await manager.close_all()

    assert len(manager) == 0
    assert [r.session_id for r in store.load()] == [
        session.session_id
    ]


@pytest.mark.asyncio
async def test_close_session_removes_record(
    tmp_path: Path,
) -> None:
    store = SessionStore(tmp_path / "sessions.json")

    manager = SessionManager(store=store)

    session = await manager.create_session(
        working_directory=tmp_path,
    )

    assert await manager.close_session(session.session_id) is True
    assert store.load() == []


@pytest.mark.asyncio
async def test_list_records_includes_pending(
    tmp_path: Path,
) -> None:
    store = SessionStore(tmp_path / "sessions.json")

    store.save(
        [
            SessionRecord(
                session_id="old",
                working_directory="",
                created_at=1.0,
                metadata={},
            )
        ]
    )

    manager = SessionManager(store=store)

    live = await manager.create_session(
        working_directory=tmp_path,
    )

    ids = [record.session_id for record in manager.list_records()]

    assert "old" in ids
    assert live.session_id in ids
    assert ids[0] == "old"

    await manager.close_all()

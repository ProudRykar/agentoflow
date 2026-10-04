"""Working state must survive a restart, not just the transcript.

A restored session used to come back with its dialogue but an empty
anchor, checkpoint and evidence store, and the next
``continue_run`` minted a new task id that no longer matched the
restored conversation.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from agent_workflow.core.application.session_state_store import (
    MAX_RESTORED_EVIDENCE,
    SessionStateStore,
)
from agent_workflow.core.context.checkpoint import TaskCheckpoint
from agent_workflow.core.context.evidence import EvidenceStore
from agent_workflow.core.context.task_anchor import TaskAnchor
from agent_workflow.core.entities.models.agent_orchestrator import (
    AgentOrchestrator,
)
from agent_workflow.core.entities.models.agent_trace import AgentStarted
from agent_workflow.core.entities.models.planner import Planner
from agent_workflow.core.entities.models.research_contract import (
    ResearchPage,
)
from agent_workflow.core.entities.models.task_contract import TaskContract

from conftest import session_manager_factory


def _orchestrator() -> AgentOrchestrator:
    orchestrator = AgentOrchestrator()
    orchestrator.prepare_task(
        Planner().plan("Crawl the docs"),
        TaskContract(),
        prompt="Crawl the docs",
    )
    orchestrator.on_agent_started("Crawl the docs")

    return orchestrator


def _page(url: str, body: str) -> ResearchPage:
    return ResearchPage(
        url=url,
        depth=0,
        title=url,
        content=body,
        links=(),
        content_bytes=len(body),
    )


# ======================================================================
# Store round trips
# ======================================================================


def test_anchor_round_trip(tmp_path: Path) -> None:
    store = SessionStateStore(tmp_path / "state.db")

    anchor = _orchestrator().task_anchor

    assert anchor is not None

    store.save("s1", anchor=anchor)

    assert store.anchor("s1") == anchor


def test_checkpoint_round_trip(tmp_path: Path) -> None:
    store = SessionStateStore(tmp_path / "state.db")

    checkpoint = TaskCheckpoint(
        task_id="task-1",
        objective="crawl",
        completed_steps=("step-1",),
    )

    store.save("s1", checkpoint=checkpoint)

    assert store.checkpoint("s1") == checkpoint


def test_evidence_round_trip(tmp_path: Path) -> None:
    store = SessionStateStore(tmp_path / "state.db")

    evidence = EvidenceStore()
    page = evidence.append_page(
        _page("https://example.test/a", "body a"),
        "https://example.test",
    )

    store.save("s1", evidence=evidence)

    restored = store.evidence("s1")

    assert restored is not None
    assert restored.get(page.evidence_id) is not None


def test_saving_one_part_keeps_the_others(tmp_path: Path) -> None:
    # A caller holding only an anchor must not erase the checkpoint.
    store = SessionStateStore(tmp_path / "state.db")

    anchor = _orchestrator().task_anchor
    checkpoint = TaskCheckpoint(task_id="t", objective="o")

    store.save("s1", anchor=anchor, checkpoint=checkpoint)
    store.save("s1", anchor=anchor)

    assert store.checkpoint("s1") == checkpoint


def test_unknown_session_has_no_state(tmp_path: Path) -> None:
    store = SessionStateStore(tmp_path / "state.db")

    assert store.anchor("nope") is None
    assert store.checkpoint("nope") is None
    assert store.evidence("nope") is None
    assert store.load("nope") == {}


def test_corrupt_payload_is_ignored(tmp_path: Path) -> None:
    store = SessionStateStore(tmp_path / "state.db")

    import sqlite3

    store.save("s1", anchor=_orchestrator().task_anchor)

    with sqlite3.connect(tmp_path / "state.db") as connection:
        connection.execute(
            "UPDATE session_state SET payload = ?",
            ("{not json",),
        )

    assert store.load("s1") == {}
    assert store.anchor("s1") is None


def test_forget_removes_state(tmp_path: Path) -> None:
    store = SessionStateStore(tmp_path / "state.db")

    store.save("s1", anchor=_orchestrator().task_anchor)
    store.forget("s1")

    assert store.load("s1") == {}
    assert store.session_ids() == []


def test_restored_evidence_is_capped(tmp_path: Path) -> None:
    store = SessionStateStore(tmp_path / "state.db")

    evidence = EvidenceStore(max_items=10_000)

    for index in range(MAX_RESTORED_EVIDENCE + 50):
        evidence.append_page(
            _page(f"https://example.test/{index}", "body"),
            "https://example.test",
        )

    store.save("s1", evidence=evidence)

    restored = store.evidence("s1")

    assert restored is not None
    assert len(restored) <= MAX_RESTORED_EVIDENCE


# ======================================================================
# End to end through the manager
# ======================================================================


async def test_working_state_survives_a_restart(
    stub_runtime,
    registry_path: Path,
    events_path: Path,
    state_path: Path,
) -> None:
    build = session_manager_factory()

    first = build(registry_path, events_path, state_path)

    session = await first.create_session()

    orchestrator = _orchestrator()

    evidence = orchestrator.evidence_store
    evidence.append_page(
        _page("https://example.test/doc", "IMPORTANT PAGE"),
        "https://example.test",
    )

    session.runtime.agent.orchestrator.restore_working_state(
        anchor=orchestrator.task_anchor,
        evidence=evidence,
    )

    await session.events.publish(
        AgentStarted(prompt="Crawl the docs")
    )
    session.flush_events()
    session.flush_working_state()

    task_id = orchestrator.task_anchor.task_id

    assert task_id is not None

    await first.close_all()

    second = build(registry_path, events_path, state_path)

    restored = await second.get_session(session.session_id)

    live = restored.runtime.agent.orchestrator

    # The anchor came back, so the restored transcript still belongs
    # to a task instead of floating free.
    assert live.task_anchor is not None
    assert live.task_anchor.task_id == task_id
    assert live.task_anchor.objective == (
        orchestrator.task_anchor.objective
    )

    pages = [item.content for item in live.evidence_store.all()]

    assert any("IMPORTANT PAGE" in body for body in pages)

    await second.close_all()


async def test_restored_session_keeps_its_task_on_continue(
    stub_runtime,
    registry_path: Path,
    events_path: Path,
    state_path: Path,
) -> None:
    """Regression: continue_run forced a new anchor, so the task id
    changed under a restored conversation."""

    build = session_manager_factory()

    first = build(registry_path, events_path, state_path)

    session = await first.create_session()

    orchestrator = _orchestrator()

    session.runtime.agent.orchestrator.restore_working_state(
        anchor=orchestrator.task_anchor,
    )

    await session.events.publish(
        AgentStarted(prompt="Crawl the docs")
    )
    session.flush_events()
    session.flush_working_state()

    original = orchestrator.task_anchor.task_id

    await first.close_all()

    second = build(registry_path, events_path, state_path)

    restored = await second.get_session(session.session_id)

    live = restored.runtime.agent.orchestrator

    assert live.task_anchor is not None
    assert live.task_anchor.task_id == original

    await second.close_all()


async def test_deleting_a_session_drops_its_working_state(
    stub_runtime,
    registry_path: Path,
    events_path: Path,
    state_path: Path,
) -> None:
    manager = session_manager_factory()(
        registry_path,
        events_path,
        state_path,
    )

    session = await manager.create_session()

    session.runtime.agent.orchestrator.restore_working_state(
        anchor=_orchestrator().task_anchor,
    )

    session.flush_working_state()

    store = SessionStateStore(state_path)

    assert store.anchor(session.session_id) is not None

    await manager.close_session(session.session_id)

    assert store.load(session.session_id) == {}

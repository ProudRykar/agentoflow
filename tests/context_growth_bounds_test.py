"""Bounded growth for evidence and history.

Both stores grew for the life of a session with no eviction at all:
evidence kept every crawled page body, history every tool result.
A long crawl or a long tool loop therefore grew the process without
limit while the assembler was already trimming what it sent.
"""

from __future__ import annotations

import pytest

from agent_workflow.core.context.evidence import EvidenceStore
from agent_workflow.core.context.history import (
    HistoryKind,
    InMemoryHistoryStore,
    MAX_HISTORY_CONTENT_CHARS,
)
from agent_workflow.core.entities.models.agent_orchestrator import (
    AgentOrchestrator,
)
from agent_workflow.core.entities.models.planner import Planner
from agent_workflow.core.entities.models.research_contract import (
    ResearchPage,
)
from agent_workflow.core.entities.models.task_contract import TaskContract


def _page(
    body: str = "body",
) -> ResearchPage:
    return ResearchPage(
        url="https://example.test/page",
        title="Page",
        content=body,
        depth=0,
        links=(),
        content_bytes=len(body),
    )


# ======================================================================
# Evidence
# ======================================================================


def test_evidence_store_is_bounded() -> None:
    store = EvidenceStore(max_items=25)

    for index in range(200):
        store.append_page(_page(f"page {index}"), "https://example.test")

    assert len(store) == 25

    # The newest pages are the ones kept.
    kept = [item.content for item in store.all()]

    assert "page 199" in kept[-1]
    assert "page 0" not in kept


def test_evidence_keeps_evidence_referenced_by_a_checkpoint() -> None:
    store = EvidenceStore(max_items=10)

    first = store.append_page(_page("important"), "https://example.test")

    # A checkpoint cites it, so it is pinned while it still exists.
    store.pin(first.evidence_id)

    for index in range(50):
        store.append_page(
            _page(f"filler {index}"),
            "https://example.test",
        )

    assert store.get(first.evidence_id) is not None

    # The soft limit applies to unpinned evidence: 10 unpinned plus
    # the one pinned on top.
    assert len(store) == 11


def test_evidence_pinning_respects_a_hard_ceiling() -> None:
    store = EvidenceStore(max_items=5)

    pinned = [
        store.append_page(
            _page(f"pinned {index}"),
            "https://example.test",
        ).evidence_id
        for index in range(5)
    ]

    for evidence_id in pinned:
        store.pin(evidence_id)

    for index in range(100):
        store.append_page(
            _page(f"filler {index}"),
            "https://example.test",
        )

    # Pinned items may exceed the soft limit, but not without bound.
    assert len(store) <= store.hard_max_items

    # The newest filler still made it in.
    assert store.all()[-1].content == "filler 99"


def test_evidence_default_limit_is_finite() -> None:
    store = EvidenceStore()

    assert store.max_items > 0
    assert store.hard_max_items >= store.max_items


def test_evidence_serialisation_round_trip_keeps_the_limit() -> None:
    store = EvidenceStore(max_items=4)

    for index in range(10):
        store.append_page(_page(f"page {index}"), "https://example.test")

    restored = EvidenceStore.from_dict(store.to_dict())

    assert len(restored) == len(store)
    assert restored.max_items == store.max_items


# ======================================================================
# History
# ======================================================================


def test_history_store_is_bounded() -> None:
    store = InMemoryHistoryStore(max_items=30)

    for index in range(500):
        store.append(
            task_id="task-1",
            run_id="run-1",
            window_id="window-01",
            kind=HistoryKind.TOOL_RESULT,
            content=f"result {index}",
            reference=None,
        )

    assert len(store) == 30

    # Newest kept.
    items = store.window_items("task-1", "window-01")

    assert items[-1].content == "result 499"


def test_history_default_limit_is_finite() -> None:
    store = InMemoryHistoryStore()

    assert store.max_items > 0


def test_history_still_truncates_content() -> None:
    store = InMemoryHistoryStore()

    item = store.append(
        task_id="task-1",
        run_id="run-1",
        window_id="window-01",
        kind=HistoryKind.TOOL_RESULT,
        content="x" * (MAX_HISTORY_CONTENT_CHARS * 3),
        reference=None,
    )

    assert len(item.content) <= MAX_HISTORY_CONTENT_CHARS + 64


@pytest.mark.parametrize("limit", [1, 5, 100])
def test_history_eviction_keeps_recent_items(limit: int) -> None:
    store = InMemoryHistoryStore(max_items=limit)

    for index in range(limit * 3):
        store.append(
            task_id="task-1",
            run_id="run-1",
            window_id="window-01",
            kind=HistoryKind.USER_MESSAGE,
            content=f"turn {index}",
            reference=None,
        )

    contents = [
        item.content
        for item in store.window_items("task-1", "window-01")
    ]

    assert len(contents) == limit
    assert contents[-1] == f"turn {limit * 3 - 1}"

# ======================================================================
# Plan steps in the rendered state
# ======================================================================


def test_plan_step_rendering_is_bounded() -> None:
    from agent_workflow.core.context.context_assembler import (
        ContextAssembler,
    )
    from agent_workflow.core.entities.models.agent_phase import AgentPhase
    from agent_workflow.core.entities.models.agent_plan import AgentPlan
    from agent_workflow.core.entities.models.plan_step import PlanStep

    plan = AgentPlan(
        objective="plan",
        steps=[
            PlanStep(
                id=f"step-{index:03d}",
                description=f"do work {index}",
                phase=AgentPhase.EXECUTION,
            )
            for index in range(300)
        ]
    )

    orchestrator = AgentOrchestrator()
    orchestrator.prepare_task(
        Planner().plan("plan"),
        TaskContract(),
        prompt="plan",
    )

    request = ContextAssembler().build(
        anchor=orchestrator.task_anchor,
        task_state=orchestrator.task_state,
        execution=orchestrator.execution_context,
        conversation_recent=(),
        evidence_selected=(),
        execution_plan=plan,
    )

    state_block = next(
        str(message.get("content", ""))
        for message in request.messages
        if str(message.get("content", "")).startswith("[TASK STATE]")
    )

    assert "plan progress: 0/300 steps completed" in state_block
    assert "and 288 earlier step(s) not shown" in state_block

    # The newest tail is kept; the oldest steps are summarised.
    assert "do work 299" in state_block
    assert "do work 288" in state_block
    assert "do work 0 " not in state_block


def test_active_step_is_always_shown() -> None:
    from agent_workflow.core.context.context_assembler import (
        ContextAssembler,
    )
    from agent_workflow.core.entities.models.agent_phase import AgentPhase
    from agent_workflow.core.entities.models.agent_plan import AgentPlan
    from agent_workflow.core.entities.models.plan_step import (
        PlanStep,
        PlanStepStatus,
    )

    steps = [
        PlanStep(
            id=f"step-{index:03d}",
            description=f"do work {index}",
            phase=AgentPhase.EXECUTION,
            status=PlanStepStatus.PENDING,
        )
        for index in range(100)
    ]

    steps[3].status = PlanStepStatus.ACTIVE

    orchestrator = AgentOrchestrator()
    orchestrator.prepare_task(
        Planner().plan("plan"),
        TaskContract(),
        prompt="plan",
    )

    request = ContextAssembler().build(
        anchor=orchestrator.task_anchor,
        task_state=orchestrator.task_state,
        execution=orchestrator.execution_context,
        conversation_recent=(),
        evidence_selected=(),
        execution_plan=AgentPlan(objective="plan", steps=steps),
    )

    state_block = next(
        str(message.get("content", ""))
        for message in request.messages
        if str(message.get("content", "")).startswith("[TASK STATE]")
    )

    assert "do work 3" in state_block

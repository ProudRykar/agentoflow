import dataclasses

import pytest

from core.context.conversation import ConversationManager
from core.context.execution_context import ExecutionContext
from core.context.task_anchor import TaskAnchor
from core.context.task_state import TaskState
from core.context.tokens import ApproximateTokenCounter
from core.entities.models.agent_orchestrator import AgentOrchestrator
from core.entities.models.agent_phase import AgentPhase
from core.entities.models.context_manager import ContextManager
from core.entities.models.planner import Planner
from core.entities.models.research_contract import ResearchCoverage
from core.entities.models.task_contract import TaskContract
from core.entities.models.task_plan import TaskPlan


def test_task_anchor_is_immutable() -> None:
    anchor = TaskAnchor.create(
        original_prompt="Analyze Textual",
        objective="Textual library",
    )

    assert anchor.task_id
    assert anchor.constraints == ()
    assert anchor.parent_task_id is None

    with pytest.raises(dataclasses.FrozenInstanceError):
        anchor.objective = "changed"  # type: ignore[misc]


def test_task_anchor_rejects_empty_fields() -> None:
    with pytest.raises(ValueError):
        TaskAnchor(
            task_id="",
            original_prompt="prompt",
            objective="objective",
        )

    with pytest.raises(ValueError):
        TaskAnchor.create(
            original_prompt="",
            objective="objective",
        )

    with pytest.raises(ValueError):
        TaskAnchor.create(
            original_prompt="prompt",
            objective="",
        )


def test_task_anchor_serialization_roundtrip() -> None:
    anchor = TaskAnchor.create(
        original_prompt="Analyze Textual",
        objective="Textual library",
        constraints=("use docs",),
    )

    restored = TaskAnchor.from_dict(anchor.to_dict())

    assert restored == anchor


def test_conversation_manager_aliases_context_manager() -> None:
    assert ConversationManager is ContextManager


def test_approximate_token_counter() -> None:
    counter = ApproximateTokenCounter()

    assert counter.count("") == 0
    assert counter.count("a") == 1
    assert counter.count("abcd") == 1
    assert counter.count("abcdefgh") == 2

    with pytest.raises(ValueError):
        ApproximateTokenCounter(divisor=0)


def test_ensure_task_anchor_reuses_prepared_task() -> None:
    orchestrator = AgentOrchestrator()
    plan = Planner().plan("Analyze Textual")

    orchestrator.prepare_task(
        plan,
        TaskContract(),
        prompt="Analyze Textual",
    )

    first = orchestrator.task_anchor
    assert first is not None
    assert first.original_prompt == "Analyze Textual"
    assert first.objective == plan.objective

    second = orchestrator.ensure_task_anchor("Analyze Textual")

    assert second is first


def test_ensure_task_anchor_force_new_replaces() -> None:
    orchestrator = AgentOrchestrator()
    plan = Planner().plan("Analyze Textual")

    orchestrator.prepare_task(
        plan,
        TaskContract(),
        prompt="Analyze Textual",
    )

    first = orchestrator.task_anchor

    second = orchestrator.ensure_task_anchor(
        "Compare with Rich",
        force_new=True,
    )

    assert second is not first
    assert second.original_prompt == "Compare with Rich"


def test_ensure_task_anchor_fallback_without_plan() -> None:
    orchestrator = AgentOrchestrator()

    anchor = orchestrator.ensure_task_anchor("Read test.txt")

    assert anchor.original_prompt == "Read test.txt"
    assert anchor.objective


def test_prepare_task_without_prompt_invalidates_anchor() -> None:
    orchestrator = AgentOrchestrator()
    first_plan = Planner().plan("Analyze Textual")

    orchestrator.prepare_task(
        first_plan,
        TaskContract(),
        prompt="Analyze Textual",
    )

    assert orchestrator.task_anchor is not None

    second_plan = Planner().plan("Compare with Rich")

    orchestrator.prepare_task(
        second_plan,
        TaskContract(),
    )

    assert orchestrator.task_anchor is None


def test_task_state_is_snapshot_not_live_view() -> None:
    orchestrator = AgentOrchestrator()
    plan = Planner().plan("Analyze Textual")

    orchestrator.prepare_task(
        plan,
        TaskContract(),
        prompt="Analyze Textual",
    )
    orchestrator.on_agent_started("Analyze Textual")

    snapshot = orchestrator.task_state

    assert isinstance(snapshot, TaskState)
    assert snapshot.anchor is orchestrator.task_anchor
    assert snapshot.plan == plan

    # Later progress must not mutate the taken snapshot.
    orchestrator.task_progress.research_coverage.fetched_urls.add(
        "https://example.com/",
    )

    assert (
        "https://example.com/"
        not in snapshot.coverage.fetched_urls
    )


def test_task_state_requires_anchor() -> None:
    orchestrator = AgentOrchestrator()

    with pytest.raises(RuntimeError):
        orchestrator.task_state  # noqa: B018


def test_execution_context_reflects_run() -> None:
    orchestrator = AgentOrchestrator()
    orchestrator.set_run_id("run-01")
    orchestrator.on_agent_started("Read test.txt")
    orchestrator.on_llm_requested(3)
    orchestrator.on_tool_started("read_file")

    context = orchestrator.execution_context

    assert isinstance(context, ExecutionContext)
    assert context.run_id == "run-01"
    assert context.iteration == 3
    assert context.tool_calls_used == 1
    assert context.blocked_reason is None


def test_execution_context_blocked_reason() -> None:
    orchestrator = AgentOrchestrator()
    orchestrator.on_agent_started("Read test.txt")
    orchestrator.block("boom")

    context = orchestrator.execution_context

    assert context.phase is AgentPhase.BLOCKED
    assert context.blocked_reason == "boom"


def test_on_agent_started_reuses_prepared_anchor() -> None:
    orchestrator = AgentOrchestrator()
    plan = TaskPlan(objective="custom objective")

    orchestrator.prepare_task(
        plan,
        TaskContract(),
        prompt="Analyze Textual",
    )

    prepared = orchestrator.task_anchor

    orchestrator.on_agent_started("Analyze Textual")

    assert orchestrator.task_anchor is prepared


def test_research_coverage_snapshot_copies_sets() -> None:
    coverage = ResearchCoverage()
    coverage.fetched_urls.add("https://example.com/")

    anchor = TaskAnchor.create(
        original_prompt="prompt",
        objective="objective",
    )

    snapshot = TaskState.snapshot(
        anchor=anchor,
        plan=None,
        contract=TaskContract(),
        coverage=coverage,
    )

    coverage.fetched_urls.add("https://other.example/")

    assert snapshot.coverage.fetched_urls == {
        "https://example.com/"
    }

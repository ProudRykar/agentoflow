import pytest

from core.context.checkpoint import (
    TaskCheckpoint,
    build_checkpoint,
)
from core.entities.models.agent_orchestrator import AgentOrchestrator
from core.entities.models.agent_phase import AgentPhase
from core.entities.models.planner import Planner
from core.entities.models.research_contract import (
    ResearchContract,
    ResearchPage,
    ResearchResult,
)
from core.entities.models.task_contract import TaskContract


def _started_orchestrator(
    prompt: str = "Read test.txt",
) -> AgentOrchestrator:
    orchestrator = AgentOrchestrator()
    orchestrator.prepare_task(
        Planner().plan(prompt),
        TaskContract(),
        prompt=prompt,
    )
    orchestrator.on_agent_started(prompt)

    return orchestrator


def test_checkpoint_copies_objective_from_anchor() -> None:
    orchestrator = _started_orchestrator("Analyze Textual")

    checkpoint = build_checkpoint(orchestrator)

    assert checkpoint.task_id == orchestrator.task_anchor.task_id
    assert checkpoint.objective == orchestrator.task_anchor.objective
    assert checkpoint.objective == "Textual"


def test_checkpoint_tracks_plan_progress() -> None:
    orchestrator = _started_orchestrator()

    checkpoint = build_checkpoint(orchestrator)

    assert checkpoint.active_step == "execution"
    assert checkpoint.completed_steps == ()
    assert any(
        "execution" in action
        for action in checkpoint.next_actions
    )


def test_checkpoint_advance_moves_active_step() -> None:
    orchestrator = _started_orchestrator()
    orchestrator.advance_plan(reason="step done")

    checkpoint = build_checkpoint(orchestrator)

    assert checkpoint.completed_steps == ("execution",)
    assert checkpoint.active_step == "synthesis"


def test_checkpoint_reports_missing_requirements() -> None:
    prompt = "Research https://example.com docs"
    plan = Planner().plan(prompt)

    orchestrator = AgentOrchestrator()
    orchestrator.prepare_task(
        plan,
        TaskContract(
            requires_research=True,
            research=plan.research,
        ),
        prompt=prompt,
    )
    orchestrator.on_agent_started(prompt)

    checkpoint = build_checkpoint(orchestrator)

    assert "research" in checkpoint.missing_requirements
    assert checkpoint.research_evidence_ids == ()


def test_checkpoint_collects_evidence_ids() -> None:
    prompt = "Research https://example.com docs"
    plan = Planner().plan(prompt)

    orchestrator = AgentOrchestrator()
    orchestrator.prepare_task(
        plan,
        TaskContract(
            requires_research=True,
            research=plan.research,
        ),
        prompt=prompt,
    )
    orchestrator.on_agent_started(prompt)

    orchestrator.store_research_evidence(
        ResearchResult(
            root_url="https://example.com/",
            pages=(
                ResearchPage(
                    url="https://example.com/",
                    depth=0,
                    title="Example",
                    content="body",
                    links=(),
                    content_bytes=4,
                ),
            ),
            discovered_urls=(),
            failed_urls=(),
            max_depth_reached=0,
            total_bytes=4,
            page_limit_reached=False,
            byte_limit_reached=False,
        )
    )

    checkpoint = build_checkpoint(orchestrator)

    assert checkpoint.research_evidence_ids == ("ev-0001",)


def test_checkpoint_requires_anchor() -> None:
    with pytest.raises(RuntimeError):
        build_checkpoint(AgentOrchestrator())


def test_checkpoint_render_and_roundtrip() -> None:
    orchestrator = _started_orchestrator()

    checkpoint = build_checkpoint(orchestrator)
    text = checkpoint.render()

    assert "objective:" in text
    assert "active step: execution" in text

    restored = TaskCheckpoint.from_dict(
        checkpoint.to_dict()
    )

    assert restored == checkpoint


def test_checkpoint_default_cursors_empty() -> None:
    orchestrator = _started_orchestrator()

    checkpoint = build_checkpoint(orchestrator)

    assert checkpoint.decisions == ()
    assert checkpoint.learnings == ()
    assert checkpoint.conversation_cursor is None
    assert checkpoint.history_window_id is None


def test_checkpoint_missing_phase_after_block() -> None:
    orchestrator = _started_orchestrator()
    orchestrator.block("boom")

    assert (
        orchestrator.execution_context.phase
        is AgentPhase.BLOCKED
    )

    checkpoint = build_checkpoint(orchestrator)

    assert checkpoint.active_step == "execution"

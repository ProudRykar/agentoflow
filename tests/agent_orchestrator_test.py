import pytest

from core.entities.models.agent_orchestrator import (
    AgentOrchestrator,
)
from core.entities.models.agent_phase import AgentPhase


def test_agent_starts_in_planning() -> None:
    orchestrator = AgentOrchestrator()

    transition = orchestrator.on_agent_started("test task")

    # The run passes through PLANNING and lands on the first
    # plan step (execution for a prompt without research).
    assert transition == (
        AgentPhase.PLANNING,
        AgentPhase.EXECUTION,
        "initial plan step activated: execution",
    )

    assert orchestrator.state.started is True
    assert orchestrator.state.phase is AgentPhase.EXECUTION


def test_web_tool_enters_research() -> None:
    from core.entities.models.planner import Planner
    from core.entities.models.task_contract import TaskContract

    prompt = "Research https://example.com documentation"
    task_plan = Planner().plan(prompt)

    orchestrator = AgentOrchestrator()
    orchestrator.prepare_task(
        task_plan,
        TaskContract(
            requires_research=True,
            research=task_plan.research,
        ),
        prompt=prompt,
    )
    orchestrator.on_agent_started(prompt)

    assert orchestrator.state.phase is AgentPhase.RESEARCH

    transition = orchestrator.on_tool_started(
        "web_fetch",
    )

    # Already in the research step phase: no transition needed.
    assert transition is None
    assert orchestrator.state.phase is AgentPhase.RESEARCH
    assert orchestrator.state.tool_calls == 1


def test_regular_tool_enters_execution() -> None:
    orchestrator = AgentOrchestrator()

    orchestrator.on_agent_started("test task")

    transition = orchestrator.on_tool_started(
        "read_file",
    )

    # Already in the execution step phase: no transition needed.
    assert transition is None
    assert orchestrator.state.phase is AgentPhase.EXECUTION
    assert orchestrator.state.tool_calls == 1
    assert orchestrator.state.last_tool_name == "read_file"


def test_failed_tool_enters_debugging() -> None:
    orchestrator = AgentOrchestrator()

    orchestrator.on_agent_started("test task")
    orchestrator.on_tool_started("read_file")

    transition = orchestrator.on_tool_finished(
        error_code="execution_error",
        error_message="boom",
    )

    assert transition is not None
    assert transition[0] is AgentPhase.EXECUTION
    assert transition[1] is AgentPhase.DEBUGGING
    assert orchestrator.state.last_tool_succeeded is False
    assert orchestrator.state.last_error == "boom"


def test_candidate_completion_goes_through_synthesis() -> None:
    orchestrator = AgentOrchestrator()

    orchestrator.on_agent_started("test task")

    synthesis = orchestrator.begin_synthesis()

    assert synthesis is not None
    assert synthesis[1] is AgentPhase.SYNTHESIS

    completed = orchestrator.complete()

    assert completed is not None
    assert completed[0] is AgentPhase.SYNTHESIS
    assert completed[1] is AgentPhase.COMPLETED
    assert orchestrator.state.finished is True


def test_completed_run_cannot_continue_without_reset() -> None:
    orchestrator = AgentOrchestrator()

    orchestrator.on_agent_started("test task")
    orchestrator.begin_synthesis()
    orchestrator.complete()

    with pytest.raises(
        RuntimeError,
        match="Invalid agent phase transition",
    ):
        orchestrator.transition(
            AgentPhase.EXECUTION,
            reason="continue",
        )


def test_new_run_resets_previous_state() -> None:
    orchestrator = AgentOrchestrator()

    orchestrator.on_agent_started("test task")
    orchestrator.on_tool_started("web_fetch")
    orchestrator.on_tool_finished(
        error_code="execution_error",
        error_message="failed",
    )
    orchestrator.begin_reflection()

    orchestrator.on_agent_started("test task")

    assert orchestrator.state.phase is AgentPhase.EXECUTION
    assert orchestrator.state.iteration == 0
    assert orchestrator.state.tool_calls == 0
    assert orchestrator.state.last_tool_name is None
    assert orchestrator.state.finished is False
import pytest

from core.entities.models.agent_orchestrator import (
    AgentOrchestrator,
)
from core.entities.models.agent_phase import AgentPhase


def test_agent_starts_in_planning() -> None:
    orchestrator = AgentOrchestrator()

    transition = orchestrator.on_agent_started()

    assert transition == (
        AgentPhase.IDLE,
        AgentPhase.PLANNING,
        "agent run started",
    )

    assert orchestrator.state.started is True
    assert orchestrator.state.phase is AgentPhase.PLANNING


def test_web_tool_enters_research() -> None:
    orchestrator = AgentOrchestrator()

    orchestrator.on_agent_started()

    transition = orchestrator.on_tool_started(
        "web_fetch",
    )

    assert transition is not None
    assert transition[0] is AgentPhase.PLANNING
    assert transition[1] is AgentPhase.RESEARCH
    assert orchestrator.state.tool_calls == 1


def test_regular_tool_enters_execution() -> None:
    orchestrator = AgentOrchestrator()

    orchestrator.on_agent_started()

    transition = orchestrator.on_tool_started(
        "read_file",
    )

    assert transition is not None
    assert transition[0] is AgentPhase.PLANNING
    assert transition[1] is AgentPhase.EXECUTION


def test_failed_tool_enters_debugging() -> None:
    orchestrator = AgentOrchestrator()

    orchestrator.on_agent_started()
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

    orchestrator.on_agent_started()

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

    orchestrator.on_agent_started()
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

    orchestrator.on_agent_started()
    orchestrator.on_tool_started("web_fetch")
    orchestrator.on_tool_finished(
        error_code="execution_error",
        error_message="failed",
    )
    orchestrator.begin_reflection()

    orchestrator.on_agent_started()

    assert orchestrator.state.phase is AgentPhase.PLANNING
    assert orchestrator.state.iteration == 0
    assert orchestrator.state.tool_calls == 0
    assert orchestrator.state.last_tool_name is None
    assert orchestrator.state.finished is False
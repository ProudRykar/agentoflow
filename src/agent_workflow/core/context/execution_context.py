from __future__ import annotations

from dataclasses import dataclass

from agent_workflow.core.entities.models.agent_phase import AgentPhase


@dataclass(slots=True, frozen=True)
class ExecutionContext:
    """Immutable snapshot of a single execution run.

    Belongs to a run, not to a task: one task may have
    several runs, each with its own iteration/phase/step.
    Budgets and retry counters will live here later;
    they must never leak into TaskState.
    """

    run_id: str
    iteration: int
    phase: AgentPhase
    current_step_id: str | None
    tool_calls_used: int
    blocked_reason: str | None = None

    def __post_init__(self) -> None:
        if self.iteration < 0:
            raise ValueError(
                "iteration must be >= 0",
            )

        if self.tool_calls_used < 0:
            raise ValueError(
                "tool_calls_used must be >= 0",
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "iteration": self.iteration,
            "phase": self.phase.value,
            "current_step_id": self.current_step_id,
            "tool_calls_used": self.tool_calls_used,
            "blocked_reason": self.blocked_reason,
        }

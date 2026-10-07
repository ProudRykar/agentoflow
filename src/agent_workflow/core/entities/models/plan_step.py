from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from agent_workflow.core.entities.models.agent_phase import AgentPhase
from agent_workflow.core.entities.models.subagent import SubagentPower


class PlanStepStatus(StrEnum):
    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

    @property
    def todo_marker(self) -> str:
        """Checkbox glyph, as a reader-scannable list would show it.

        One status, one marker, defined once: the model reading the
        prompt and the user reading the panel must not see two
        different vocabularies for the same state.

        Markers are distinct rather than a tick-and-cross pair so a
        failed step reads as something that happened and was not done,
        not as a step to retry on sight.
        """

        return _TODO_MARKERS[self]


_TODO_MARKERS: dict[PlanStepStatus, str] = {
    PlanStepStatus.COMPLETED: "[✓]",
    PlanStepStatus.ACTIVE: "[•]",
    PlanStepStatus.PENDING: "[○]",
    PlanStepStatus.FAILED: "[✗]",
    PlanStepStatus.SKIPPED: "[-]",
}


@dataclass(slots=True, frozen=True)
class DelegationHint:
    """Advisory hint: this step may be delegated to a subagent.

    Hints are rendered into the LLM context by ContextAssembler.
    The runtime never acts on them; the model decides.
    """

    role: str
    power: SubagentPower
    reason: str

    def __post_init__(self) -> None:
        if not self.role:
            raise ValueError(
                "role must not be empty",
            )

        if not self.reason:
            raise ValueError(
                "reason must not be empty",
            )


@dataclass(slots=True)
class PlanStep:
    """
    One executable step of an AgentPlan.

    PlanStep describes workflow position.
    It does not contain semantic task progress.
    """

    id: str
    description: str
    phase: AgentPhase

    status: PlanStepStatus = PlanStepStatus.PENDING
    attempts: int = 0

    result: str | None = None
    error: str | None = None

    delegation_hint: DelegationHint | None = None

    def start(self) -> None:
        self.status = PlanStepStatus.ACTIVE
        self.attempts += 1
        self.error = None

    def complete(
        self,
        result: str | None = None,
    ) -> None:
        self.status = PlanStepStatus.COMPLETED
        self.result = result
        self.error = None

    def fail(
        self,
        error: str,
    ) -> None:
        self.status = PlanStepStatus.FAILED
        self.error = error

    def skip(
        self,
        reason: str | None = None,
    ) -> None:
        self.status = PlanStepStatus.SKIPPED
        self.result = reason
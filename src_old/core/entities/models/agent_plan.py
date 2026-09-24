from __future__ import annotations

from dataclasses import dataclass, field

from core.entities.models.plan_step import (
    PlanStep,
    PlanStepStatus,
)


@dataclass(slots=True)
class AgentPlan:
    """
    Mutable execution plan for one agent run.

    AgentPlan represents workflow state.
    TaskProgress represents semantic task progress.
    """

    objective: str

    steps: list[PlanStep] = field(
        default_factory=list,
    )

    revision: int = 1

    current_step_id: str | None = None

    def current_step(self) -> PlanStep | None:
        if self.current_step_id is None:
            return None

        for step in self.steps:
            if step.id == self.current_step_id:
                return step

        return None

    def get_step(
        self,
        step_id: str,
    ) -> PlanStep | None:
        for step in self.steps:
            if step.id == step_id:
                return step

        return None

    def activate(
        self,
        step: PlanStep,
    ) -> PlanStep:
        self.current_step_id = step.id

        if step.status == PlanStepStatus.PENDING:
            step.start()

        return step

    def next_pending_step(self) -> PlanStep | None:
        for step in self.steps:
            if step.status == PlanStepStatus.PENDING:
                return step

        return None

    def activate_next(self) -> PlanStep | None:
        step = self.next_pending_step()

        if step is None:
            self.current_step_id = None
            return None

        return self.activate(step)

    def append(
        self,
        step: PlanStep,
    ) -> None:
        self.steps.append(step)

    def all_completed(self) -> bool:
        return bool(self.steps) and all(
            step.status == PlanStepStatus.COMPLETED
            for step in self.steps
        )

    def completed_count(self) -> int:
        return sum(
            step.status == PlanStepStatus.COMPLETED
            for step in self.steps
        )

    def total_count(self) -> int:
        return len(self.steps)

    def reset_current_to_pending(self) -> PlanStep | None:
        step = self.current_step()

        if step is None:
            return None

        step.status = PlanStepStatus.PENDING
        step.error = None

        return step
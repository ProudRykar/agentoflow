"""A checklist the model writes and the user can read.

Two vocabularies, deliberately kept apart:

- ``AgentPlan`` is the harness's plan: phases it can verify, with
  invariants (exactly one active, synthesis last) that make completion
  decidable without a model in the loop.
- ``TodoList`` is what the agent intends to do. Free-form, written
  through a tool, no invariants -- because a checklist the model cannot
  edit is a checklist the model cannot keep true.

They are not merged. Folding todos into the plan would break the
completion gate on every task where the model wrote one step too many;
keeping them apart means the plan stays decidable and the user sees
what the agent actually means to do.

Ownership is one-way per run: the planner seeds the list once, and the
model owns it after that. A later turn reseeds, because the previous
task's checklist is not this task's business.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from agent_workflow.core.entities.models.plan_step import (
    PlanStep,
    PlanStepStatus,
)


class TodoStatus(StrEnum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


_TODO_MARKERS: dict[TodoStatus, str] = {
    TodoStatus.COMPLETED: "[✓]",
    TodoStatus.IN_PROGRESS: "[•]",
    TodoStatus.PENDING: "[○]",
    TodoStatus.CANCELLED: "[-]",
}

# A model that marks everything done has marked nothing done. Rendering
# a bare count invites exactly that, so completion is per item.
_MAX_ITEMS = 60
_MAX_CONTENT = 400


@dataclass(slots=True, frozen=True)
class AgentPlanTodoItem:
    content: str
    status: TodoStatus = TodoStatus.PENDING

    def __post_init__(self) -> None:
        content = self.content.strip()

        if not content:
            raise ValueError("todo content must not be empty")

        if len(content) > _MAX_CONTENT:
            raise ValueError(
                f"todo content must be <= {_MAX_CONTENT} characters"
            )

        object.__setattr__(self, "content", content)

    @property
    def marker(self) -> str:
        return _TODO_MARKERS[self.status]


@dataclass(slots=True)
class TodoList:
    items: tuple[AgentPlanTodoItem, ...] = field(default_factory=tuple)

    def replace(
        self,
        items: tuple[AgentPlanTodoItem, ...],
    ) -> None:
        """Overwrite wholesale, as the tool contract promises.

        Trimming rather than rejecting: a list that grows past the cap
        is a model that lost track, and refusing the write would leave
        it stuck writing ever longer lists. The tail is the part most
        likely to be padding.
        """

        ordered = list(items)

        if len(ordered) > _MAX_ITEMS:
            ordered = ordered[:_MAX_ITEMS]

        self.items = tuple(ordered)

    def seed(
        self,
        steps: tuple[PlanStep, ...],
    ) -> None:
        """The planner's first draft, in the model's vocabulary.

        Only once per run: afterwards the model owns the list, and
        reseeding would discard whatever it decided in the meantime.
        """

        if self.items:
            return

        self.replace(
            tuple(
                AgentPlanTodoItem(
                    content=step.description,
                    status=self._from_plan_status(
                        step.status
                    ),
                )
                for step in steps
            ),
        )

    @staticmethod
    def _from_plan_status(
        status: PlanStepStatus,
    ) -> TodoStatus:
        if status is PlanStepStatus.COMPLETED:
            return TodoStatus.COMPLETED

        if status is PlanStepStatus.ACTIVE:
            return TodoStatus.IN_PROGRESS

        if status is PlanStepStatus.SKIPPED:
            return TodoStatus.CANCELLED

        return TodoStatus.PENDING

    @property
    def active(self) -> AgentPlanTodoItem | None:
        """The single item in progress, if any."""

        for item in self.items:
            if item.status is TodoStatus.IN_PROGRESS:
                return item

        return None

    def is_empty(self) -> bool:
        return not self.items

    def render(self) -> str:
        """The checklist, in the form both the model and the user read."""

        if not self.items:
            return "▼Todo\n(empty)"

        done = sum(
            1
            for item in self.items
            if item.status is TodoStatus.COMPLETED
        )

        lines = [f"▼Todo  ({done}/{len(self.items)})"]

        lines.extend(
            f"{item.marker} {item.content}"
            for item in self.items
        )

        return "\n".join(lines)

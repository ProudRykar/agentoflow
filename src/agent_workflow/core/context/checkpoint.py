from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from agent_workflow.core.entities.models.plan_step import PlanStepStatus

if TYPE_CHECKING:
    from core.entities.models.agent_orchestrator import (
        AgentOrchestrator,
    )


@dataclass(slots=True, frozen=True)
class TaskCheckpoint:
    """Durable task state for context transitions.

    Deterministic state serialization, NOT a model opinion about
    the state. ``objective`` is always copied from the TaskAnchor,
    never invented by a summary.

    decisions/learnings are reserved for later; the first version
    captures only runtime-owned facts.
    """

    task_id: str
    objective: str
    completed_steps: tuple[str, ...] = ()
    active_step: str | None = None
    decisions: tuple[str, ...] = ()
    learnings: tuple[str, ...] = ()
    research_evidence_ids: tuple[str, ...] = ()
    missing_requirements: tuple[str, ...] = ()
    next_actions: tuple[str, ...] = ()
    conversation_cursor: str | None = None
    history_window_id: str | None = None

    def __post_init__(self) -> None:
        if not self.task_id:
            raise ValueError(
                "task_id must not be empty",
            )

        if not self.objective:
            raise ValueError(
                "objective must not be empty",
            )

        object.__setattr__(
            self,
            "completed_steps",
            tuple(self.completed_steps),
        )
        object.__setattr__(
            self,
            "decisions",
            tuple(self.decisions),
        )
        object.__setattr__(
            self,
            "learnings",
            tuple(self.learnings),
        )
        object.__setattr__(
            self,
            "research_evidence_ids",
            tuple(self.research_evidence_ids),
        )
        object.__setattr__(
            self,
            "missing_requirements",
            tuple(self.missing_requirements),
        )
        object.__setattr__(
            self,
            "next_actions",
            tuple(self.next_actions),
        )

    def render(self) -> str:
        lines = [
            f"task: {self.task_id}",
            f"objective: {self.objective}",
        ]

        if self.completed_steps:
            lines.append(
                "completed: "
                + ", ".join(self.completed_steps)
            )
        else:
            lines.append("completed: (none)")

        lines.append(
            f"active step: {self.active_step or '(none)'}"
        )

        if self.decisions:
            lines.append(
                "decisions: " + "; ".join(self.decisions)
            )

        if self.learnings:
            lines.append(
                "learnings: " + "; ".join(self.learnings)
            )

        if self.research_evidence_ids:
            lines.append(
                "evidence: "
                + ", ".join(self.research_evidence_ids)
            )

        if self.missing_requirements:
            lines.append(
                "missing: "
                + ", ".join(self.missing_requirements)
            )

        if self.next_actions:
            lines.append(
                "next: " + "; ".join(self.next_actions)
            )

        return "\n".join(lines)

    def to_dict(self) -> dict[str, object]:
        return {
            "task_id": self.task_id,
            "objective": self.objective,
            "completed_steps": list(self.completed_steps),
            "active_step": self.active_step,
            "decisions": list(self.decisions),
            "learnings": list(self.learnings),
            "research_evidence_ids": list(
                self.research_evidence_ids
            ),
            "missing_requirements": list(
                self.missing_requirements
            ),
            "next_actions": list(self.next_actions),
            "conversation_cursor": self.conversation_cursor,
            "history_window_id": self.history_window_id,
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, object],
    ) -> TaskCheckpoint:
        def _tuple(key: str) -> tuple[str, ...]:
            raw = data.get(key, ())

            if not isinstance(raw, (list, tuple)):
                raise TypeError(
                    f"{key} must be a list or tuple",
                )

            return tuple(str(item) for item in raw)

        active_step = data.get("active_step")
        cursor = data.get("conversation_cursor")
        window = data.get("history_window_id")

        for name, value in (
            ("active_step", active_step),
            ("conversation_cursor", cursor),
            ("history_window_id", window),
        ):
            if value is not None and not isinstance(
                value,
                str,
            ):
                raise TypeError(
                    f"{name} must be a string or None",
                )

        return cls(
            task_id=str(data["task_id"]),
            objective=str(data["objective"]),
            completed_steps=_tuple("completed_steps"),
            active_step=active_step,  # type: ignore[arg-type]
            decisions=_tuple("decisions"),
            learnings=_tuple("learnings"),
            research_evidence_ids=_tuple(
                "research_evidence_ids"
            ),
            missing_requirements=_tuple(
                "missing_requirements"
            ),
            next_actions=_tuple("next_actions"),
            conversation_cursor=cursor,  # type: ignore[arg-type]
            history_window_id=window,  # type: ignore[arg-type]
        )


def build_checkpoint(
    orchestrator: AgentOrchestrator,
) -> TaskCheckpoint:
    """Capture a deterministic checkpoint from runtime state.

    Raises RuntimeError when no task is prepared.
    """

    anchor = orchestrator.task_anchor

    if anchor is None:
        raise RuntimeError(
            "Cannot build checkpoint without a TaskAnchor",
        )

    plan = orchestrator.plan

    if plan is None:
        completed: tuple[str, ...] = ()
        active: str | None = None
    else:
        completed = tuple(
            step.id
            for step in plan.steps
            if step.status is PlanStepStatus.COMPLETED
        )
        current = plan.current_step()
        active = current.id if current is not None else None

    completion = orchestrator.check_completion()

    next_actions: list[str] = []

    if plan is not None:
        current = plan.current_step()

        if current is not None:
            next_actions.append(
                f"continue {current.phase.value} step "
                f"'{current.id}': {current.description}"
            )

    if completion.missing:
        next_actions.append(
            "missing requirements: "
            + ", ".join(completion.missing)
        )

    return TaskCheckpoint(
        task_id=anchor.task_id,
        objective=anchor.objective,
        completed_steps=completed,
        active_step=active,
        research_evidence_ids=(
            orchestrator.evidence_store.ids
        ),
        missing_requirements=completion.missing,
        next_actions=tuple(next_actions),
    )

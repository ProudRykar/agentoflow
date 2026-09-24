from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4


@dataclass(slots=True, frozen=True)
class TaskAnchor:
    """Immutable source of truth for WHAT the user asked.

    Created once per task. Never mutated, never recreated from
    tool output, conversation history, or model summaries.

    ``task_id`` is unique per task. ``parent_task_id`` is reserved
    for future task lineage (e.g. ``continue_run()`` follow-ups)
    and is currently always ``None``.
    """

    task_id: str
    original_prompt: str
    objective: str
    constraints: tuple[str, ...] = ()
    parent_task_id: str | None = None

    def __post_init__(self) -> None:
        if not self.task_id:
            raise ValueError(
                "task_id must not be empty",
            )

        if not self.original_prompt:
            raise ValueError(
                "original_prompt must not be empty",
            )

        if not self.objective:
            raise ValueError(
                "objective must not be empty",
            )

        object.__setattr__(
            self,
            "constraints",
            tuple(self.constraints),
        )

    @classmethod
    def create(
        cls,
        original_prompt: str,
        objective: str,
        constraints: tuple[str, ...] = (),
        task_id: str | None = None,
        parent_task_id: str | None = None,
    ) -> TaskAnchor:
        """Create a new anchor with a generated task_id by default."""

        return cls(
            task_id=(
                task_id
                or f"task-{uuid4().hex[:8]}"
            ),
            original_prompt=original_prompt,
            objective=objective,
            constraints=tuple(constraints),
            parent_task_id=parent_task_id,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "task_id": self.task_id,
            "original_prompt": self.original_prompt,
            "objective": self.objective,
            "constraints": list(self.constraints),
            "parent_task_id": self.parent_task_id,
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, object],
    ) -> TaskAnchor:
        raw_constraints = data.get(
            "constraints",
            (),
        )

        if isinstance(
            raw_constraints,
            (list, tuple),
        ):
            constraints = tuple(
                str(item)
                for item in raw_constraints
            )
        else:
            raise TypeError(
                "constraints must be a list or tuple",
            )

        parent_task_id = data.get(
            "parent_task_id",
        )

        if parent_task_id is not None and not isinstance(
            parent_task_id,
            str,
        ):
            raise TypeError(
                "parent_task_id must be a string or None",
            )

        return cls(
            task_id=str(data["task_id"]),
            original_prompt=str(
                data["original_prompt"],
            ),
            objective=str(data["objective"]),
            constraints=constraints,
            parent_task_id=parent_task_id,
        )

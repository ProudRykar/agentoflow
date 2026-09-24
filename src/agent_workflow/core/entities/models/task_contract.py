from __future__ import annotations

from dataclasses import dataclass, field

from agent_workflow.core.entities.models.research_contract import (
    ResearchContract,
    ResearchCoverage,
    ResearchResult,
)


@dataclass(slots=True, frozen=True)
class TaskContract:
    """
    Explicit runtime requirements for completing a task.
    """

    requires_research: bool = False

    requires_verification: bool = False

    requires_reflection: bool = False

    research: ResearchContract | None = None

    def __post_init__(self) -> None:
        if (
            self.research is not None
            and not self.requires_research
        ):
            object.__setattr__(
                self,
                "requires_research",
                True,
            )


@dataclass(slots=True)
class TaskProgress:
    """
    Runtime-owned progress.

    Nothing here is inferred from tool names.
    """

    research_completed: bool = False

    verification_completed: bool = False

    reflection_completed: bool = False

    research_coverage: ResearchCoverage = field(
        default_factory=ResearchCoverage,
    )

    def reset(self) -> None:
        self.research_completed = False

        self.verification_completed = False

        self.reflection_completed = False

        self.research_coverage.reset()

    def record_research_result(
        self,
        result: ResearchResult,
    ) -> None:
        self.research_coverage.merge(
            result
        )

    def mark_research_completed(
        self,
    ) -> None:
        self.research_completed = True

    def mark_verification_completed(
        self,
    ) -> None:
        self.verification_completed = True

    def mark_reflection_completed(
        self,
    ) -> None:
        self.reflection_completed = True
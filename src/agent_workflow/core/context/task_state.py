from __future__ import annotations

from dataclasses import dataclass

from agent_workflow.core.context.task_anchor import TaskAnchor
from agent_workflow.core.entities.models.research_contract import (
    ResearchCoverage,
)
from agent_workflow.core.entities.models.task_contract import TaskContract
from agent_workflow.core.entities.models.task_plan import TaskPlan


@dataclass(slots=True, frozen=True)
class TaskState:
    """Immutable snapshot of WHAT the task is.

    Read-view over the orchestrator's mutable runtime fields.
    ``coverage`` is a defensive copy: later research progress
    never mutates an already taken snapshot.

    Anchor = WHAT was asked.
    Plan = HOW it is executed.
    Contract = WHAT must hold for completion.
    Coverage = WHAT was found so far.
    """

    anchor: TaskAnchor
    plan: TaskPlan | None
    contract: TaskContract
    coverage: ResearchCoverage

    @classmethod
    def snapshot(
        cls,
        anchor: TaskAnchor,
        plan: TaskPlan | None,
        contract: TaskContract,
        coverage: ResearchCoverage,
    ) -> TaskState:
        return cls(
            anchor=anchor,
            plan=plan,
            contract=contract,
            coverage=ResearchCoverage(
                fetched_urls=set(
                    coverage.fetched_urls,
                ),
                failed_urls=set(
                    coverage.failed_urls,
                ),
                discovered_urls=set(
                    coverage.discovered_urls,
                ),
                max_depth_reached=(
                    coverage.max_depth_reached
                ),
                total_bytes=coverage.total_bytes,
            ),
        )

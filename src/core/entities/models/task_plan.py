from __future__ import annotations

from dataclasses import dataclass

from core.entities.models.research_contract import ResearchContract


@dataclass(slots=True, frozen=True)
class TaskPlan:
    """
    Structured task plan produced by the Planner.

    Contains the objective and optional research requirements.
    """

    objective: str

    research: ResearchContract | None = None
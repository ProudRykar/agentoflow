from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Literal

from core.entities.models.model_catalog import (
    ModelCapability,
)


class SubagentStatus(StrEnum):
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


class TaskComplexity(StrEnum):
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"


class SubagentPower(StrEnum):
    """Explicit compute tier chosen by the caller (or main LLM).

    LOW/MEDIUM/HIGH map onto ModelProfile.resource_usage bands
    1-2 / 3 / 4-5. AUTO leaves the choice to the router.
    """

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    AUTO = "auto"


@dataclass(slots=True, frozen=True)
class AgentRun:
    run_id: str
    parent_run_id: str | None
    agent_id: str
    role: str
    model: str


@dataclass(slots=True, frozen=True)
class TaskProfile:
    capability: ModelCapability = (
        ModelCapability.GENERAL
    )

    complexity: TaskComplexity = (
        TaskComplexity.MEDIUM
    )

    power: SubagentPower = SubagentPower.AUTO

    min_context_tokens: int | None = None


@dataclass(slots=True, frozen=True)
class SubagentTask:
    role: str
    objective: str

    instructions: str = ""
    context: str = ""

    model: str = "auto"

    profile: TaskProfile = TaskProfile()

    tools: tuple[str, ...] = ()
    permissions: frozenset[str] = frozenset()

    # None means "use SubagentConfig default".
    max_iterations: int | None = None
    max_tool_calls: int | None = None
    max_tokens: int | None = None
    timeout_seconds: float | None = None


@dataclass(slots=True, frozen=True)
class SubagentResult:
    run_id: str
    parent_run_id: str

    status: SubagentStatus

    summary: str | None
    output: str | None

    iterations: int
    tool_calls: int

    input_tokens: int | None
    output_tokens: int | None

    duration_seconds: float

    error: str | None = None
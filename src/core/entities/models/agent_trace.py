from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.entities.models.agent_phase import AgentPhase


@dataclass(slots=True, frozen=True)
class AgentStarted:
    prompt: str
    run_id: str = ""
    parent_run_id: str | None = None
    agent_id: str = "main"
    role: str = "main"
    model: str = ""


@dataclass(slots=True, frozen=True)
class AgentPhaseChanged:
    previous_phase: AgentPhase
    phase: AgentPhase
    reason: str
    iteration: int
    run_id: str = ""
    parent_run_id: str | None = None


@dataclass(slots=True, frozen=True)
class LLMRequested:
    iteration: int
    message_count: int
    tool_count: int
    run_id: str = ""
    parent_run_id: str | None = None
    estimated_tokens: int | None = None


@dataclass(slots=True, frozen=True)
class LLMThinkingChunk:
    iteration: int
    content: str
    run_id: str = ""
    parent_run_id: str | None = None


@dataclass(slots=True, frozen=True)
class LLMContentChunk:
    iteration: int
    content: str
    run_id: str = ""
    parent_run_id: str | None = None


@dataclass(slots=True, frozen=True)
class LLMResponded:
    iteration: int
    content: str | None
    thinking: str | None
    tool_call_count: int
    run_id: str = ""
    parent_run_id: str | None = None


@dataclass(slots=True, frozen=True)
class ToolStarted:
    iteration: int
    tool_call_id: str
    tool_name: str
    arguments: dict[str, Any]
    run_id: str = ""
    parent_run_id: str | None = None


@dataclass(slots=True, frozen=True)
class ToolFinished:
    iteration: int
    tool_call_id: str
    tool_name: str
    output: str | None
    error_code: str | None
    error_message: str | None
    run_id: str = ""
    parent_run_id: str | None = None
    duration_seconds: float | None = None


@dataclass(slots=True, frozen=True)
class AgentFinished:
    result: str
    run_id: str = ""
    parent_run_id: str | None = None
    agent_id: str = "main"
    role: str = "main"
    model: str = ""


AgentEvent = (
    AgentStarted
    | AgentPhaseChanged
    | LLMRequested
    | LLMThinkingChunk
    | LLMContentChunk
    | LLMResponded
    | ToolStarted
    | ToolFinished
    | AgentFinished
)
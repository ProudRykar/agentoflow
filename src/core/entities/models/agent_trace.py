from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True, frozen=True)
class AgentStarted:
    prompt: str


@dataclass(slots=True, frozen=True)
class LLMRequested:
    iteration: int
    message_count: int
    tool_count: int


@dataclass(slots=True, frozen=True)
class LLMThinkingChunk:
    iteration: int
    content: str


@dataclass(slots=True, frozen=True)
class LLMContentChunk:
    iteration: int
    content: str


@dataclass(slots=True, frozen=True)
class LLMResponded:
    iteration: int
    content: str | None
    thinking: str | None
    tool_call_count: int


@dataclass(slots=True, frozen=True)
class ToolStarted:
    iteration: int
    tool_call_id: str
    tool_name: str
    arguments: dict[str, Any]


@dataclass(slots=True, frozen=True)
class ToolFinished:
    iteration: int
    tool_call_id: str
    tool_name: str
    output: str | None
    error_code: str | None
    error_message: str | None


@dataclass(slots=True, frozen=True)
class AgentFinished:
    result: str


AgentEvent = (
    AgentStarted
    | LLMRequested
    | LLMThinkingChunk
    | LLMContentChunk
    | LLMResponded
    | ToolStarted
    | ToolFinished
    | AgentFinished
)

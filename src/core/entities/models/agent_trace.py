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
class LLMResponded:
    iteration: int
    content: str | None
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


@dataclass(slots=True, frozen=True)
class AgentFinished:
    result: str


AgentEvent = (
    AgentStarted
    | LLMRequested
    | LLMResponded
    | ToolStarted
    | ToolFinished
    | AgentFinished
)
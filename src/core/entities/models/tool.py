from dataclasses import dataclass
from collections.abc import Callable, Awaitable
from enum import StrEnum
from pathlib import Path
from typing import Generic, Mapping, TypeVar

# Переменные:
InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")
type ToolHandler[InputT, OutputT] = Callable[
    [InputT, ToolContext], Awaitable[OutputT]
]

@dataclass(slots=True, frozen=True)
class ToolError:
    message: str
    code: str
    retryable: bool


@dataclass(slots=True)
class ToolContext:
    working_directory: Path
    environment: Mapping[str, str]
    allowed_path: tuple[Path, ...]
    permissions: frozenset[str]


@dataclass(slots=True, frozen=True)
class ToolResult:
    output: str | None = None
    error: ToolError | None = None


@dataclass(slots=True, frozen=True)
class ToolPolicy:
    permissions: frozenset[str]
    timeout: float
    max_output_size: int


class ToolSource(StrEnum):
    BUILTIN = "builtin"
    MCP = "mcp"
    PLUGIN = "plugin"


@dataclass(slots=True, frozen=True)
class Tool(Generic[InputT, OutputT]):
    name: str
    description: str
    input_type: type[InputT]
    handler: ToolHandler[InputT, OutputT]
    policy: ToolPolicy
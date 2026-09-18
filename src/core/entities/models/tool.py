from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Generic, TypeVar


InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")


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
    approved_permissions: frozenset[str] = frozenset()

    run_id: str = ""
    parent_run_id: str | None = None

    # Callback для потоковой передачи событий Agent наружу.
    # Используется, в частности, для отображения событий дочерних
    # subagent'ов в UI.
    event_callback: (
        Callable[[object], Awaitable[None]] | None
    ) = None


type ToolHandler[InputT, OutputT] = Callable[
    [InputT, ToolContext],
    Awaitable[OutputT],
]


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
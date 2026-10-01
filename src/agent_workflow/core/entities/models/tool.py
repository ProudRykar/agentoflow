from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
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

    # Optional reference to the agent for skill management tools
    agent: object = None

    # Resources published by plugins during setup, keyed by
    # plugin name. ToolContext is a slots dataclass, so a
    # plugin cannot attach its client as an attribute; this
    # mapping is the supported channel instead.
    plugin_resources: Mapping[str, Mapping[str, object]] = (
        field(default_factory=dict)
    )

    def plugin_resource(
        self,
        plugin_name: str,
        key: str,
        default: object = None,
    ) -> object:
        """Fetch a resource published by a plugin."""

        resources = self.plugin_resources.get(plugin_name)

        if not resources:
            return default

        return resources.get(key, default)


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

    # When true, the tool is gated by the approval flow even
    # if every permission it declares is already granted. This
    # is how plugins mark destructive operations.
    requires_approval: bool = False


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
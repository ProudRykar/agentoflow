from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.markdown import Markdown
from rich.status import Status

from core.entities.models.agent_trace import (
    AgentFinished,
    AgentStarted,
    LLMRequested,
    LLMResponded,
    ToolFinished,
    ToolStarted,
)


class Renderer:
    def __init__(
        self,
        console: Console | None = None,
        verbose: bool = False,
    ) -> None:
        self._console = console or Console()
        self._verbose = verbose
        self._status: Status | None = None

    async def render(self, event: Any) -> None:
        if isinstance(event, AgentStarted):
            return

        if isinstance(event, LLMRequested):
            self._start_thinking()

            if self._verbose:
                self._console.print(
                    f"[dim]llm · iteration={event.iteration} "
                    f"messages={event.message_count} "
                    f"tools={event.tool_count}[/dim]"
                )

            return

        if isinstance(event, LLMResponded):
            self._stop_thinking()

            if self._verbose:
                self._console.print(
                    f"[dim]llm · iteration={event.iteration} "
                    f"tool_calls={event.tool_call_count}[/dim]"
                )

            return

        if isinstance(event, ToolStarted):
            self._stop_thinking()
            self._render_tool_started(event)
            return

        if isinstance(event, ToolFinished):
            self._render_tool_finished(event)
            return

        if isinstance(event, AgentFinished):
            self._stop_thinking()
            return

    def _start_thinking(self) -> None:
        if self._status is not None:
            return

        self._status = Status(
            "Thinking...",
            spinner="dots",
        )
        self._status.start()

    def _stop_thinking(self) -> None:
        if self._status is None:
            return

        self._status.stop()
        self._status = None

    def _render_tool_started(self, event: ToolStarted) -> None:
        self._console.print(
            f"  [bold]● {event.tool_name}[/bold]"
        )

        for name, value in event.arguments.items():
            self._console.print(
                f"    [dim]{name}:[/dim] {value}"
            )

    def _render_tool_finished(self, event: ToolFinished) -> None:
        if event.error_code is None:
            self._console.print(
                "    [dim]✓[/dim]"
            )
        else:
            self._console.print(
                f"    [red]✗ {event.error_code}[/red]"
            )

    def response(self, content: str) -> None:
        self._console.print(Markdown(content))
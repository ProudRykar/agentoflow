from __future__ import annotations

from typing import Any

from rich.console import Console, Group
from rich.markdown import Markdown
from rich.panel import Panel
from rich.spinner import Spinner
from rich.text import Text

from cli.ui.state import (
    AssistantMessage,
    ApprovalView,
    ToolStatus,
    ToolView,
    UIState,
    UserMessage,
)
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
        self._state = UIState()

    @property
    def state(self) -> UIState:
        return self._state

    async def render(
        self,
        event: Any,
    ) -> None:
        if isinstance(event, AgentStarted):
            self._handle_agent_started(event)
            return

        if isinstance(event, LLMRequested):
            self._handle_llm_requested(event)
            return

        if isinstance(event, LLMResponded):
            self._handle_llm_responded(event)
            return

        if isinstance(event, ToolStarted):
            self._handle_tool_started(event)
            return

        if isinstance(event, ToolFinished):
            self._handle_tool_finished(event)
            return

        if isinstance(event, AgentFinished):
            self._handle_agent_finished(event)
            return

    def _handle_agent_started(
        self,
        event: AgentStarted,
    ) -> None:
        self._state.reset_execution()

        self._state.add_user_message(
            event.prompt,
        )

        self._render()

    def _handle_llm_requested(
        self,
        event: LLMRequested,
    ) -> None:
        self._state.start_thinking(
            iteration=event.iteration,
        )

        self._render()

        if self._verbose:
            self._console.print(
                Text(
                    f"llm · iteration={event.iteration} "
                    f"messages={event.message_count} "
                    f"tools={event.tool_count}",
                    style="dim",
                )
            )

    def _handle_llm_responded(
        self,
        event: LLMResponded,
    ) -> None:
        self._state.stop_thinking()

        if self._verbose:
            self._console.print(
                Text(
                    f"llm · iteration={event.iteration} "
                    f"tool_calls={event.tool_call_count}",
                    style="dim",
                )
            )

        self._render()

    def _handle_tool_started(
        self,
        event: ToolStarted,
    ) -> None:
        self._state.stop_thinking()

        self._state.add_tool(
            call_id=event.tool_call_id,
            name=event.tool_name,
            arguments=event.arguments,
        )

        self._render()

    def _handle_tool_finished(
        self,
        event: ToolFinished,
    ) -> None:
        self._state.finish_tool(
            call_id=event.tool_call_id,
            output=event.output,
            error_code=event.error_code,
            error_message=event.error_message,
        )

        self._render()

    def _handle_agent_finished(
        self,
        event: AgentFinished,
    ) -> None:
        self._state.stop_thinking()

        self._state.add_assistant_message(
            event.result,
        )

        self._render()

    def approval_requested(
        self,
        tool_name: str,
        permission: str,
        reason: str,
        arguments: dict[str, Any],
    ) -> None:
        self._state.start_approval(
            tool_name=tool_name,
            permission=permission,
            reason=reason,
            arguments=arguments,
        )

        self._render()

    def approval_finished(self) -> None:
        self._state.finish_approval()
        self._render()

    def _render(self) -> None:
        self._console.clear()

        self._render_conversation()

        if self._state.thinking:
            self._console.print()
            self._console.print(
                Spinner(
                    "dots",
                    text=Text(
                        f"Thinking · iteration "
                        f"{self._state.iteration}",
                        style="dim",
                    ),
                )
            )

        if self._state.active_approval is not None:
            self._console.print()
            self._render_approval()

    def _render_conversation(self) -> None:
        for item in self._state.conversation:
            self._render_item(item)

    def _render_item(
        self,
        item: UserMessage | AssistantMessage | ToolView,
    ) -> None:
        if isinstance(item, UserMessage):
            self._console.print(
                Text.assemble(
                    ("▌ ", "bold"),
                    ("> ", "bold"),
                    (item.content, ""),
                )
            )
            self._console.print()
            return

        if isinstance(item, AssistantMessage):
            self._console.print(
                Group(
                    Text(
                        "Gemma",
                        style="bold",
                    ),
                    Markdown(item.content),
                )
            )
            self._console.print()
            return

        self._render_tool(item)

    def _render_tool(
        self,
        tool: ToolView,
    ) -> None:
        if tool.status is ToolStatus.RUNNING:
            symbol = "●"
            style = "bold"
        elif tool.status is ToolStatus.SUCCESS:
            symbol = "▸"
            style = "dim"
        else:
            symbol = "✗"
            style = "red"

        self._console.print(
            Text.assemble(
                ("  ", ""),
                (symbol, style),
                (" ", ""),
                (tool.name, "bold"),
            )
        )

        for name, value in tool.arguments.items():
            self._console.print(
                Text.assemble(
                    ("      ", ""),
                    (f"{name}:", "dim"),
                    (" ", ""),
                    (str(value), ""),
                )
            )

        if tool.status is ToolStatus.SUCCESS:
            self._console.print(
                "      [dim]✓[/dim]"
            )

        elif tool.status is ToolStatus.ERROR:
            self._console.print(
                f"      [red]✗ {tool.error_code}[/red]"
            )

            if self._verbose and tool.error_message:
                self._console.print(
                    f"        [dim]{tool.error_message}[/dim]"
                )

        self._console.print()

    def _render_approval(self) -> None:
        approval = self._state.active_approval

        if approval is None:
            return

        lines = [
            "[bold]Permission required[/bold]",
            "",
            f"[dim]tool:[/dim]       {approval.tool_name}",
            f"[dim]permission:[/dim] {approval.permission}",
            f"[dim]reason:[/dim]     {approval.reason}",
        ]

        for name, value in approval.arguments.items():
            lines.append(
                f"[dim]{name}:[/dim] {value}"
            )

        lines.extend(
            [
                "",
                "[bold][a][/bold] Allow    "
                "[bold][d][/bold] Deny",
            ]
        )

        self._console.print(
            Panel(
                "\n".join(lines),
                border_style="yellow",
                expand=False,
            )
        )
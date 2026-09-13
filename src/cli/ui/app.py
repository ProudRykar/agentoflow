from __future__ import annotations

import asyncio
from typing import Any

from prompt_toolkit.application import Application
from prompt_toolkit.formatted_text import StyleAndTextTuples
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import HSplit, Layout, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.styles import Style
from prompt_toolkit.widgets import TextArea

from cli.approval import ApprovalController
from cli.input import CommandCompleter, create_history
from cli.ui.state import (
    AssistantMessage,
    ThinkingView,
    ToolStatus,
    ToolView,
    UIState,
    UserMessage,
)
from core.entities.models.agent_trace import (
    AgentEvent,
    AgentFinished,
    AgentStarted,
    LLMContentChunk,
    LLMRequested,
    LLMResponded,
    LLMThinkingChunk,
    ToolFinished,
    ToolStarted,
)


class AgentUI:
    def __init__(
        self,
        runtime,
        approval: ApprovalController,
    ) -> None:
        self.runtime = runtime
        self.approval = approval
        self.state = UIState()

        self._tasks: set[asyncio.Task[Any]] = set()

        # Когда True, conversation автоматически следует
        # за последней строкой вывода.
        #
        # Когда False, пользователь вручную просматривает историю.
        self._follow_output = True

        self._scroll_position = 0

        self.conversation = Window(
            content=FormattedTextControl(
                self._conversation_text,
            ),
            wrap_lines=True,
            always_hide_cursor=True,
            get_vertical_scroll=self._get_vertical_scroll,
        )

        self.input = TextArea(
            text="",
            multiline=True,
            wrap_lines=True,
            scrollbar=False,
            height=3,
            prompt="> ",
            completer=CommandCompleter(),
            complete_while_typing=True,
            history=create_history(
                runtime.paths.history,
            ),
            accept_handler=self._accept_input,
        )

        self.header = Window(
            content=FormattedTextControl(
                self._header_text,
            ),
            height=2,
        )

        self.footer = Window(
            content=FormattedTextControl(
                self._footer_text,
            ),
            height=1,
        )

        root = HSplit(
            [
                self.header,
                Window(height=1),
                self.conversation,
                self.footer,
                self.input,
            ],
        )

        self.application = Application(
            layout=Layout(
                root,
                focused_element=self.input,
            ),
            key_bindings=self._create_key_bindings(),
            style=Style.from_dict(
                {
                    "header": "bold",
                    "user": "bold",
                    "assistant": "bold",
                    "tool": "bold",
                    "thinking": "#888888",
                    "thinking_header": "bold #888888",
                    "error": "bold red",
                    "approval": "bold yellow",
                    "dim": "#888888",
                },
            ),
            full_screen=True,
            mouse_support=True,
            refresh_interval=0.05,
        )

        self.approval.set_on_change(
            self.invalidate,
        )

    async def run(self) -> None:
        await self.application.run_async()

    def invalidate(self) -> None:
        self.application.invalidate()

    # ------------------------------------------------------------------
    # Scrolling
    # ------------------------------------------------------------------

    def _get_vertical_scroll(
        self,
        window: Window,
    ) -> int:
        render_info = window.render_info

        if render_info is None:
            return self._scroll_position

        if self._follow_output:
            return max(
                0,
                render_info.content_height
                - render_info.window_height,
            )

        return self._scroll_position

    def _scroll_to_end(self) -> None:
        self._follow_output = True
        self.invalidate()

    def _scroll_up(
        self,
        amount: int = 5,
    ) -> None:
        render_info = self.conversation.render_info

        if render_info is None:
            return

        self._follow_output = False

        self._scroll_position = max(
            0,
            render_info.vertical_scroll - amount,
        )

        self.invalidate()

    def _scroll_down(
        self,
        amount: int = 5,
    ) -> None:
        render_info = self.conversation.render_info

        if render_info is None:
            return

        max_scroll = max(
            0,
            render_info.content_height
            - render_info.window_height,
        )

        self._scroll_position = min(
            max_scroll,
            render_info.vertical_scroll + amount,
        )

        if self._scroll_position >= max_scroll:
            self._follow_output = True

        self.invalidate()

    def _scroll_home(self) -> None:
        self._follow_output = False
        self._scroll_position = 0
        self.invalidate()

    def _scroll_end(self) -> None:
        self._scroll_to_end()

    # ------------------------------------------------------------------
    # Key bindings
    # ------------------------------------------------------------------

    def _create_key_bindings(self) -> KeyBindings:
        kb = KeyBindings()

        @kb.add("enter")
        def _enter(event) -> None:
            if self.approval.active:
                return

            self._accept_input(
                self.input.buffer,
            )

        @kb.add("f13")
        def _newline(event) -> None:
            if self.approval.active or self.state.running:
                return

            self.input.buffer.insert_text(
                "\n",
            )

        @kb.add("a", eager=True)
        def _allow(event) -> None:
            if self.approval.active:
                self.approval.allow()

        @kb.add("d", eager=True)
        def _deny(event) -> None:
            if self.approval.active:
                self.approval.deny()

        @kb.add("pageup")
        def _page_up(event) -> None:
            self._scroll_up(
                amount=10,
            )

        @kb.add("pagedown")
        def _page_down(event) -> None:
            self._scroll_down(
                amount=10,
            )

        @kb.add("c-home")
        def _home(event) -> None:
            self._scroll_home()

        @kb.add("c-end")
        def _end(event) -> None:
            self._scroll_end()

        @kb.add("tab")
        def _toggle(event) -> None:
            self._toggle_last_expandable()

        return kb

    # ------------------------------------------------------------------
    # Expand/collapse
    # ------------------------------------------------------------------

    def _toggle_last_expandable(self) -> None:
        for item in reversed(
            self.state.conversation,
        ):
            if isinstance(item, ThinkingView):
                self.state.toggle_thinking()
                self.invalidate()
                return

            if isinstance(item, ToolView):
                self.state.toggle_tool(
                    item.call_id,
                )
                self.invalidate()
                return

    # ------------------------------------------------------------------
    # Input
    # ------------------------------------------------------------------

    def _accept_input(
        self,
        buffer,
    ) -> None:
        if self.approval.active or self.state.running:
            return

        prompt = buffer.text.strip()

        if not prompt:
            return

        buffer.reset()

        if prompt == "/quit":
            self._exit()
            return

        if prompt == "/clear":
            self.runtime.clear_context()
            self.state.clear_conversation()
            self._scroll_to_end()
            return

        if prompt == "/help":
            self.state.add_assistant_message(
                "Available commands:\n"
                "/help — Show available commands\n"
                "/clear — Clear conversation\n"
                "/quit — Exit agentoflow",
            )
            self._scroll_to_end()
            return

        task = asyncio.create_task(
            self._run_agent(prompt),
        )

        self._tasks.add(task)

        task.add_done_callback(
            self._tasks.discard,
        )

    def _exit(self) -> None:
        for task in tuple(self._tasks):
            task.cancel()

        self.application.exit()

    # ------------------------------------------------------------------
    # Agent
    # ------------------------------------------------------------------

    async def _run_agent(
        self,
        prompt: str,
    ) -> None:
        self.state.running = True

        self.state.add_user_message(
            prompt,
        )

        self._scroll_to_end()

        try:
            if self.state.first_message:
                await self.runtime.agent.run(
                    prompt=prompt,
                    context=self.runtime.context,
                    on_event=self.handle_event,
                )

                self.state.first_message = False

            else:
                await self.runtime.agent.continue_run(
                    prompt=prompt,
                    context=self.runtime.context,
                    on_event=self.handle_event,
                )

        except asyncio.CancelledError:
            raise

        except Exception as exc:
            self.state.stop_thinking()
            self.state.finish_assistant_stream()

            self.state.add_assistant_message(
                f"Error: {exc}",
            )

        finally:
            self.state.running = False
            self.state.stop_thinking()
            self.state.finish_assistant_stream()

            self._scroll_to_end()

    # ------------------------------------------------------------------
    # Agent events
    # ------------------------------------------------------------------

    async def handle_event(
        self,
        event: AgentEvent,
    ) -> None:
        if isinstance(event, AgentStarted):
            pass

        elif isinstance(event, LLMRequested):
            self.state.start_thinking(
                event.iteration,
            )

        elif isinstance(event, LLMThinkingChunk):
            self.state.append_thinking(
                event.content,
            )

        elif isinstance(event, LLMContentChunk):
            self.state.stop_thinking()

            if not self.state.assistant_streaming:
                self.state.start_assistant_stream()

            self.state.append_assistant(
                event.content,
            )

        elif isinstance(event, LLMResponded):
            self.state.stop_thinking()
            self.state.finish_assistant_stream()

        elif isinstance(event, ToolStarted):
            self.state.stop_thinking()

            self.state.add_tool(
                call_id=event.tool_call_id,
                name=event.tool_name,
                arguments=event.arguments,
            )

        elif isinstance(event, ToolFinished):
            self.state.finish_tool(
                call_id=event.tool_call_id,
                output=event.output,
                error_code=event.error_code,
                error_message=event.error_message,
            )

        elif isinstance(event, AgentFinished):
            self.state.stop_thinking()
            self.state.finish_assistant_stream()

            if not self.state.assistant_streamed_current:
                self.state.add_assistant_message(
                    event.result,
                )

        # В follow mode _get_vertical_scroll()
        # самостоятельно пересчитает новый низ.
        #
        # Если пользователь листает историю,
        # его позиция не изменяется.
        self.invalidate()

    # ------------------------------------------------------------------
    # Header
    # ------------------------------------------------------------------

    def _header_text(
        self,
    ) -> StyleAndTextTuples:
        model = self.runtime.config.llm.model
        cwd = self.runtime.context.working_directory

        return [
            ("class:header", " agentoflow"),
            ("", f"   {model}"),
            ("class:dim", f"   {cwd}"),
            ("", "\n"),
            ("class:dim", "─" * 100),
        ]

    # ------------------------------------------------------------------
    # Conversation
    # ------------------------------------------------------------------

    def _conversation_text(
        self,
    ) -> StyleAndTextTuples:
        result: StyleAndTextTuples = []

        for item in self.state.conversation:
            if isinstance(item, UserMessage):
                result.extend(
                    [
                        ("class:user", "\n▌ > "),
                        ("", item.content),
                        ("", "\n"),
                    ],
                )

            elif isinstance(item, ThinkingView):
                result.extend(
                    self._thinking_text(
                        item,
                    ),
                )

            elif isinstance(item, ToolView):
                result.extend(
                    self._tool_text(
                        item,
                    ),
                )

            elif isinstance(item, AssistantMessage):
                result.extend(
                    [
                        ("class:assistant", "\nGemma\n"),
                        ("", item.content),
                        ("", "\n"),
                    ],
                )

        approval = self.approval.request

        if approval is not None:
            result.extend(
                self._approval_text(
                    approval,
                ),
            )

        return result

    # ------------------------------------------------------------------
    # Thinking
    # ------------------------------------------------------------------

    def _thinking_text(
        self,
        thinking: ThinkingView,
    ) -> StyleAndTextTuples:
        symbol = (
            "▾"
            if thinking.expanded
            else "▸"
        )

        result: StyleAndTextTuples = [
            (
                "class:thinking_header",
                f"\n  {symbol} Thinking · "
                f"iteration {thinking.iteration}\n",
            ),
        ]

        if not thinking.expanded:
            return result

        if thinking.content:
            result.append(
                (
                    "class:thinking",
                    "\n".join(
                        f"  {line}"
                        for line in thinking.content.splitlines()
                    )
                    + "\n",
                ),
            )
        else:
            result.append(
                (
                    "class:thinking",
                    "  ⠋ Thinking...\n",
                ),
            )

        return result

    # ------------------------------------------------------------------
    # Tools
    # ------------------------------------------------------------------

    def _tool_text(
        self,
        tool: ToolView,
    ) -> StyleAndTextTuples:
        if tool.status is ToolStatus.RUNNING:
            return self._running_tool_text(
                tool,
            )

        if tool.status is ToolStatus.ERROR:
            return self._error_tool_text(
                tool,
            )

        return self._success_tool_text(
            tool,
        )

    def _running_tool_text(
        self,
        tool: ToolView,
    ) -> StyleAndTextTuples:
        result: StyleAndTextTuples = [
            (
                "class:tool",
                f"\n  ● {tool.name}\n",
            ),
        ]

        for name, value in tool.arguments.items():
            result.extend(
                [
                    (
                        "class:dim",
                        f"      {name}: ",
                    ),
                    (
                        "",
                        f"{value}\n",
                    ),
                ],
            )

        return result

    def _success_tool_text(
        self,
        tool: ToolView,
    ) -> StyleAndTextTuples:
        symbol = (
            "▾"
            if tool.expanded
            else "▸"
        )

        result: StyleAndTextTuples = [
            (
                "class:dim",
                f"\n  {symbol} {tool.name} ✓\n",
            ),
        ]

        if not tool.expanded:
            return result

        for name, value in tool.arguments.items():
            result.extend(
                [
                    (
                        "class:dim",
                        f"      {name}: ",
                    ),
                    (
                        "",
                        f"{value}\n",
                    ),
                ],
            )

        if tool.output:
            result.extend(
                [
                    (
                        "class:dim",
                        "      output:\n",
                    ),
                    (
                        "",
                        f"{tool.output}\n",
                    ),
                ],
            )

        return result

    def _error_tool_text(
        self,
        tool: ToolView,
    ) -> StyleAndTextTuples:
        result: StyleAndTextTuples = [
            (
                "class:error",
                f"\n  ✗ {tool.name}\n",
            ),
        ]

        for name, value in tool.arguments.items():
            result.extend(
                [
                    (
                        "class:dim",
                        f"      {name}: ",
                    ),
                    (
                        "",
                        f"{value}\n",
                    ),
                ],
            )

        if tool.error_code:
            result.append(
                (
                    "class:error",
                    f"      error: {tool.error_code}\n",
                ),
            )

        if tool.error_message:
            result.append(
                (
                    "class:dim",
                    f"      {tool.error_message}\n",
                ),
            )

        return result

    # ------------------------------------------------------------------
    # Approval
    # ------------------------------------------------------------------

    def _approval_text(
        self,
        request,
    ) -> StyleAndTextTuples:
        result: StyleAndTextTuples = [
            (
                "class:approval",
                "\n"
                "┌─ Permission required "
                "──────────────────────────────┐\n",
            ),
            (
                "",
                f"│ tool:       {request.tool_name}\n",
            ),
            (
                "",
                f"│ permission: {request.permission}\n",
            ),
            (
                "",
                "│\n",
            ),
            (
                "",
                f"│ {request.reason}\n",
            ),
        ]

        for name, value in request.arguments.items():
            result.append(
                (
                    "",
                    f"│ {name}: {value}\n",
                ),
            )

        result.extend(
            [
                (
                    "",
                    "│\n",
                ),
                (
                    "class:approval",
                    "│        [a] Allow        [d] Deny\n",
                ),
                (
                    "class:approval",
                    "└───────────────────────────────────────────┘\n",
                ),
            ],
        )

        return result

    # ------------------------------------------------------------------
    # Footer
    # ------------------------------------------------------------------

    def _footer_text(
        self,
    ) -> StyleAndTextTuples:
        if self.approval.active:
            return [
                (
                    "class:approval",
                    " permission required · a allow · d deny",
                ),
            ]

        if self.state.running:
            return [
                (
                    "class:dim",
                    f" iter {self.state.iteration}"
                    "   working...",
                ),
            ]

        return [
            (
                "class:dim",
                f" iter {self.state.iteration}"
                f"   model {self.runtime.config.llm.model}"
                "   Shift+Enter newline",
            ),
        ]

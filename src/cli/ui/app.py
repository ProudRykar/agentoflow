from __future__ import annotations

import asyncio
import shutil
import subprocess
from typing import Any

from rich.console import Console, Group
from rich.markdown import Markdown
from rich.panel import Panel
from rich.segment import Segment
from rich.text import Text
from textual import events
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Button, Label, Static, TextArea

from cli.approval import ApprovalController
from cli.ui.state import (
    AgentRunView,
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
from core.entities.models.planner import Planner
from core.entities.models.task_contract import TaskContract
from core.entities.models.task_plan import TaskPlan


class AgentInput(TextArea):
    """
    Main agent input with Enter-to-send and Shift+Enter newline.
    """

    def _on_key(
        self,
        event: events.Key,
    ) -> None:
        if event.key == "enter":
            event.prevent_default()
            event.stop()

            if isinstance(
                self.app,
                AgentUI,
            ):
                self.app.action_submit()

            return

        if event.key == "shift+enter":
            event.prevent_default()
            event.stop()

            self.insert(
                "\n",
            )

            return

        super()._on_key(
            event,
        )


class ConversationView(Static):
    """
    Rich-powered conversation renderer.

    Rich performs the Markdown / Panel / code layout first.
    The final result is converted to one Text object so Textual
    can select it using its native selection system.
    """

    ALLOW_SELECT = True

    def __init__(
        self,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            **kwargs,
            markup=False,
        )

        self._renderables: list[object] = []

        self._rendered_text: Text | None = None

        self._render_width = 0

    def set_renderables(
        self,
        renderables: list[object],
    ) -> None:
        self._renderables = list(
            renderables,
        )

        self._rendered_text = None

        self.refresh(
            layout=True,
        )

    def _build_text(
        self,
        width: int,
    ) -> Text:
        console = Console(
            width=width,
            force_terminal=True,
            color_system="truecolor",
            markup=False,
        )

        if self._renderables:
            renderable: object = Group(
                *self._renderables,
            )
        else:
            renderable = Text("")

        options = console.options.update_width(
            width,
        )

        lines: list[list[Segment]] = console.render_lines(
            renderable,
            options,
            pad=False,
            new_lines=False,
        )

        text = Text()

        for line_index, line in enumerate(
            lines,
        ):
            for segment in line:
                if not segment.text:
                    continue

                text.append(
                    segment.text,
                    style=segment.style,
                )

            if line_index < len(lines) - 1:
                text.append("\n")

        return text

    def render(self) -> Text:
        width = max(
            1,
            self.content_region.width,
        )

        if (
            self._rendered_text is None
            or self._render_width != width
        ):
            self._rendered_text = self._build_text(
                width,
            )

            self._render_width = width

        return self._rendered_text

    def get_selection(
        self,
        selection: Any,
    ) -> tuple[str, str] | None:
        """
        Convert Textual's screen selection into text.
        """

        text = self._rendered_text

        if text is None:
            text = self.render()

        plain_text = str(text)

        if not plain_text:
            return None

        try:
            selected = selection.extract(
                plain_text,
            )
        except AttributeError:
            return None

        if not selected:
            return None

        return selected, "\n"


class Transcript(VerticalScroll):
    """
    Scrollable conversation area.

    It never receives keyboard focus.
    """

    def __init__(
        self,
        owner: AgentUI,
    ) -> None:
        super().__init__(
            id="transcript",
            can_focus=False,
            can_focus_children=False,
        )

        self.owner = owner

    def _on_mouse_scroll_up(
        self,
        event: events.MouseScrollUp,
    ) -> None:
        self.owner._follow_output = False

        super()._on_mouse_scroll_up(
            event,
        )

    def _on_mouse_scroll_down(
        self,
        event: events.MouseScrollDown,
    ) -> None:
        super()._on_mouse_scroll_down(
            event,
        )

        self.owner._follow_output = (
            self.is_vertical_scroll_end
        )


class AgentUI(App[None]):
    """
    agentoflow TUI built on Textual + Rich.
    """

    TITLE = "agentoflow"

    ALLOW_SELECT = True

    ENABLE_COMMAND_PALETTE = False

    BINDINGS = [
        Binding(
            "ctrl+up",
            "scroll_up",
            "Scroll up",
            show=False,
        ),
        Binding(
            "ctrl+down",
            "scroll_down",
            "Scroll down",
            show=False,
        ),
        Binding(
            "pageup",
            "page_up",
            "Page up",
            show=False,
        ),
        Binding(
            "pagedown",
            "page_down",
            "Page down",
            show=False,
        ),
        Binding(
            "ctrl+home",
            "scroll_home",
            "Scroll home",
            show=False,
        ),
        Binding(
            "ctrl+end",
            "scroll_end",
            "Scroll end",
            show=False,
        ),
        Binding(
            "tab",
            "toggle_last",
            "Toggle",
            show=False,
            priority=True,
        ),
        Binding(
            "ctrl+shift+c",
            "copy_text",
            "Copy",
            show=False,
            priority=True,
        ),
    ]

    CSS = """
    Screen {
        background: #090d0b;
        color: #c9d4cc;
    }

    /* ===========================================================
       ROOT
       =========================================================== */

    #root {
        width: 1fr;
        height: 1fr;
    }

    /* ===========================================================
       HEADER
       =========================================================== */

    #header {
        width: 1fr;
        height: 3;
        min-height: 3;
        max-height: 3;

        background: #0d1310;
        border-bottom: solid #1a2920;

        padding: 0 1;
    }

    #header-title {
        width: 1fr;
        height: 1;

        color: #83b98d;
        text-style: bold;
    }

    #header-meta {
        width: 1fr;
        height: 1;

        color: #607065;
    }

    /* ===========================================================
       TRANSCRIPT
       =========================================================== */

    #transcript {
        width: 1fr;
        height: 1fr;
        min-height: 1;

        padding: 1 2;

        background: #090d0b;

        scrollbar-gutter: stable;
        scrollbar-size: 1 1;

        scrollbar-color: #26392e;
        scrollbar-color-hover: #365643;
        scrollbar-color-active: #477055;

        scrollbar-background: #090d0b;
    }

    #conversation {
        width: 1fr;
        height: auto;

        background: #090d0b;
        padding: 0;
    }

    /* ===========================================================
       STATUS
       =========================================================== */

    #status-bar {
        width: 1fr;
        height: 1;
        min-height: 1;
        max-height: 1;

        background: #0d1310;
        border-top: solid #1a2920;

        padding: 0 1;

        align: center middle;
    }

    #status-left {
        width: 1fr;
        min-width: 0;

        height: 1;

        color: #79ad83;

        content-align: left middle;
        overflow-x: hidden;
    }

    #status-right {
        width: 40;
        min-width: 40;
        max-width: 40;

        height: 1;

        color: #526158;

        content-align: right middle;
        overflow-x: hidden;
    }

    /* ===========================================================
       APPROVAL INFO
       =========================================================== */

    #approval-info {
        width: 1fr;
        height: auto;

        padding: 0 2;

        background: #11150f;

        border-top: solid #28351f;
        border-bottom: solid #28351f;

        color: #9baa94;
    }

    /* ===========================================================
       INPUT
       =========================================================== */

    #input-container {
        width: 1fr;

        height: 4;
        min-height: 4;
        max-height: 4;

        padding: 0 1;

        background: #0d1310;
        border-top: solid #1a2920;

        align: center middle;
    }

    #input-prompt {
        width: 2;
        height: 3;

        padding: 0;

        color: #78b684;
        text-style: bold;

        content-align: center middle;
    }

    #input {
        width: 1fr;

        height: 3;
        min-height: 3;
        max-height: 3;

        background: #090d0b;

        border: round #294334;

        color: #d0ddd3;

        padding: 0 1;
    }

    #input:focus {
        border: round #4f7c5b;
    }

    /* ===========================================================
       APPROVAL BUTTONS
       =========================================================== */

    #approval-actions {
        width: 1fr;

        height: 3;
        min-height: 3;
        max-height: 3;

        display: none;

        align: center middle;
    }

    #approval-actions > Button {
        width: 14;
        height: 3;

        margin: 0 1;
    }

    #allow {
        background: #1a3b24;
        color: #9dcc9f;

        border: round #4f865b;
    }

    #allow:focus {
        background: #23522f;
        border: round #78b684;
    }

    #deny {
        background: #14273a;
        color: #91afd0;

        border: round #385776;
    }

    #deny:focus {
        background: #1b3750;
        border: round #6e9ac8;
    }

    /* ===========================================================
       SELECTION
       =========================================================== */

    Screen > .-textual-system-selection {
        background: #294a33;
        color: #e0ece2;
    }
    """

    def __init__(
        self,
        runtime,
        approval: ApprovalController,
    ) -> None:
        super().__init__()

        self.runtime = runtime

        self.approval = approval

        self.state = UIState()

        self._tasks: set[
            asyncio.Task[Any]
        ] = set()

        # run_id субагента -> slot в основной conversation.
        self._subagent_slots: dict[
            str,
            int,
        ] = {}

        # Stable launch order.
        self._subagent_order: list[
            str
        ] = []

        # True:
        #   автоматически следуем за новым выводом.
        #
        # False:
        #   пользователь вручную смотрит историю.
        self._follow_output = True

        self._spinner_frames = (
            "⠋",
            "⠙",
            "⠹",
            "⠸",
            "⠼",
            "⠴",
            "⠦",
            "⠧",
            "⠇",
            "⠏",
        )

        self._spinner_index = 0

        self._last_render_signature = ""

        self._last_copied_selection = ""

        self._shutting_down = False

        self.transcript: Transcript | None = None

        self.conversation: ConversationView | None = None

        self.status_left: Static | None = None

        self.status_right: Static | None = None

        self.approval_info: Static | None = None

        self.approval_actions: Horizontal | None = None

        self.input_container: Horizontal | None = None

        self.input_prompt: Label | None = None

        self.input: AgentInput | None = None

        self.approval.set_on_change(
            self._on_approval_change,
        )

    # ===========================================================
    # Compose
    # ===========================================================

    def compose(self) -> ComposeResult:
        with Vertical(
            id="root",
        ):
            with Static(
                id="header",
            ):
                yield Static(
                    " agentoflow",
                    id="header-title",
                )

                yield Static(
                    "",
                    id="header-meta",
                )

            yield Transcript(
                self,
            )

            with Horizontal(
                id="status-bar",
            ):
                yield Static(
                    "",
                    id="status-left",
                )

                yield Static(
                    "",
                    id="status-right",
                )

            yield Static(
                "",
                id="approval-info",
            )

            with Horizontal(
                id="input-container",
            ):
                yield Label(
                    ">",
                    id="input-prompt",
                )

                yield AgentInput(
                    "",
                    id="input",
                    soft_wrap=True,
                    show_line_numbers=False,
                    placeholder="Message agentoflow…",
                )

                with Horizontal(
                    id="approval-actions",
                ):
                    yield Button(
                        "Allow",
                        id="allow",
                    )

                    yield Button(
                        "Deny",
                        id="deny",
                    )

    # ===========================================================
    # Lifecycle
    # ===========================================================

    async def run(self) -> None:
        """
        Async entrypoint.

        commands.py already runs inside asyncio.run(),
        therefore Textual's synchronous App.run() must not be called.
        """

        await self.run_async(
            mouse=True,
        )

    async def on_mount(self) -> None:
        self.transcript = self.query_one(
            "#transcript",
            Transcript,
        )

        self.conversation = ConversationView(
            id="conversation",
        )

        await self.transcript.mount(
            self.conversation,
        )

        self.status_left = self.query_one(
            "#status-left",
            Static,
        )

        self.status_right = self.query_one(
            "#status-right",
            Static,
        )

        self.approval_info = self.query_one(
            "#approval-info",
            Static,
        )

        self.input_container = self.query_one(
            "#input-container",
            Horizontal,
        )

        self.input_prompt = self.query_one(
            "#input-prompt",
            Label,
        )

        self.input = self.query_one(
            "#input",
            AgentInput,
        )

        self.approval_actions = self.query_one(
            "#approval-actions",
            Horizontal,
        )

        self.set_interval(
            0.08,
            self._tick_spinner,
        )

        self._render_header()

        self._render_status()

        self._render_approval()

        self._render_conversation()

        self._focus_input()

    # ===========================================================
    # Spinner
    # ===========================================================

    def _tick_spinner(self) -> None:
        if self._shutting_down:
            return

        main_thinking = self.state.thinking

        subagent_thinking = any(
            run.thinking
            for run in self.state.runs.values()
        )

        if (
            not main_thinking
            and not subagent_thinking
        ):
            return

        self._spinner_index = (
            self._spinner_index + 1
        ) % len(
            self._spinner_frames,
        )

        self._render_conversation()

    # ===========================================================
    # Clipboard
    # ===========================================================

    def action_copy_text(self) -> None:
        selected_text = self.screen.get_selected_text()

        if selected_text:
            self.copy_to_clipboard(
                selected_text,
            )

    def copy_to_clipboard(
        self,
        text: str,
    ) -> None:
        if not text:
            return

        try:
            super().copy_to_clipboard(
                text,
            )
        except Exception:
            pass

        if shutil.which(
            "wl-copy",
        ):
            try:
                subprocess.run(
                    ["wl-copy"],
                    input=text,
                    text=True,
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except (
                OSError,
                subprocess.SubprocessError,
            ):
                pass

    def on_mouse_up(
        self,
        event: events.MouseUp,
    ) -> None:
        del event

        self.call_after_refresh(
            self._copy_selection_after_mouse,
        )

    def _copy_selection_after_mouse(self) -> None:
        if self._shutting_down:
            return

        selected_text = self.screen.get_selected_text()

        if not selected_text:
            self._last_copied_selection = ""
            return

        if (
            selected_text
            == self._last_copied_selection
        ):
            return

        self._last_copied_selection = selected_text

        self.copy_to_clipboard(
            selected_text,
        )

    # ===========================================================
    # Focus
    # ===========================================================

    def _focus_input(self) -> None:
        if self._shutting_down:
            return

        if self.input is None:
            return

        if not self.input.is_attached:
            return

        self.set_focus(
            self.input,
        )

    def _focus_allow(self) -> None:
        if self._shutting_down:
            return

        if self.approval_actions is None:
            return

        if not self.approval_actions.is_attached:
            return

        allow = self.query_one(
            "#allow",
            Button,
        )

        if not allow.is_attached:
            return

        if not allow.display:
            return

        self.set_focus(
            allow,
        )

    # ===========================================================
    # Header
    # ===========================================================

    def _render_header(self) -> None:
        title = self.query_one(
            "#header-title",
            Static,
        )

        meta = self.query_one(
            "#header-meta",
            Static,
        )

        model = self.runtime.config.llm.model

        cwd = self.runtime.context.working_directory

        title.update(
            Text(
                " agentoflow",
                style="#83b98d bold",
            ),
        )

        meta.update(
            Text(
                f" {model}  ·  {cwd}",
                style="#607065",
            ),
        )

    # ===========================================================
    # Status
    # ===========================================================

    def _render_status(self) -> None:
        if (
            self.status_left is None
            or self.status_right is None
        ):
            return

        model = self.runtime.config.llm.model

        if self.approval.active:
            self.status_left.update(
                Text(
                    f"● permission  ·  {model}",
                    style="#a8b899",
                ),
            )

            self.status_right.update(
                Text(
                    "Tab switch  ·  Enter choose",
                    style="#657267",
                ),
            )

            return

        if self.state.running:
            self.status_left.update(
                Text(
                    f"● working  ·  iter {self.state.iteration}"
                    f"  ·  {model}",
                    style="#79ad83",
                ),
            )

        else:
            self.status_left.update(
                Text(
                    f"○ idle  ·  iter {self.state.iteration}"
                    f"  ·  {model}",
                    style="#607065",
                ),
            )

        self.status_right.update(
            Text(
                "Enter send  ·  Shift+Enter newline"
                "  ·  select = copy",
                style="#526158",
            ),
        )

    # ===========================================================
    # Approval
    # ===========================================================

    def _render_approval(self) -> None:
        if (
            self.approval_info is None
            or self.approval_actions is None
            or self.input is None
            or self.input_prompt is None
        ):
            return

        request = self.approval.request

        if request is None:
            self.approval_info.update(
                "",
            )

            self.approval_info.display = False

            self.approval_actions.display = False

            self.input_prompt.display = True

            self.input.display = True

            self.input.disabled = False

            return

        lines = [
            Text(
                "permission required",
                style="#a8b899 bold",
            ),
            Text.assemble(
                ("  tool: ", "#657267"),
                (
                    request.tool_name,
                    "#9fc5a5 bold",
                ),
            ),
            Text.assemble(
                ("  permission: ", "#657267"),
                (
                    request.permission,
                    "#89a7ca",
                ),
            ),
        ]

        if request.reason:
            lines.append(
                Text(
                    f"  {request.reason}",
                    style="#95a392",
                ),
            )

        for name, value in request.arguments.items():
            lines.append(
                Text.assemble(
                    (f"  {name}: ", "#657267"),
                    (
                        str(value),
                        "#aab9ad",
                    ),
                ),
            )

        self.approval_info.update(
            Group(
                *lines,
            ),
        )

        self.approval_info.display = True

        self.input_prompt.display = False

        self.input.display = False

        self.input.disabled = True

        self.approval_actions.display = True

    def on_button_pressed(
        self,
        event: Button.Pressed,
    ) -> None:
        button_id = event.button.id

        if button_id == "allow":
            if self.approval.active:
                self.approval.allow()

            return

        if button_id == "deny":
            if self.approval.active:
                self.approval.deny()

            return

    def _on_approval_change(self) -> None:
        if self._shutting_down:
            return

        self.call_after_refresh(
            self._apply_approval_state,
        )

    def _apply_approval_state(self) -> None:
        if self._shutting_down:
            return

        self._render_approval()

        self._render_status()

        if self.approval.active:
            self._focus_allow()

        else:
            self._focus_input()

    # ===========================================================
    # Conversation refresh
    # ===========================================================

    def invalidate(
        self,
        *,
        conversation_changed: bool = True,
    ) -> None:
        if self._shutting_down:
            return

        if conversation_changed:
            self._render_conversation()

        self._render_status()

        self._render_approval()

        if (
            self._follow_output
            and self.transcript is not None
        ):
            self.call_after_refresh(
                self._scroll_follow_output,
            )

        self.refresh()

    def _scroll_follow_output(self) -> None:
        if self._shutting_down:
            return

        if not self._follow_output:
            return

        if self.transcript is None:
            return

        self.transcript.scroll_end(
            animate=False,
            immediate=True,
        )

    # ===========================================================
    # Conversation
    # ===========================================================

    def _render_conversation(self) -> None:
        if self.conversation is None:
            return

        signature = self._conversation_signature()

        if (
            signature
            == self._last_render_signature
        ):
            return

        self._last_render_signature = signature

        self.conversation.set_renderables(
            self._conversation_renderables(),
        )

    def _conversation_signature(self) -> str:
        parts: list[str] = []

        for item in self.state.conversation:
            parts.append(
                repr(item),
            )

        for run_id in self._subagent_order:
            run = self.state.get_run(
                run_id,
            )

            if run is not None:
                parts.append(
                    repr(run),
                )

        request = self.approval.request

        if request is not None:
            parts.append(
                repr(request),
            )

        return "\n".join(parts)

    def _conversation_renderables(self) -> list[object]:
        result: list[object] = []

        subagents_by_slot: dict[
            int,
            list[AgentRunView],
        ] = {}

        for run_id in self._subagent_order:
            slot = self._subagent_slots.get(
                run_id,
            )

            if slot is None:
                continue

            run = self.state.get_run(
                run_id,
            )

            if run is None:
                continue

            subagents_by_slot.setdefault(
                slot,
                [],
            ).append(run)

        self._append_subagents_at_slot(
            result,
            subagents_by_slot,
            0,
        )

        for index, item in enumerate(
            self.state.conversation,
        ):
            if isinstance(
                item,
                UserMessage,
            ):
                result.append(
                    self._render_user(
                        item,
                    ),
                )

            elif isinstance(
                item,
                ThinkingView,
            ):
                result.append(
                    self._render_thinking(
                        item,
                    ),
                )

            elif isinstance(
                item,
                ToolView,
            ):
                result.append(
                    self._render_tool(
                        item,
                    ),
                )

            elif isinstance(
                item,
                AssistantMessage,
            ):
                result.append(
                    self._render_assistant(
                        item,
                    ),
                )

            self._append_subagents_at_slot(
                result,
                subagents_by_slot,
                index + 1,
            )

        return result

    # ===========================================================
    # Main renderables
    # ===========================================================

    @staticmethod
    def _render_user(
        message: UserMessage,
    ) -> object:
        return Panel(
            Text(
                message.content.rstrip(),
                style="#bac8bd",
            ),
            title=Text(
                "▌ User",
                style="#79b982 bold",
            ),
            title_align="left",
            border_style="#34553f",
            padding=(0, 1),
        )

    def _render_thinking(
        self,
        thinking: ThinkingView,
    ) -> object:
        symbol = (
            "▾"
            if thinking.expanded
            else "▸"
        )

        spinner = (
            self._spinner_frames[
                self._spinner_index
            ]
            if thinking.expanded
            else ""
        )

        title = Text.assemble(
            (
                f"{symbol} ",
                "#75877a bold",
            ),
            (
                f"{spinner} ",
                "#78b684 bold",
            ),
            (
                "Thinking",
                "#75877a bold",
            ),
            (
                f"  ·  iteration {thinking.iteration}",
                "#617067",
            ),
        )

        if not thinking.expanded:
            body: object = Text(
                "collapsed",
                style="#536157",
            )

        elif thinking.content:
            body = Markdown(
                thinking.content.rstrip(),
                code_theme="native",
            )

        else:
            body = Text(
                "Thinking…",
                style="#708778",
            )

        return Panel(
            body,
            title=title,
            title_align="left",
            border_style="#2b4032",
            padding=(0, 1),
        )

    @staticmethod
    def _render_tool(
        tool: ToolView,
    ) -> object:
        if tool.status is ToolStatus.RUNNING:
            symbol = "●"
            status = "running"
            border = "#31506e"
            status_style = "#7d9cc0"

        elif tool.status is ToolStatus.ERROR:
            symbol = "✗"
            status = "error"
            border = "#66583a"
            status_style = "#c0a86e"

        else:
            symbol = "▸"
            status = "done"
            border = "#334d3c"
            status_style = "#73977d"

        title = Text.assemble(
            (
                f"{symbol} {tool.name}",
                "#86a8d1 bold",
            ),
            (
                f"  ·  {status}",
                status_style,
            ),
        )

        body: list[object] = []

        for name, value in tool.arguments.items():
            body.append(
                Text.assemble(
                    (
                        f"{name}: ",
                        "#65756a",
                    ),
                    (
                        str(value),
                        "#aab8ad",
                    ),
                ),
            )

        if (
            tool.status is ToolStatus.SUCCESS
            and tool.output
        ):
            body.append(
                Text(""),
            )

            body.append(
                Markdown(
                    tool.output.rstrip(),
                    code_theme="native",
                ),
            )

        if tool.status is ToolStatus.ERROR:
            if tool.error_code:
                body.append(
                    Text.assemble(
                        (
                            "error: ",
                            "#9c8858",
                        ),
                        (
                            tool.error_code,
                            "#c0a86e",
                        ),
                    ),
                )

            if tool.error_message:
                body.append(
                    Text(
                        tool.error_message.rstrip(),
                        style="#ae9864",
                    ),
                )

        if not body:
            body.append(
                Text(
                    "waiting…",
                    style="#617067",
                ),
            )

        return Panel(
            Group(
                *body,
            ),
            title=title,
            title_align="left",
            border_style=border,
            padding=(0, 1),
        )

    @staticmethod
    def _render_assistant(
        message: AssistantMessage,
    ) -> object:
        return Group(
            Text(
                "Gemma",
                style="#8eaed1 bold",
            ),
            Markdown(
                message.content.rstrip(),
                code_theme="native",
            ),
        )

    # ===========================================================
    # Subagents
    # ===========================================================

    def _append_subagents_at_slot(
        self,
        result: list[object],
        subagents_by_slot: dict[
            int,
            list[AgentRunView],
        ],
        slot: int,
    ) -> None:
        for run in subagents_by_slot.get(
            slot,
            [],
        ):
            result.append(
                self._render_subagent(
                    run,
                ),
            )

    def _render_subagent(
        self,
        run: AgentRunView,
    ) -> object:
        if run.running:
            status = "running"
            border = "#31506e"
            status_style = "#7d9cc0"

        else:
            status = "finished"
            border = "#304438"
            status_style = "#709278"

        title = Text.assemble(
            (
                "Subagent ",
                "#71869c",
            ),
            (
                run.run_id,
                "#8aa8ce bold",
            ),
            (
                f"  ·  {status}",
                status_style,
            ),
        )

        body: list[object] = []

        for item in run.conversation:
            if isinstance(
                item,
                ThinkingView,
            ):
                body.append(
                    self._render_subagent_thinking(
                        item,
                    ),
                )

            elif isinstance(
                item,
                ToolView,
            ):
                body.append(
                    self._render_subagent_tool(
                        item,
                    ),
                )

            elif isinstance(
                item,
                AssistantMessage,
            ):
                body.extend(
                    [
                        Text(
                            "Assistant",
                            style="#8eaed1 bold",
                        ),
                        Markdown(
                            item.content.rstrip(),
                            code_theme="native",
                        ),
                    ],
                )

        if not body:
            body.append(
                Text(
                    "⠋ waiting for agent…",
                    style="#607067",
                ),
            )

        return Panel(
            Group(
                *body,
            ),
            title=title,
            title_align="left",
            border_style=border,
            padding=(0, 1),
        )

    def _render_subagent_thinking(
        self,
        thinking: ThinkingView,
    ) -> object:
        symbol = (
            "▾"
            if thinking.expanded
            else "▸"
        )

        spinner = (
            self._spinner_frames[
                self._spinner_index
            ]
            if thinking.expanded
            else ""
        )

        title = Text.assemble(
            (
                f"{symbol} ",
                "#75877a bold",
            ),
            (
                f"{spinner} ",
                "#78b684 bold",
            ),
            (
                "Thinking",
                "#75877a bold",
            ),
            (
                f"  ·  iteration {thinking.iteration}",
                "#617067",
            ),
        )

        if not thinking.expanded:
            body: object = Text(
                "collapsed",
                style="#536157",
            )

        elif thinking.content:
            body = Markdown(
                thinking.content.rstrip(),
                code_theme="native",
            )

        else:
            body = Text(
                "Thinking…",
                style="#708778",
            )

        return Panel(
            body,
            title=title,
            title_align="left",
            border_style="#2b4032",
            padding=(0, 1),
        )

    @staticmethod
    def _render_subagent_tool(
        tool: ToolView,
    ) -> object:
        if tool.status is ToolStatus.RUNNING:
            symbol = "●"
            border = "#31506e"
            status_style = "#7d9cc0"

        elif tool.status is ToolStatus.ERROR:
            symbol = "✗"
            border = "#66583a"
            status_style = "#c0a86e"

        else:
            symbol = "▸"
            border = "#334d3c"
            status_style = "#73977d"

        title = Text.assemble(
            (
                f"{symbol} {tool.name}",
                "#86a8d1 bold",
            ),
            (
                " ✓"
                if tool.status is ToolStatus.SUCCESS
                else "",
                status_style,
            ),
        )

        body: list[object] = []

        for name, value in tool.arguments.items():
            body.append(
                Text.assemble(
                    (
                        f"{name}: ",
                        "#65756a",
                    ),
                    (
                        str(value),
                        "#aab8ad",
                    ),
                ),
            )

        if (
            tool.status is ToolStatus.SUCCESS
            and tool.output
        ):
            body.append(
                Markdown(
                    tool.output.rstrip(),
                    code_theme="native",
                ),
            )

        if tool.status is ToolStatus.ERROR:
            if tool.error_code:
                body.append(
                    Text.assemble(
                        (
                            "error: ",
                            "#9c8858",
                        ),
                        (
                            tool.error_code,
                            "#c0a86e",
                        ),
                    ),
                )

            if tool.error_message:
                body.append(
                    Text(
                        tool.error_message.rstrip(),
                        style="#ae9864",
                    ),
                )

        if not body:
            body.append(
                Text(
                    "waiting…",
                    style="#617067",
                ),
            )

        return Panel(
            Group(
                *body,
            ),
            title=title,
            title_align="left",
            border_style=border,
            padding=(0, 1),
        )

    # ===========================================================
    # Input
    # ===========================================================

    def action_submit(self) -> None:
        if self.approval.active:
            return

        if self.state.running:
            return

        if self.input is None:
            return

        prompt = self.input.text.strip()

        if not prompt:
            return

        self.input.clear()

        if prompt == "/quit":
            self.action_quit()

            return

        if prompt == "/clear":
            self._command_clear()

            return

        if prompt == "/help":
            self._command_help()

            return

        task = asyncio.create_task(
            self._run_agent(
                prompt,
            ),
        )

        self._tasks.add(
            task,
        )

        task.add_done_callback(
            self._tasks.discard,
        )

    def _command_clear(self) -> None:
        self.runtime.clear_context()

        self.state.clear_conversation()

        self._subagent_slots.clear()

        self._subagent_order.clear()

        self._follow_output = True

        self._last_render_signature = ""

        self._last_copied_selection = ""

        try:
            self.clear_selection()
        except Exception:
            pass

        if self.transcript is not None:
            self.transcript.scroll_home(
                animate=False,
                immediate=True,
            )

        if self.conversation is not None:
            self.conversation.set_renderables(
                [],
            )

        self.invalidate()

        self._focus_input()

    def _command_help(self) -> None:
        self.state.add_assistant_message(
            "## Commands\n\n"
            "`/help` — show available commands\n\n"
            "`/clear` — clear the conversation\n\n"
            "`/quit` — exit agentoflow",
        )

        self._follow_output = True

        self.invalidate()

        self._focus_input()

    # ===========================================================
    # Agent
    # ===========================================================

    async def _run_agent(
        self,
        prompt: str,
    ) -> None:
        self.state.running = True

        self.state.add_user_message(
            prompt,
        )

        self._follow_output = True

        self.invalidate()

        planner = Planner()

        plan: TaskPlan = planner.plan(
            prompt,
        )

        research = plan.research

        if (
            research is None
            or not research.root_urls
        ):
            research = None

        task_contract = TaskContract(
            requires_research=(
                research is not None
            ),
            research=research,
        )

        try:
            if self.state.first_message:
                await self.runtime.agent.run(
                    prompt=prompt,
                    context=self.runtime.context,
                    on_event=self.handle_event,
                    task_contract=task_contract,
                )

                self.state.first_message = False

            else:
                await self.runtime.agent.continue_run(
                    prompt=prompt,
                    context=self.runtime.context,
                    on_event=self.handle_event,
                    task_contract=task_contract,
                )

        except asyncio.CancelledError:
            raise

        except Exception as exc:
            self.state.stop_thinking()

            self.state.finish_assistant_stream()

            self.state.add_assistant_message(
                f"Error: {exc}",
            )

            self.invalidate()

        finally:
            self.state.running = False

            self.state.stop_thinking()

            self.state.finish_assistant_stream()

            self._follow_output = True

            self.invalidate()

            self._focus_input()

    # ===========================================================
    # Agent events
    # ===========================================================

    async def handle_event(
        self,
        event: AgentEvent,
    ) -> None:
        if isinstance(
            event,
            AgentStarted,
        ):
            self._handle_agent_started(
                event,
            )

        elif self._is_subagent_event(
            event,
        ):
            self._handle_subagent_event(
                event,
            )

        elif isinstance(
            event,
            LLMRequested,
        ):
            self.state.start_thinking(
                event.iteration,
            )

        elif isinstance(
            event,
            LLMThinkingChunk,
        ):
            self.state.append_thinking(
                event.content,
            )

        elif isinstance(
            event,
            LLMContentChunk,
        ):
            self.state.stop_thinking()

            if not self.state.assistant_streaming:
                self.state.start_assistant_stream()

            self.state.append_assistant(
                event.content,
            )

        elif isinstance(
            event,
            LLMResponded,
        ):
            self.state.stop_thinking()

            self.state.finish_assistant_stream()

        elif isinstance(
            event,
            ToolStarted,
        ):
            self.state.stop_thinking()

            self.state.add_tool(
                call_id=event.tool_call_id,
                name=event.tool_name,
                arguments=event.arguments,
            )

        elif isinstance(
            event,
            ToolFinished,
        ):
            self.state.finish_tool(
                call_id=event.tool_call_id,
                output=event.output,
                error_code=event.error_code,
                error_message=event.error_message,
            )

        elif isinstance(
            event,
            AgentFinished,
        ):
            self.state.stop_thinking()

            self.state.finish_assistant_stream()

            if not self.state.assistant_streamed_current:
                self.state.add_assistant_message(
                    event.result,
                )

            self.state.finish_run(
                event.run_id,
            )

        self.invalidate()

    def _handle_agent_started(
        self,
        event: AgentStarted,
    ) -> None:
        self.state.start_run(
            run_id=event.run_id,
            parent_run_id=event.parent_run_id,
        )

        if event.parent_run_id is not None:
            self.state.active_subagent_run_id = (
                event.run_id
            )

            self._subagent_slots[
                event.run_id
            ] = len(
                self.state.conversation,
            )

            if (
                event.run_id
                not in self._subagent_order
            ):
                self._subagent_order.append(
                    event.run_id,
                )

    def _is_subagent_event(
        self,
        event: AgentEvent,
    ) -> bool:
        run_id = getattr(
            event,
            "run_id",
            None,
        )

        if (
            run_id is None
            or self.state.main_run_id is None
        ):
            return False

        return (
            run_id
            != self.state.main_run_id
        )

    def _handle_subagent_event(
        self,
        event: AgentEvent,
    ) -> None:
        run_id = getattr(
            event,
            "run_id",
            None,
        )

        if run_id is None:
            return

        run = self.state.get_run(
            run_id,
        )

        if run is None:
            run = self.state.start_run(
                run_id=run_id,
                parent_run_id=getattr(
                    event,
                    "parent_run_id",
                    None,
                ),
            )

            self._subagent_slots.setdefault(
                run_id,
                len(
                    self.state.conversation,
                ),
            )

            if (
                run_id
                not in self._subagent_order
            ):
                self._subagent_order.append(
                    run_id,
                )

        self.state.active_subagent_run_id = (
            run_id
        )

        if isinstance(
            event,
            LLMRequested,
        ):
            run.start_thinking(
                event.iteration,
            )

        elif isinstance(
            event,
            LLMThinkingChunk,
        ):
            run.append_thinking(
                event.content,
            )

        elif isinstance(
            event,
            LLMContentChunk,
        ):
            run.stop_thinking()

            if not run.assistant_streaming:
                run.start_assistant_stream()

            run.append_assistant(
                event.content,
            )

        elif isinstance(
            event,
            LLMResponded,
        ):
            run.stop_thinking()

            run.finish_assistant_stream()

        elif isinstance(
            event,
            ToolStarted,
        ):
            run.stop_thinking()

            run.add_tool(
                call_id=event.tool_call_id,
                name=event.tool_name,
                arguments=event.arguments,
            )

        elif isinstance(
            event,
            ToolFinished,
        ):
            run.finish_tool(
                call_id=event.tool_call_id,
                output=event.output,
                error_code=event.error_code,
                error_message=event.error_message,
            )

        elif isinstance(
            event,
            AgentFinished,
        ):
            run.stop_thinking()

            run.finish_assistant_stream()

            if not run.assistant_streamed_current:
                run.add_assistant_message(
                    event.result,
                )

            self.state.finish_run(
                run_id,
            )

    # ===========================================================
    # Scrolling
    # ===========================================================

    def action_scroll_up(self) -> None:
        if self.transcript is None:
            return

        self._follow_output = False

        self.transcript.scroll_relative(
            y=-3,
            animate=False,
            immediate=True,
        )

    def action_scroll_down(self) -> None:
        if self.transcript is None:
            return

        self.transcript.scroll_relative(
            y=3,
            animate=False,
            immediate=True,
        )

        self._follow_output = (
            self.transcript.is_vertical_scroll_end
        )

    def action_page_up(self) -> None:
        if self.transcript is None:
            return

        self._follow_output = False

        height = max(
            1,
            self.transcript.scrollable_content_region.height,
        )

        self.transcript.scroll_relative(
            y=-height,
            animate=False,
            immediate=True,
        )

    def action_page_down(self) -> None:
        if self.transcript is None:
            return

        height = max(
            1,
            self.transcript.scrollable_content_region.height,
        )

        self.transcript.scroll_relative(
            y=height,
            animate=False,
            immediate=True,
        )

        self._follow_output = (
            self.transcript.is_vertical_scroll_end
        )

    def action_scroll_home(self) -> None:
        if self.transcript is None:
            return

        self._follow_output = False

        self.transcript.scroll_home(
            animate=False,
            immediate=True,
        )

    def action_scroll_end(self) -> None:
        if self.transcript is None:
            return

        self._follow_output = True

        self.transcript.scroll_end(
            animate=False,
            immediate=True,
        )

        self._focus_input()

    # ===========================================================
    # Expand / collapse
    # ===========================================================

    def action_toggle_last(self) -> None:
        for item in reversed(
            self.state.conversation,
        ):
            if isinstance(
                item,
                ThinkingView,
            ):
                self.state.toggle_thinking()

                self.invalidate()

                return

            if isinstance(
                item,
                ToolView,
            ):
                self.state.toggle_tool(
                    item.call_id,
                )

                self.invalidate()

                return

        run = self._active_subagent()

        if run is None:
            return

        for item in reversed(
            run.conversation,
        ):
            if isinstance(
                item,
                ThinkingView,
            ):
                run.toggle_thinking()

                self.invalidate()

                return

            if isinstance(
                item,
                ToolView,
            ):
                run.toggle_tool(
                    item.call_id,
                )

                self.invalidate()

                return

    def _active_subagent(
        self,
    ) -> AgentRunView | None:
        run_id = (
            self.state.active_subagent_run_id
        )

        if run_id is None:
            return None

        return self.state.get_run(
            run_id,
        )

    # ===========================================================
    # Exit
    # ===========================================================

    def action_quit(self) -> None:
        if self._shutting_down:
            return

        self._shutting_down = True

        for task in tuple(
            self._tasks,
        ):
            task.cancel()

        self.exit()
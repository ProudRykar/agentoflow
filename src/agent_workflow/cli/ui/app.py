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
from textual.containers import (
    Horizontal,
    Vertical,
    VerticalScroll,
)
from textual.widgets import (
    Button,
    Label,
    Static,
    TextArea,
)

from agent_workflow.cli.approval import ApprovalController
from agent_workflow.cli.ui.state import (
    TOOL_OUTPUT_COLLAPSED_CHARS,
    AgentRunView,
    AssistantMessage,
    ThinkingView,
    ToolStatus,
    ToolView,
    UIState,
    UserMessage,
)
from agent_workflow.core.context.history import HistoryKind
from agent_workflow.core.entities.models.agent_phase import AgentPhase
from agent_workflow.core.entities.models.agent_plan import AgentPlan
from agent_workflow.core.entities.models.agent_trace import (
    AgentEvent,
    AgentFinished,
    AgentPhaseChanged,
    AgentStarted,
    LLMContentChunk,
    LLMRequested,
    LLMResponded,
    LLMThinkingChunk,
    ToolFinished,
    ToolStarted,
)
from agent_workflow.core.entities.models.planner import Planner
from agent_workflow.core.entities.models.task_contract import TaskContract
from agent_workflow.core.entities.models.task_plan import TaskPlan


# ============================================================================
# Input
# ============================================================================


class AgentInput(TextArea):
    """
    Main agent input.

    Enter:
        send message

    Shift+Enter:
        insert newline
    """

    def _on_key(
        self,
        event: events.Key,
    ) -> None:
        if event.key == "enter":
            event.prevent_default()
            event.stop()

            app = self.app

            if isinstance(app, AgentUI):
                app.action_submit()

            return

        if event.key == "shift+enter":
            event.prevent_default()
            event.stop()

            self.insert("\n")
            return

        super()._on_key(event)


# ============================================================================
# Conversation renderer
# ============================================================================


class ConversationView(Static):
    """
    Rich-powered conversation renderer.

    Rich renders Markdown, panels and code blocks.
    The rendered Rich segments are converted to Text so
    Textual native selection continues to work.
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

        lines: list[list[Segment]] = (
            console.render_lines(
                renderable,
                options,
                pad=False,
                new_lines=False,
            )
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
            self._rendered_text = (
                self._build_text(width)
            )

            self._render_width = width

        return self._rendered_text

    def get_selection(
        self,
        selection: Any,
    ) -> tuple[str, str] | None:
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


# ============================================================================
# Transcript
# ============================================================================


class Transcript(VerticalScroll):
    """
    Scrollable conversation area.

    Scrolling upward disables follow mode.
    Returning to the bottom enables it again.
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


# ============================================================================
# Main application
# ============================================================================


class AgentUI(App[None]):
    """
    agentoflow Textual UI.

    Architecture:

        Planner
            ↓
        TaskPlan
            ↓
        TaskContract
            ↓
        Agent / Runtime
            ↓
        AgentEvent
            ↓
        UI

    Runtime owns the authoritative execution state.
    UI observes and renders runtime state.

    Approval is controlled by ApprovalController.
    UI only renders the current request and sends
    allow / deny decisions back to the controller.
    """

    TITLE = "agentoflow"

    ALLOW_SELECT = True

    ENABLE_COMMAND_PALETTE = False

    BINDINGS = [
        # --------------------------------------------------------------
        # Scrolling
        # --------------------------------------------------------------

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

        # --------------------------------------------------------------
        # Approval
        # --------------------------------------------------------------

        Binding(
            "a",
            "approval_allow",
            "Allow",
            show=False,
        ),
        Binding(
            "d",
            "approval_deny",
            "Deny",
            show=False,
        ),

        # --------------------------------------------------------------
        # Toggle
        # --------------------------------------------------------------

        Binding(
            "tab",
            "toggle_last",
            "Toggle",
            show=False,
            priority=True,
        ),

        # --------------------------------------------------------------
        # Clipboard
        # --------------------------------------------------------------

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
        width: 100%;
        height: 100%;
    }

    /* ===========================================================
    HEADER
    =========================================================== */

    #header {
        width: 100%;
        height: 3;
        min-height: 3;
        max-height: 3;

        background: #0d1310;

        border-bottom: solid #1a2920;

        padding: 0 1;
    }

    #header-title {
        width: 100%;
        height: 1;

        color: #83b98d;
        text-style: bold;
    }

    #header-meta {
        width: 100%;
        height: 1;

        color: #607065;
    }

    /* ===========================================================
    TRANSCRIPT
    =========================================================== */

    #transcript {
        width: 100%;
        height: 1fr;
        min-height: 0;

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
        width: 100%;
        height: auto;

        padding: 0;

        background: #090d0b;
    }

    /* ===========================================================
    BOTTOM PANEL
    =========================================================== */

    #bottom-panel {
        width: 100%;

        height: auto;
        min-height: 8;

        background: #0d1310;
    }

    /* ===========================================================
    RUNTIME INFO
    =========================================================== */

    #runtime-info {
        width: 100%;

        height: 4;
        min-height: 4;
        max-height: 4;

        background: #0d1310;

        border-top: solid #1a2920;
        border-bottom: solid #1a2920;
    }

    /* ===========================================================
    STATUS
    =========================================================== */

    #status-bar {
        width: 100%;

        height: 1;
        min-height: 1;
        max-height: 1;

        padding: 0 1;

        color: #79ad83;

        content-align: left middle;

        overflow: hidden;
    }

    /* ===========================================================
    STAGE
    =========================================================== */

    #stage-bar {
        width: 100%;

        height: 1;
        min-height: 1;
        max-height: 1;

        padding: 0 1;

        color: #79ad83;

        content-align: left middle;

        overflow: hidden;
    }

    /* ===========================================================
    APPROVAL INFO
    =========================================================== */

    #approval-info {
        width: 100%;

        height: auto;
        min-height: 0;

        padding: 0 2;

        background: #11150f;

        border-top: solid #28351f;
        border-bottom: solid #28351f;

        color: #9baa94;

        display: none;
    }

    /* ===========================================================
    INPUT
    =========================================================== */

    #input-container {
        width: 100%;

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
    APPROVAL ACTIONS
    =========================================================== */

    #approval-actions {
        width: 100%;

        height: 4;
        min-height: 4;
        max-height: 4;

        display: none;

        background: #0d1310;

        border-top: solid #1a2920;

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

    # ========================================================================
    # Initialization
    # ========================================================================

    def __init__(
        self,
        runtime,
        approval: ApprovalController,
        model_context_size: int | None = None,
    ) -> None:
        super().__init__()

        self.runtime = runtime
        self.approval = approval

        self.state = UIState()
        self.state.model_context_size = model_context_size

        self._tasks: set[
            asyncio.Task[Any]
        ] = set()

        # ----------------------------------------------------------
        # Subagents
        # ----------------------------------------------------------

        self._subagent_slots: dict[
            str,
            int,
        ] = {}

        self._subagent_order: list[str] = []

        # ----------------------------------------------------------
        # Scrolling
        # ----------------------------------------------------------

        self._follow_output = True

        # ----------------------------------------------------------
        # Spinner
        # ----------------------------------------------------------

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

        # ----------------------------------------------------------
        # Conversation rendering
        # ----------------------------------------------------------

        self._last_render_signature = ""
        self._last_copied_selection = ""

        # ----------------------------------------------------------
        # Shutdown
        # ----------------------------------------------------------

        self._shutting_down = False

        # ----------------------------------------------------------
        # Runtime state mirrored by UI
        # ----------------------------------------------------------

        self._current_phase = AgentPhase.IDLE
        self._current_phase_reason = ""

        self._task_plan: TaskPlan | None = None
        self._execution_plan: AgentPlan | None = None

        # ----------------------------------------------------------
        # Approval state
        # ----------------------------------------------------------

        self._approval_focus = "allow"
        self._approval_signature = ""

        # ----------------------------------------------------------
        # Widgets
        # ----------------------------------------------------------

        self.transcript: Transcript | None = None
        self.conversation: ConversationView | None = None

        self.status_bar: Static | None = None
        self.stage_bar: Static | None = None

        self.approval_info: Static | None = None
        self.approval_actions: Horizontal | None = None

        self.input_container: Horizontal | None = None
        self.input_prompt: Label | None = None
        self.input: AgentInput | None = None

        self.approval.set_on_change(
            self._on_approval_change,
        )

    # ========================================================================
    # Compose
    # ========================================================================

    def compose(self) -> ComposeResult:
        with Vertical(id="root"):

            with Vertical(id="header"):
                yield Static(
                    " agentoflow",
                    id="header-title",
                )

                yield Static(
                    "",
                    id="header-meta",
                )

            yield Transcript(self)

            with Vertical(id="bottom-panel"):

                with Vertical(id="runtime-info"):
                    yield Static(
                        "",
                        id="status-bar",
                    )

                    yield Static(
                        "",
                        id="stage-bar",
                    )

                yield Static(
                    "",
                    id="approval-info",
                )

                with Horizontal(id="input-container"):
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

                with Horizontal(id="approval-actions"):
                    yield Button(
                        "Allow",
                        id="allow",
                    )

                    yield Button(
                        "Deny",
                        id="deny",
                    )

    # ========================================================================
    # Lifecycle
    # ========================================================================

    async def run(self) -> None:
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

        self.status_bar = self.query_one(
            "#status-bar",
            Static,
        )

        self.stage_bar = self.query_one(
            "#stage-bar",
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

        self._sync_runtime_plan()

        self._render_header()
        self._render_status()
        self._render_stage()
        self._render_approval()
        self._render_conversation()

        self._focus_input()

    # ========================================================================
    # Runtime helpers
    # ========================================================================

    def _model_name(self) -> str:
        config = getattr(
            self.runtime,
            "config",
            None,
        )

        llm = getattr(
            config,
            "llm",
            None,
        )

        model = getattr(
            llm,
            "model",
            None,
        )

        if model:
            return str(model)

        return "unknown-model"

    def _working_directory(self) -> str:
        context = getattr(
            self.runtime,
            "context",
            None,
        )

        cwd = getattr(
            context,
            "working_directory",
            None,
        )

        if cwd:
            return str(cwd)

        return "unknown-directory"

    def _orchestrator(self):
        agent = getattr(
            self.runtime,
            "agent",
            None,
        )

        return getattr(
            agent,
            "orchestrator",
            None,
        )

    # ========================================================================
    # Planner / contract
    # ========================================================================

    def _build_task_contract(
        self,
        prompt: str,
    ) -> tuple[
        TaskPlan,
        TaskContract,
    ]:
        """
        Build TaskPlan and TaskContract for one user request.

        Planner is invoked exactly once here.

        The UI does not create AgentPlan.
        AgentPlan remains runtime state.
        """

        planner = Planner()

        task_plan = planner.plan(
            prompt,
        )

        research = task_plan.research

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

        return (
            task_plan,
            task_contract,
        )

    # ========================================================================
    # Spinner
    # ========================================================================

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
        ) % len(self._spinner_frames)

        self._render_conversation()
        self._render_stage()

    # ========================================================================
    # Header
    # ========================================================================

    def _render_header(self) -> None:
        title = self.query_one(
            "#header-title",
            Static,
        )

        meta = self.query_one(
            "#header-meta",
            Static,
        )

        title.update(
            Text(
                " agentoflow",
                style="#83b98d bold",
            ),
        )

        meta.update(
            Text(
                f" {self._model_name()}"
                f"  ·  {self._working_directory()}",
                style="#607065",
            ),
        )

    # ========================================================================
    # Status
    # ========================================================================

    def _context_suffix(self) -> str:
        estimated = self.state.estimated_tokens

        if estimated is None:
            return ""

        text = f"  ·  ctx ~{self._format_tokens(estimated)}"

        if self.state.model_context_size:
            text += (
                f"/{self._format_tokens(self.state.model_context_size)}"
            )

        return text

    def _render_status(self) -> None:
        if self.status_bar is None:
            return

        model = self._model_name()

        if self.approval.active:
            self.status_bar.update(
                Text(
                    f"● permission  ·  {model}",
                    style="#a8b899 bold",
                ),
            )
            return

        if self.state.running:
            status_text = (
                f"● working  ·  iter {self.state.iteration}"
                f"  ·  {model}"
                f"{self._context_suffix()}"
            )

            subagent = self._active_subagent()

            if (
                subagent is not None
                and subagent.running
                and subagent.role != "main"
            ):
                status_text += f"  →  {subagent.role}"

                if subagent.model:
                    status_text += f"/{subagent.model}"

            self.status_bar.update(
                Text(
                    status_text,
                    style="#79ad83",
                ),
            )
            return

        self.status_bar.update(
            Text(
                f"○ idle  ·  iter {self.state.iteration}"
                f"  ·  {model}"
                f"{self._context_suffix()}",
                style="#607065",
            ),
        )

    # ========================================================================
    # Stage
    # ========================================================================

    def _render_stage(self) -> None:
        """
        Render the authoritative runtime phase.

        Preferred source:

            orchestrator.state.phase

        Plan source:

            orchestrator.plan
        """

        if self.stage_bar is None:
            return

        phase = self._current_phase
        plan = self._execution_plan

        current_step = (
            plan.current_step()
            if plan is not None
            else None
        )

        # --------------------------------------------------------------
        # IDLE
        # --------------------------------------------------------------

        if phase is AgentPhase.IDLE:
            self.stage_bar.update(
                Text(
                    "○ IDLE",
                    style="#526158",
                ),
            )
            return

        # --------------------------------------------------------------
        # COMPLETED
        # --------------------------------------------------------------

        if phase is AgentPhase.COMPLETED:
            self.stage_bar.update(
                Text(
                    "✓ COMPLETED",
                    style="#79b982 bold",
                ),
            )
            return

        # --------------------------------------------------------------
        # BLOCKED
        # --------------------------------------------------------------

        if phase is AgentPhase.BLOCKED:
            text = Text(
                "✗ BLOCKED",
                style="#c27c72 bold",
            )

            if self._current_phase_reason:
                text.append(
                    "  ·  "
                    f"{self._current_phase_reason}",
                    style="#80635e",
                )

            self.stage_bar.update(
                text,
            )
            return

        # --------------------------------------------------------------
        # Normal phase
        # --------------------------------------------------------------

        text = Text.assemble(
            (
                "● ",
                "#78b684 bold",
            ),
            (
                phase.value.upper(),
                "#8fc49a bold",
            ),
        )

        if (
            plan is not None
            and current_step is not None
        ):
            try:
                step_index = (
                    plan.steps.index(
                        current_step,
                    )
                    + 1
                )
            except ValueError:
                step_index = 0

            total = len(
                plan.steps,
            )

            text.append(
                f"  ·  step {step_index}/{total}",
                style="#627268",
            )

            if current_step.description:
                text.append(
                    f"  ·  {current_step.description}",
                    style="#83958a",
                )

            if current_step.attempts > 1:
                text.append(
                    f"  ·  attempt "
                    f"{current_step.attempts}",
                    style="#756d55",
                )

        elif self._current_phase_reason:
            text.append(
                "  ·  "
                f"{self._current_phase_reason}",
                style="#627268",
            )

        self.stage_bar.update(
            text,
        )

    def _set_phase(
        self,
        phase: AgentPhase,
        reason: str = "",
    ) -> None:
        self._current_phase = phase
        self._current_phase_reason = reason

        self._sync_runtime_plan()

        self._render_stage()

    # ========================================================================
    # Runtime plan synchronization
    # ========================================================================

    def _sync_runtime_plan(self) -> None:
        """
        Mirror runtime-owned state.

        UI never creates or modifies AgentPlan.
        """

        orchestrator = self._orchestrator()

        if orchestrator is None:
            return

        plan = getattr(
            orchestrator,
            "plan",
            None,
        )

        if isinstance(
            plan,
            AgentPlan,
        ):
            self._execution_plan = plan

        task_plan = getattr(
            orchestrator,
            "task_plan",
            None,
        )

        if isinstance(
            task_plan,
            TaskPlan,
        ):
            self._task_plan = task_plan

    # ========================================================================
    # Approval
    # ========================================================================

    def _render_approval(self) -> None:
        if (
            self.approval_info is None
            or self.approval_actions is None
            or self.input_container is None
            or self.input_prompt is None
            or self.input is None
        ):
            return

        request = self.approval.request

        # --------------------------------------------------------------
        # No active request
        # --------------------------------------------------------------

        if request is None:
            self._approval_signature = ""
            self._approval_focus = "allow"

            self.approval_info.update("")
            self.approval_info.display = False

            self.approval_actions.display = False

            self.input_container.display = True

            self.input_prompt.display = True
            self.input.display = True

            self.input.disabled = False

            return

        # --------------------------------------------------------------
        # New request
        # --------------------------------------------------------------

        signature = repr(
            request,
        )

        if (
            signature
            != self._approval_signature
        ):
            self._approval_signature = signature
            self._approval_focus = "allow"

        # --------------------------------------------------------------
        # Request content
        # --------------------------------------------------------------

        lines: list[object] = [
            Text(
                "permission required",
                style="#a8b899 bold",
            ),
            Text.assemble(
                (
                    "  tool: ",
                    "#657267",
                ),
                (
                    request.tool_name,
                    "#9fc5a5 bold",
                ),
            ),
            Text.assemble(
                (
                    "  permission: ",
                    "#657267",
                ),
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

        for name, value in (
            request.arguments.items()
        ):
            lines.append(
                Text.assemble(
                    (
                        f"  {name}: ",
                        "#657267",
                    ),
                    (
                        str(value),
                        "#aab9ad",
                    ),
                ),
            )

        self.approval_info.update(
            Group(*lines),
        )

        self.approval_info.display = True

        # --------------------------------------------------------------
        # Hide normal input
        # --------------------------------------------------------------

        self.input_container.display = False

        self.input_prompt.display = False
        self.input.display = False
        self.input.disabled = True

        # --------------------------------------------------------------
        # Show approval actions
        # --------------------------------------------------------------

        self.approval_actions.display = True

    def on_button_pressed(
        self,
        event: Button.Pressed,
    ) -> None:
        if not self.approval.active:
            return

        button_id = event.button.id

        if button_id == "allow":
            self._approval_focus = "allow"
            self.approval.allow()
            return

        if button_id == "deny":
            self._approval_focus = "deny"
            self.approval.deny()

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
        self._render_stage()

        if not self.approval.active:
            self._focus_input()
            return

        if self._approval_focus == "deny":
            self._focus_deny()
        else:
            self._focus_allow()

    # ========================================================================
    # Approval actions
    # ========================================================================

    def action_approval_allow(self) -> None:
        if not self.approval.active:
            return

        self._approval_focus = "allow"

        self._focus_allow()

        self.approval.allow()

    def action_approval_deny(self) -> None:
        if not self.approval.active:
            return

        self._approval_focus = "deny"

        self._focus_deny()

        self.approval.deny()

    # ========================================================================
    # Focus
    # ========================================================================

    def _focus_input(self) -> None:
        if self._shutting_down:
            return

        if self.input is None:
            return

        if not self.input.is_attached:
            return

        if not self.input.display:
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

        if not self.approval_actions.display:
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

    def _focus_deny(self) -> None:
        if self._shutting_down:
            return

        if self.approval_actions is None:
            return

        if not self.approval_actions.is_attached:
            return

        if not self.approval_actions.display:
            return

        deny = self.query_one(
            "#deny",
            Button,
        )

        if not deny.is_attached:
            return

        if not deny.display:
            return

        self.set_focus(
            deny,
        )

    # ========================================================================
    # Clipboard
    # ========================================================================

    def action_copy_text(self) -> None:
        selected_text = (
            self.screen.get_selected_text()
        )

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

        if shutil.which("wl-copy"):
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

        selected_text = (
            self.screen.get_selected_text()
        )

        if not selected_text:
            self._last_copied_selection = ""
            return

        if (
            selected_text
            == self._last_copied_selection
        ):
            return

        self._last_copied_selection = (
            selected_text
        )

        self.copy_to_clipboard(
            selected_text,
        )

    # ========================================================================
    # UI invalidation
    # ========================================================================

    def _invalidate_ui(
        self,
        *,
        conversation_changed: bool = True,
    ) -> None:
        """
        Re-render application state.

        IMPORTANT:
        Do not name this method `invalidate`.

        `invalidate()` belongs to Textual Widget.
        """

        if self._shutting_down:
            return

        self._sync_runtime_plan()

        if conversation_changed:
            self._render_conversation()

        self._render_status()
        self._render_stage()
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

    # ========================================================================
    # Conversation rendering
    # ========================================================================

    def _render_conversation(self) -> None:
        if self.conversation is None:
            return

        signature = (
            self._conversation_signature()
        )

        if (
            signature
            == self._last_render_signature
        ):
            return

        self._last_render_signature = (
            signature
        )

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

        return "\n".join(
            parts,
        )

    def _conversation_renderables(
        self,
    ) -> list[object]:
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

    # ========================================================================
    # Main renderables
    # ========================================================================

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
                f"  ·  iteration "
                f"{thinking.iteration}",
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
    def _format_duration(
        seconds: float | None,
    ) -> str:
        if seconds is None:
            return ""

        if seconds < 1:
            return f" · {seconds * 1000:.0f}ms"

        return f" · {seconds:.0f}s"

    @staticmethod
    def _format_tokens(
        value: int | None,
    ) -> str:
        if value is None:
            return "?"

        if value >= 1000:
            return f"{value / 1000:.1f}k"

        return str(value)

    def _history_tool_output(
        self,
        call_id: str,
    ) -> str | None:
        """Full output evicted from the view, if recorded."""

        agent = getattr(
            self.runtime,
            "agent",
            None,
        )
        controller = getattr(
            agent,
            "controller",
            None,
        )
        history = getattr(
            controller,
            "history",
            None,
        )
        by_reference = getattr(
            history,
            "by_reference",
            None,
        )

        if by_reference is None:
            return None

        for item in reversed(by_reference(call_id)):
            if (
                item.kind is HistoryKind.TOOL_RESULT
                and item.content
            ):
                return item.content

        return None

    def _tool_output_body(
        self,
        tool: ToolView,
    ) -> list[object]:
        """Bounded output rendering for RAM and speed.

        Collapsed views show plain-text heads (no Markdown
        parse of megabyte blobs). Full text renders only on
        expand, from the view or, when evicted, from history.
        """

        parts: list[object] = []

        if tool.research_summary:
            parts.append(
                Text(
                    tool.research_summary,
                    style="#73977d",
                ),
            )

        output = tool.output

        if not output:
            return parts

        if not tool.expanded:
            parts.append(
                Text(
                    output[:TOOL_OUTPUT_COLLAPSED_CHARS].rstrip()
                ),
            )

            total = tool.output_full_length or len(output)

            if total > TOOL_OUTPUT_COLLAPSED_CHARS:
                parts.append(
                    Text(
                        f"… (+{total} chars total — expand)",
                        style="#607065",
                    ),
                )

            return parts

        if not tool.output_evicted:
            parts.append(
                Markdown(
                    output.rstrip(),
                    code_theme="native",
                ),
            )

            return parts

        restored = self._history_tool_output(
            tool.call_id,
        )

        if restored:
            parts.append(
                Markdown(
                    restored.rstrip(),
                    code_theme="native",
                ),
            )
        else:
            parts.append(
                Text(output.rstrip()),
            )
            parts.append(
                Text(
                    f"… (+{tool.output_full_length} chars "
                    "evicted from UI memory)",
                    style="#607065",
                ),
            )

        return parts

    def _render_tool(
        self,
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
                f"  ·  {status}"
                f"{self._format_duration(tool.duration_seconds)}",
                status_style,
            ),
        )

        body: list[object] = []

        for name, value in (
            tool.arguments.items()
        ):
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

            body.extend(
                self._tool_output_body(tool),
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
            Group(*body),
            title=title,
            title_align="left",
            border_style=border,
            padding=(0, 1),
        )

    def _render_assistant(
        self,
        message: AssistantMessage,
    ) -> object:
        return Group(
            Text(
                self._model_name(),
                style="#8eaed1 bold",
            ),
            Markdown(
                message.content.rstrip(),
                code_theme="native",
            ),
        )

    # ========================================================================
    # Subagents
    # ========================================================================

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

        if run.role and run.role != "main":
            label = run.role
        else:
            label = run.run_id[:8]

        title_parts: list[tuple[str, str]] = [
            (
                "Subagent ",
                "#71869c",
            ),
            (
                label,
                "#8aa8ce bold",
            ),
        ]

        if run.model:
            title_parts.append(
                (
                    f"  ·  {run.model}",
                    "#8aa8ce",
                ),
            )

        title_parts.append(
            (
                f"  ·  {status}",
                status_style,
            ),
        )

        title = Text.assemble(*title_parts)

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
                            self._model_name(),
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
            Group(*body),
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
                f"  ·  iteration "
                f"{thinking.iteration}",
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

    def _render_subagent_tool(
        self,
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
                (
                    " ✓"
                    if tool.status is ToolStatus.SUCCESS
                    else ""
                )
                + self._format_duration(
                    tool.duration_seconds
                ),
                status_style,
            ),
        )

        body: list[object] = []

        for name, value in (
            tool.arguments.items()
        ):
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
            body.extend(
                self._tool_output_body(tool),
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
            Group(*body),
            title=title,
            title_align="left",
            border_style=border,
            padding=(0, 1),
        )

    # ========================================================================
    # Input / commands
    # ========================================================================

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

        self._current_phase = AgentPhase.IDLE
        self._current_phase_reason = ""

        self._task_plan = None
        self._execution_plan = None

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

        self._invalidate_ui()

        self._focus_input()

    def _command_help(self) -> None:
        self.state.add_assistant_message(
            "## Commands\n\n"
            "`/help` — show available commands\n\n"
            "`/clear` — clear the conversation\n\n"
            "`/quit` — exit agentoflow",
        )

        self._follow_output = True

        self._invalidate_ui()

        self._focus_input()

    # ========================================================================
    # Agent execution
    # ========================================================================

    async def _run_agent(
        self,
        prompt: str,
    ) -> None:
        self.state.running = True

        self.add_user_message(
            prompt,
        )

        self._follow_output = True

        self._invalidate_ui()

        task_plan, task_contract = (
            self._build_task_contract(
                prompt,
            )
        )

        orchestrator = self._orchestrator()

        if orchestrator is None:
            raise RuntimeError(
                "Agent orchestrator is not configured",
            )

        orchestrator.prepare_task(
            task_plan,
            task_contract,
        )

        self._task_plan = task_plan

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

            self._set_phase(
                AgentPhase.BLOCKED,
                str(exc),
            )

            self.state.add_assistant_message(
                f"Error: {exc}",
            )

            self._invalidate_ui()

        finally:
            self.state.running = False

            self.state.stop_thinking()

            self.state.finish_assistant_stream()

            self._follow_output = True

            self._invalidate_ui()

            self._focus_input()

    # ========================================================================
    # Agent events
    # ========================================================================

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
            AgentPhaseChanged,
        ):
            self._handle_phase_changed(
                event,
            )

        elif isinstance(
            event,
            LLMRequested,
        ):
            self.state.start_thinking(
                event.iteration,
            )
            self.state.estimated_tokens = (
                event.estimated_tokens
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
                duration_seconds=event.duration_seconds,
            )

        elif isinstance(
            event,
            AgentFinished,
        ):
            self._handle_agent_finished(
                event,
            )

        self._invalidate_ui()

    def _handle_agent_started(
        self,
        event: AgentStarted,
    ) -> None:
        self.state.start_run(
            run_id=event.run_id,
            parent_run_id=event.parent_run_id,
            model=event.model or None,
            agent_id=event.agent_id,
            role=event.role,
        )

        self._sync_runtime_plan()

        if event.parent_run_id is None:
            return

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

    def _handle_phase_changed(
        self,
        event: AgentPhaseChanged,
    ) -> None:
        phase = getattr(
            event,
            "phase",
            None,
        )

        if isinstance(
            phase,
            AgentPhase,
        ):
            resolved_phase = phase

        elif isinstance(
            phase,
            str,
        ):
            try:
                resolved_phase = AgentPhase(
                    phase,
                )
            except ValueError:
                return

        else:
            return

        reason = getattr(
            event,
            "reason",
            "",
        )

        if reason is None:
            reason = ""

        self._set_phase(
            resolved_phase,
            str(reason),
        )

    def _handle_agent_finished(
        self,
        event: AgentFinished,
    ) -> None:
        self.state.stop_thinking()

        self.state.finish_assistant_stream()

        if not self.state.assistant_streamed_current:
            self.state.add_assistant_message(
                event.result,
            )

        self.state.finish_run(
            event.run_id,
        )

        self._sync_runtime_plan()

        orchestrator = self._orchestrator()

        if orchestrator is None:
            return

        runtime_state = getattr(
            orchestrator,
            "state",
            None,
        )

        if runtime_state is None:
            return

        phase = getattr(
            runtime_state,
            "phase",
            None,
        )

        if isinstance(
            phase,
            AgentPhase,
        ):
            self._set_phase(
                phase,
            )

    def add_user_message(
        self,
        prompt: str,
    ) -> None:
        self.state.add_user_message(
            prompt,
        )

    # ========================================================================
    # Subagent events
    # ========================================================================

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
            AgentStarted,
        ):
            self.state.start_run(
                run_id=run_id,
                parent_run_id=getattr(
                    event,
                    "parent_run_id",
                    None,
                ),
                model=event.model or None,
                agent_id=event.agent_id,
                role=event.role,
            )

            return

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
                duration_seconds=event.duration_seconds,
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

    # ========================================================================
    # Scrolling
    # ========================================================================

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
            self.transcript
            .scrollable_content_region
            .height,
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
            self.transcript
            .scrollable_content_region
            .height,
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

        if self.approval.active:
            if self._approval_focus == "deny":
                self._focus_deny()
            else:
                self._focus_allow()
        else:
            self._focus_input()

    # ========================================================================
    # Expand / collapse
    # ========================================================================

    def action_toggle_last(self) -> None:
        # --------------------------------------------------------------
        # Approval mode
        # --------------------------------------------------------------

        if self.approval.active:
            if self._approval_focus == "allow":
                self._approval_focus = "deny"
                self._focus_deny()
            else:
                self._approval_focus = "allow"
                self._focus_allow()

            return

        # --------------------------------------------------------------
        # Main conversation
        # --------------------------------------------------------------

        for item in reversed(
            self.state.conversation,
        ):
            if isinstance(
                item,
                ThinkingView,
            ):
                self.state.toggle_thinking()

                self._invalidate_ui()

                return

            if isinstance(
                item,
                ToolView,
            ):
                self.state.toggle_tool(
                    item.call_id,
                )

                self._invalidate_ui()

                return

        # --------------------------------------------------------------
        # Active subagent
        # --------------------------------------------------------------

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

                self._invalidate_ui()

                return

            if isinstance(
                item,
                ToolView,
            ):
                run.toggle_tool(
                    item.call_id,
                )

                self._invalidate_ui()

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

    # ========================================================================
    # Exit
    # ========================================================================

    def action_quit(self) -> None:
        if self._shutting_down:
            return

        self._shutting_down = True

        for task in tuple(
            self._tasks,
        ):
            task.cancel()

        self.exit()
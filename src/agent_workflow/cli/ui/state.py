from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from agent_workflow.core.entities.models.research_contract import ResearchResult


class ToolStatus(str, Enum):
    RUNNING = "running"
    SUCCESS = "success"
    ERROR = "error"


# UI memory bounds: views keep a head for instant render.
# Full bodies live in HistoryStore (and EvidenceStore for
# research); RAM-heavy blobs are evicted from the view.
TOOL_OUTPUT_PREVIEW_CHARS = 2_000
TOOL_OUTPUT_KEEP_CHARS = 8_000
TOOL_OUTPUT_COLLAPSED_CHARS = 500


def store_tool_output(
    output: str | None,
) -> tuple[str | None, bool, int]:
    """Split tool output for UI memory.

    Returns (kept, evicted, full_length): kept is the full
    output when small, else a preview head.
    """

    if not output:
        return None, False, 0

    if len(output) <= TOOL_OUTPUT_KEEP_CHARS:
        return output, False, len(output)

    return (
        output[:TOOL_OUTPUT_PREVIEW_CHARS],
        True,
        len(output),
    )


def research_summary(
    output: str | None,
) -> str | None:
    """Compact card for research JSON blobs, if parseable.

    Tries to extract the first valid JSON object from the output
    to handle cases where extra text was appended.
    """

    if not output:
        return None

    result = ResearchResult.from_json(output)

    if result is None:
        # Try to extract the first valid JSON object
        import re

        match = re.search(r"\{.*\}", output, re.DOTALL)

        while match:
            try:
                result = ResearchResult.from_json(match.group())

                if result is not None:
                    break
            except (ValueError, TypeError):
                pass

            # Try to find a shorter match
            match = re.search(r"\{.*?\}", match.group()[1:], re.DOTALL)

        if result is None:
            return None

    return (
        f"research: {len(result.pages)} page(s), "
        f"{result.total_bytes} bytes, "
        f"depth {result.max_depth_reached}, "
        f"failed {len(result.failed_urls)}"
    )


@dataclass(slots=True, frozen=True)
class UserMessage:
    content: str


@dataclass(slots=True)
class AssistantMessage:
    content: str


@dataclass(slots=True)
class ThinkingView:
    iteration: int
    content: str = ""
    expanded: bool = True


@dataclass(slots=True)
class ToolView:
    call_id: str
    name: str
    arguments: dict[str, Any]
    status: ToolStatus = ToolStatus.RUNNING
    output: str | None = None
    output_evicted: bool = False
    output_full_length: int = 0
    research_summary: str | None = None
    duration_seconds: float | None = None
    error_code: str | None = None
    error_message: str | None = None
    expanded: bool = False


ConversationItem = (
    UserMessage
    | AssistantMessage
    | ThinkingView
    | ToolView
)


@dataclass(slots=True)
class AgentRunView:
    run_id: str
    parent_run_id: str | None

    model: str | None = None
    agent_id: str = "main"
    role: str = "main"

    conversation: list[ConversationItem] = field(
        default_factory=list,
    )

    thinking: bool = False
    streaming: bool = False
    iteration: int = 0
    running: bool = True

    assistant_streaming: bool = False
    assistant_streamed_current: bool = False

    def start_thinking(
        self,
        iteration: int,
    ) -> None:
        self.thinking = True
        self.iteration = iteration

        self.assistant_streaming = False
        self.assistant_streamed_current = False

        # The view is created lazily by append_thinking: models
        # without thinking output must not leave an empty
        # "Thinking…" panel behind.

    def append_thinking(
        self,
        content: str,
    ) -> None:
        if not content:
            return

        for item in reversed(self.conversation):
            if not isinstance(item, ThinkingView):
                continue

            if item.iteration != self.iteration:
                break

            item.content += content
            return

        self.conversation.append(
            ThinkingView(
                iteration=self.iteration,
                content=content,
            ),
        )

    def stop_thinking(self) -> None:
        self.thinking = False

    def finalize_thinking(self) -> None:
        self.thinking = False

        self.conversation = [
            item
            for item in self.conversation
            if not isinstance(item, ThinkingView)
        ]

    def toggle_thinking(self) -> None:
        for item in reversed(self.conversation):
            if not isinstance(item, ThinkingView):
                continue

            item.expanded = not item.expanded
            return

    def start_assistant_stream(self) -> None:
        self.streaming = True
        self.assistant_streaming = True
        self.assistant_streamed_current = True

        self.conversation.append(
            AssistantMessage(
                content="",
            ),
        )

    def append_assistant(
        self,
        content: str,
    ) -> None:
        if not content:
            return

        for item in reversed(self.conversation):
            if not isinstance(item, AssistantMessage):
                continue

            item.content += content
            return

    def finish_assistant_stream(self) -> None:
        self.streaming = False
        self.assistant_streaming = False

    def add_assistant_message(
        self,
        content: str,
    ) -> None:
        self.conversation.append(
            AssistantMessage(
                content=content,
            ),
        )

    def add_tool(
        self,
        call_id: str,
        name: str,
        arguments: dict[str, Any],
    ) -> None:
        self.conversation.append(
            ToolView(
                call_id=call_id,
                name=name,
                arguments=arguments,
            ),
        )

    def finish_tool(
        self,
        call_id: str,
        output: str | None,
        error_code: str | None,
        error_message: str | None,
        duration_seconds: float | None = None,
    ) -> None:
        for item in reversed(self.conversation):
            if not isinstance(item, ToolView):
                continue

            if item.call_id != call_id:
                continue

            item.status = (
                ToolStatus.SUCCESS
                if error_code is None
                else ToolStatus.ERROR
            )

            kept, evicted, full_length = store_tool_output(
                output,
            )

            item.output = kept
            item.output_evicted = evicted
            item.output_full_length = full_length
            item.research_summary = research_summary(output)
            item.duration_seconds = duration_seconds
            item.error_code = error_code
            item.error_message = error_message

            if error_code is not None:
                item.expanded = True

            return

    def toggle_tool(
        self,
        call_id: str,
    ) -> None:
        for item in reversed(self.conversation):
            if not isinstance(item, ToolView):
                continue

            if item.call_id != call_id:
                continue

            item.expanded = not item.expanded
            return


@dataclass(slots=True)
class UIState:
    conversation: list[ConversationItem] = field(
        default_factory=list,
    )

    # Все активные и завершённые runs этого UI-сеанса.
    #
    # Main agent тоже хранится здесь после AgentStarted,
    # но основная история продолжает жить непосредственно
    # в UIState.conversation для совместимости с текущим UI.
    runs: dict[str, AgentRunView] = field(
        default_factory=dict,
    )

    # run_id текущего main agent.
    main_run_id: str | None = None

    # Последний активный subagent. Нужен для UI-навигации
    # и дальнейшего expand/collapse.
    active_subagent_run_id: str | None = None

    thinking: bool = False
    streaming: bool = False
    iteration: int = 0
    running: bool = False
    first_message: bool = True

    estimated_tokens: int | None = None
    model_context_size: int | None = None

    assistant_streaming: bool = False
    assistant_streamed_current: bool = False

    def start_run(
        self,
        run_id: str,
        parent_run_id: str | None,
        model: str | None = None,
        agent_id: str = "main",
        role: str = "main",
    ) -> AgentRunView:
        run = self.runs.get(run_id)

        if run is not None:
            run.running = True

            if model:
                run.model = model

            if agent_id != "main":
                run.agent_id = agent_id

            if role != "main":
                run.role = role

            return run

        run = AgentRunView(
            run_id=run_id,
            parent_run_id=parent_run_id,
            model=model,
            agent_id=agent_id,
            role=role,
        )

        self.runs[run_id] = run

        if parent_run_id is None:
            self.main_run_id = run_id
        else:
            self.active_subagent_run_id = run_id

        return run

    def get_run(
        self,
        run_id: str,
    ) -> AgentRunView | None:
        return self.runs.get(run_id)

    def finish_run(
        self,
        run_id: str,
    ) -> None:
        run = self.runs.get(run_id)

        if run is None:
            return

        run.running = False
        run.thinking = False
        run.streaming = False
        run.assistant_streaming = False

        if self.active_subagent_run_id == run_id:
            self.active_subagent_run_id = None

    def add_user_message(
        self,
        content: str,
    ) -> None:
        self.conversation.append(
            UserMessage(
                content=content,
            ),
        )

    # ------------------------------------------------------------------
    # Main agent state
    # ------------------------------------------------------------------

    def start_thinking(
        self,
        iteration: int,
    ) -> None:
        self.thinking = True
        self.iteration = iteration

        self.assistant_streaming = False
        self.assistant_streamed_current = False

        # The view is created lazily by append_thinking: models
        # without thinking output must not leave an empty
        # "Thinking…" panel behind.

    def append_thinking(
        self,
        content: str,
    ) -> None:
        if not content:
            return

        for item in reversed(self.conversation):
            if not isinstance(item, ThinkingView):
                continue

            if item.iteration != self.iteration:
                break

            item.content += content
            return

        self.conversation.append(
            ThinkingView(
                iteration=self.iteration,
                content=content,
            ),
        )

    def stop_thinking(self) -> None:
        self.thinking = False

    def finalize_thinking(self) -> None:
        self.thinking = False

        self.conversation = [
            item
            for item in self.conversation
            if not isinstance(item, ThinkingView)
        ]

    def toggle_thinking(self) -> None:
        for item in reversed(self.conversation):
            if not isinstance(item, ThinkingView):
                continue

            item.expanded = not item.expanded
            return

    def start_assistant_stream(self) -> None:
        self.streaming = True
        self.assistant_streaming = True
        self.assistant_streamed_current = True

        self.conversation.append(
            AssistantMessage(
                content="",
            ),
        )

    def append_assistant(
        self,
        content: str,
    ) -> None:
        if not content:
            return

        for item in reversed(self.conversation):
            if not isinstance(item, AssistantMessage):
                continue

            item.content += content
            return

    def finish_assistant_stream(self) -> None:
        self.streaming = False
        self.assistant_streaming = False

    def add_assistant_message(
        self,
        content: str,
    ) -> None:
        self.conversation.append(
            AssistantMessage(
                content=content,
            ),
        )

    def add_tool(
        self,
        call_id: str,
        name: str,
        arguments: dict[str, Any],
    ) -> None:
        self.conversation.append(
            ToolView(
                call_id=call_id,
                name=name,
                arguments=arguments,
            ),
        )

    def finish_tool(
        self,
        call_id: str,
        output: str | None,
        error_code: str | None,
        error_message: str | None,
        duration_seconds: float | None = None,
    ) -> None:
        for item in reversed(self.conversation):
            if not isinstance(item, ToolView):
                continue

            if item.call_id != call_id:
                continue

            item.status = (
                ToolStatus.SUCCESS
                if error_code is None
                else ToolStatus.ERROR
            )

            kept, evicted, full_length = store_tool_output(
                output,
            )

            item.output = kept
            item.output_evicted = evicted
            item.output_full_length = full_length
            item.research_summary = research_summary(output)
            item.duration_seconds = duration_seconds
            item.error_code = error_code
            item.error_message = error_message

            if error_code is not None:
                item.expanded = True

            return

    def toggle_tool(
        self,
        call_id: str,
    ) -> None:
        for item in reversed(self.conversation):
            if not isinstance(item, ToolView):
                continue

            if item.call_id != call_id:
                continue

            item.expanded = not item.expanded
            return

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def clear_conversation(self) -> None:
        self.conversation.clear()
        self.runs.clear()

        self.main_run_id = None
        self.active_subagent_run_id = None

        self.thinking = False
        self.streaming = False
        self.iteration = 0
        self.running = False
        self.first_message = True

        self.assistant_streaming = False
        self.assistant_streamed_current = False
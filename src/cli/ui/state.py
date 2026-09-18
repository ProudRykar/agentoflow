from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ToolStatus(str, Enum):
    RUNNING = "running"
    SUCCESS = "success"
    ERROR = "error"


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

        self.conversation.append(
            ThinkingView(
                iteration=iteration,
            ),
        )

    def append_thinking(
        self,
        content: str,
    ) -> None:
        if not content:
            return

        for item in reversed(self.conversation):
            if not isinstance(item, ThinkingView):
                continue

            item.content += content
            return

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

            item.output = output
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

    assistant_streaming: bool = False
    assistant_streamed_current: bool = False

    def start_run(
        self,
        run_id: str,
        parent_run_id: str | None,
    ) -> AgentRunView:
        run = self.runs.get(run_id)

        if run is not None:
            run.running = True
            return run

        run = AgentRunView(
            run_id=run_id,
            parent_run_id=parent_run_id,
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

        self.conversation.append(
            ThinkingView(
                iteration=iteration,
            ),
        )

    def append_thinking(
        self,
        content: str,
    ) -> None:
        if not content:
            return

        for item in reversed(self.conversation):
            if not isinstance(item, ThinkingView):
                continue

            item.content += content
            return

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

            item.output = output
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
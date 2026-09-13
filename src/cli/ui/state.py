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
class UIState:
    conversation: list[ConversationItem] = field(
        default_factory=list,
    )

    thinking: bool = False
    streaming: bool = False
    iteration: int = 0
    running: bool = False
    first_message: bool = True

    assistant_streaming: bool = False
    assistant_streamed_current: bool = False

    def add_user_message(
        self,
        content: str,
    ) -> None:
        self.conversation.append(
            UserMessage(
                content=content,
            ),
        )

    def start_thinking(
        self,
        iteration: int,
    ) -> None:
        self.thinking = True
        self.iteration = iteration

        # Новый LLM-запрос = новый текущий ответ.
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
        """
        Полностью удалить thinking из истории.

        Оставлен для совместимости с кодом,
        который может захотеть очистить временный thinking.
        В штатном streaming flow не используется.
        """
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

    def clear_conversation(self) -> None:
        self.conversation.clear()

        self.thinking = False
        self.streaming = False
        self.iteration = 0
        self.running = False
        self.first_message = True

        self.assistant_streaming = False
        self.assistant_streamed_current = False

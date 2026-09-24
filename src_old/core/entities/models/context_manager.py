from typing import Any

from core.entities.models.context_policy import ContextPolicy
from core.entities.models.system_prompt import SYSTEM_PROMPT

class ContextManager:
    def __init__(
        self,
        policy: ContextPolicy | None = None,
        system_prompt: str | None = None,
    ) -> None:
        self._policy = policy or ContextPolicy()
        self._system_prompt = system_prompt
        self._messages: list[dict[str, Any]] = []

    def create(self, prompt: str) -> None:
        self._messages.clear()

        if self._system_prompt is not None:
            self._messages.append(
                {
                    "role": "system",
                    "content": self._system_prompt,
                }
            )

        self._messages.append(
            {
                "role": "user",
                "content": prompt,
            }
        )

        self._trim()

    def add_message(
        self,
        message: dict[str, Any],
    ) -> None:
        self._messages.append(message)
        self._trim()

    def add_assistant_message(
        self,
        message: dict[str, Any],
    ) -> None:
        self.add_message(message)

    def add_tool_result(
        self,
        message: dict[str, Any],
    ) -> None:
        self.add_message(message)

    def messages(self) -> list[dict[str, Any]]:
        return [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            *self._messages,
        ]

    def dialogue(self) -> list[dict[str, Any]]:
        """
        Raw conversation without the harness system message.

        The ContextAssembler owns system-level blocks; it takes
        only the dialogue from here.
        """

        return list(self._messages)

    def _trim(self) -> None:
        max_messages = self._policy.max_messages

        if max_messages is None:
            return

        while len(self._messages) > max_messages:
            group = self._oldest_message_group()

            if not group:
                break

            del self._messages[:len(group)]

    def _oldest_message_group(
        self,
    ) -> list[dict[str, Any]]:
        start_index = 0

        # System message всегда сохраняем.
        if (
            self._messages
            and self._messages[0].get("role") == "system"
        ):
            start_index = 1

        if start_index >= len(self._messages):
            return []

        message = self._messages[start_index]

        # Обычное сообщение — самостоятельная группа.
        if (
            message.get("role") != "assistant"
            or not message.get("tool_calls")
        ):
            return [message]

        # Assistant с tool_calls начинает атомарную группу.
        group = [message]

        tool_call_ids = {
            tool_call.get("id")
            for tool_call in message["tool_calls"]
            if isinstance(tool_call, dict)
            and isinstance(tool_call.get("id"), str)
        }

        index = start_index + 1

        while index < len(self._messages):
            next_message = self._messages[index]

            if (
                next_message.get("role") != "tool"
                or next_message.get("tool_call_id") not in tool_call_ids
            ):
                break

            group.append(next_message)
            index += 1

        return group

    def add_user_message(self, prompt: str) -> None:
        self.add_message(
            {
                "role": "user",
                "content": prompt,
            }
        )

    def clear(self) -> None:
        self._messages.clear()
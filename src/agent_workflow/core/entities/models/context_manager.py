from typing import Any

from agent_workflow.core.entities.models.context_policy import ContextPolicy
from agent_workflow.core.entities.models.system_prompt import SYSTEM_PROMPT

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

    def restore(
        self,
        dialogue: list[dict[str, Any]],
    ) -> None:
        """Replace the conversation window with persisted turns.

        Used when a session is rehydrated after a restart. The
        harness system prompt is not part of ``dialogue`` and is
        added by ``messages()``, so it is deliberately not stored
        here.
        """

        self._messages = [dict(item) for item in dialogue]

        self._trim()

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

    def has_trailing_assistant(self) -> bool:
        """Whether an assistant turn is there to be replaced.

        Deliberately non-mutating. The caller needs to refuse a
        regenerate *before* anything is dropped, and asking by
        dropping would take the conversation with it -- the second call
        would then find nothing and the user's own message would be
        gone.
        """

        if not self._messages:
            return False

        return self._messages[-1].get("role") in ("assistant", "tool")

    def drop_trailing_assistant(self) -> int:
        """Remove the assistant's last answer so it can be produced again.

        Returns how many messages went.

        Only trailing assistant and tool messages are removed, back to
        and including the assistant turn that made the tool calls. A
        tool result without the call that produced it is invalid to
        the provider, so it has to go with the call; stopping at the
        assistant turn and leaving orphaned results behind produces a
        request that is rejected outright.

        Zero is returned when there is nothing of the assistant's own
        to redo, which is the case when the last thing said was the
        user's.
        """

        cutoff = len(self._messages)

        while cutoff > 0:
            role = self._messages[cutoff - 1].get("role")

            if role in ("assistant", "tool"):
                cutoff -= 1
                continue

            break

        if cutoff == len(self._messages):
            return 0

        dropped = self._messages[cutoff:]
        self._messages = self._messages[:cutoff]

        return len(dropped)

    def _trim(self) -> None:
        max_messages = self._policy.max_messages

        if max_messages is None:
            return

        while len(self._messages) > max_messages:
            group = self._oldest_message_group()

            if not group:
                break

            del self._messages[:len(group)]

        self._drop_orphan_tools()

    def _drop_orphan_tools(self) -> None:
        """Remove tool results whose assistant block is gone.

        Group trimming removes an assistant together with its tool
        results, so this should never fire. It is kept as an explicit
        invariant: a ``tool`` message is invalid to the provider
        without the ``assistant.tool_calls`` that requested it, and a
        silent malformed request is far worse than one dropped
        message.
        """

        while True:
            start = 0

            if (
                self._messages
                and self._messages[0].get("role") == "system"
            ):
                start = 1

            if (
                start < len(self._messages)
                and self._messages[start].get("role") == "tool"
            ):
                del self._messages[start]
                continue

            return

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
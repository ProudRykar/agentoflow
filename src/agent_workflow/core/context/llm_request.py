from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agent_workflow.core.entities.models.tool_definition import ToolDefinition


@dataclass(slots=True, frozen=True)
class LLMRequestContext:
    """Exactly what the LLM sees for one request.

    Ephemeral value object: built fresh by ContextAssembler
    before every LLM call, never stored as runtime state.
    """

    messages: tuple[dict[str, Any], ...]
    tools: tuple[ToolDefinition, ...] = ()

    def to_message_list(self) -> list[dict[str, Any]]:
        """Render the messages for a provider call.

        The assembler emits one system message per fixed block
        (harness, anchor, state, catalogs, checkpoint...). Several
        chat templates, Gemma among them, render only the first
        system turn and silently drop the rest, which loses the
        skill and plugin catalogs. Merging each run of
        consecutive system messages keeps every block while
        preserving the original order and the token count.
        """

        merged: list[dict[str, Any]] = []

        for message in self.messages:
            role = message.get("role")

            if (
                role == "system"
                and merged
                and merged[-1].get("role") == "system"
            ):
                previous = merged[-1]

                previous["content"] = (
                    f"{previous.get('content', '')}\n\n"
                    f"{message.get('content', '')}"
                )

                continue

            merged.append(dict(message))

        return merged

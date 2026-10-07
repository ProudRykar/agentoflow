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

    # True when the assembler had to drop or truncate something to
    # fit the budget. Surfaced so a client can say why the history
    # looks shorter than it used to.
    trimmed: bool = False

    # Which blocks were dropped or cut, by name.
    #
    # ``trimmed`` alone says that something went and a client shows a
    # generic "trimmed to fit". That is not enough to act on: memory
    # vanishing looks identical to history being trimmed, and the
    # remedy is different. Named blocks let the report say "memory
    # dropped", which is the difference between a user widening their
    # window and a user wondering why the agent forgot what it was
    # told.
    dropped_blocks: tuple[str, ...] = ()

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

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.entities.models.tool_definition import ToolDefinition


@dataclass(slots=True, frozen=True)
class LLMRequestContext:
    """Exactly what the LLM sees for one request.

    Ephemeral value object: built fresh by ContextAssembler
    before every LLM call, never stored as runtime state.
    """

    messages: tuple[dict[str, Any], ...]
    tools: tuple[ToolDefinition, ...] = ()

    def to_message_list(self) -> list[dict[str, Any]]:
        return list(self.messages)

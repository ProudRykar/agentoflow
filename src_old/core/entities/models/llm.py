from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True, frozen=True)
class LLMToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(slots=True, frozen=True)
class LLMResponse:
    content: str | None
    tool_calls: tuple[LLMToolCall, ...]
    raw: dict[str, Any]
    thinking: str | None = None
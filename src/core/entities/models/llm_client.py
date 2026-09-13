from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from core.entities.models.llm import LLMResponse
from core.entities.models.tool_definition import ToolDefinition


class LLMClient:
    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: tuple[ToolDefinition, ...] = (),
    ) -> LLMResponse:
        raise NotImplementedError

    async def chat_stream(
        self,
        messages: list[dict[str, Any]],
        tools: tuple[ToolDefinition, ...] = (),
    ) -> AsyncIterator[dict[str, Any]]:
        raise NotImplementedError

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from core.entities.models.llm import LLMResponse, LLMToolCall
from core.entities.models.tool_definition import ToolDefinition


class LLMClient:
    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: tuple[ToolDefinition, ...] = (),
    ) -> LLMResponse:
        raise NotImplementedError

    def chat_stream(
        self,
        messages: list[dict[str, Any]],
        tools: tuple[ToolDefinition, ...] = (),
    ) -> AsyncIterator[dict[str, Any]]:
        raise NotImplementedError

    def parse_stream_tool_call(
        self,
        tool_call: dict[str, Any],
    ) -> LLMToolCall:
        raise NotImplementedError
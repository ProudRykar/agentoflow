from typing import Any

import httpx

from core.entities.models.llm import LLMResponse, LLMToolCall
from core.entities.models.llm_client import LLMClient
from core.entities.models.tool_definition import ToolDefinition


class OllamaClient(LLMClient):
    def __init__(
        self,
        model: str,
        base_url: str = "http://127.0.0.1:11434",
        timeout: float = 120.0,
    ) -> None:
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: tuple[ToolDefinition, ...] = (),
    ) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "stream": False,
        }

        if tools:
            payload["tools"] = [
                self._tool_to_ollama(tool)
                for tool in tools
            ]

        async with httpx.AsyncClient(
            timeout=self._timeout,
        ) as client:
            response = await client.post(
                f"{self._base_url}/api/chat",
                json=payload,
            )

        response.raise_for_status()

        data = response.json()

        message = data["message"]

        tool_calls = tuple(
            self._parse_tool_call(tool_call)
            for tool_call in message.get("tool_calls", [])
        )

        return LLMResponse(
            content=message.get("content"),
            tool_calls=tool_calls,
            raw=data,
        )

    @staticmethod
    def _tool_to_ollama(
        tool: ToolDefinition,
    ) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.input_schema,
            },
        }

    @staticmethod
    def _parse_tool_call(
        tool_call: dict[str, Any],
    ) -> LLMToolCall:
        function = tool_call["function"]

        return LLMToolCall(
            id=tool_call.get("id", ""),
            name=function["name"],
            arguments=function.get("arguments", {}),
        )
from typing import Any

import httpx
import pytest

from core.entities.models.llm import LLMResponse
from core.entities.models.tool_definition import ToolDefinition
from core.infrastructure.ollama_client import OllamaClient


@pytest.mark.asyncio
async def test_chat_without_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def mock_post(
        self: httpx.AsyncClient,
        url: str,
        **kwargs: Any,
    ) -> httpx.Response:
        request = httpx.Request("POST", url)

        return httpx.Response(
            200,
            request=request,
            json={
                "model": "qwen3:14b",
                "message": {
                    "role": "assistant",
                    "content": "Hello!",
                },
            },
        )

    monkeypatch.setattr(
        httpx.AsyncClient,
        "post",
        mock_post,
    )

    client = OllamaClient(
        model="qwen3:14b",
    )

    response = await client.chat(
        messages=[
            {
                "role": "user",
                "content": "Hello",
            }
        ],
    )

    assert isinstance(response, LLMResponse)
    assert response.content == "Hello!"
    assert response.tool_calls == ()


@pytest.mark.asyncio
async def test_chat_with_tool_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def mock_post(
        self: httpx.AsyncClient,
        url: str,
        **kwargs: Any,
    ) -> httpx.Response:
        request = httpx.Request("POST", url)

        return httpx.Response(
            200,
            request=request,
            json={
                "model": "qwen3:14b",
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "function": {
                                "name": "read_file",
                                "arguments": {
                                    "path": "test.txt",
                                },
                            },
                        }
                    ],
                },
            },
        )

    monkeypatch.setattr(
        httpx.AsyncClient,
        "post",
        mock_post,
    )

    client = OllamaClient(
        model="qwen3:14b",
    )

    tool = ToolDefinition(
        name="read_file",
        description="Read a UTF-8 text file",
        input_schema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                },
            },
            "required": ["path"],
        },
    )

    response = await client.chat(
        messages=[
            {
                "role": "user",
                "content": "Read test.txt",
            }
        ],
        tools=(tool,),
    )

    assert response.content == ""

    assert len(response.tool_calls) == 1

    call = response.tool_calls[0]

    assert call.id == "call_1"
    assert call.name == "read_file"
    assert call.arguments == {
        "path": "test.txt",
    }
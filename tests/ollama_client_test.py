from typing import Any

import httpx
import pytest

from core.entities.models.llm import LLMResponse
from core.entities.models.tool_definition import ToolDefinition
from core.infrastructure.ollama_client import (
    OllamaClient,
    OllamaOptions,
)


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


def test_options_payload() -> None:
    client = OllamaClient(
        model="e2b",
        options=OllamaOptions(
            num_ctx=4096,
            num_predict=256,
            think=False,
            keep_alive="5m",
        ),
    )

    payload = client._build_payload(
        messages=[{"role": "user", "content": "hi"}],
        tools=(),
        stream=False,
    )

    assert payload["options"] == {
        "num_ctx": 4096,
        "num_predict": 256,
    }
    assert payload["think"] is False
    assert payload["keep_alive"] == "5m"


def test_default_payload_has_no_options() -> None:
    client = OllamaClient(model="e2b")

    payload = client._build_payload(
        messages=[{"role": "user", "content": "hi"}],
        tools=(),
        stream=False,
    )

    assert "options" not in payload
    assert "keep_alive" not in payload
    assert payload["think"] is True


def test_options_reject_non_positive() -> None:
    import pytest

    with pytest.raises(ValueError):
        OllamaOptions(num_ctx=0)


def _bad_request_post(
    body: str,
    status: int = 400,
) -> Any:
    async def mock_post(
        self: httpx.AsyncClient,
        url: str,
        **kwargs: Any,
    ) -> httpx.Response:
        request = httpx.Request("POST", url)

        return httpx.Response(
            status,
            request=request,
            text=body,
        )

    return mock_post


@pytest.mark.asyncio
async def test_http_error_includes_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        httpx.AsyncClient,
        "post",
        _bad_request_post("thinking is not supported"),
    )

    client = OllamaClient(model="e2b")

    with pytest.raises(RuntimeError) as exc_info:
        await client.chat(
            messages=[{"role": "user", "content": "hi"}],
        )

    assert "400" in str(exc_info.value)
    assert "thinking is not supported" in str(exc_info.value)


@pytest.mark.asyncio
async def test_http_error_writes_debug_dump(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    monkeypatch.setattr(
        httpx.AsyncClient,
        "post",
        _bad_request_post("bad options"),
    )

    dumps = tmp_path / "traces"
    client = OllamaClient(
        model="e2b",
        debug_dump_dir=dumps,
    )

    with pytest.raises(RuntimeError):
        await client.chat(
            messages=[{"role": "user", "content": "hi"}],
        )

    files = list(dumps.glob("ollama-chat-400-*.json"))

    assert len(files) == 1
    assert "bad options" in files[0].read_text(
        encoding="utf-8"
    )


@pytest.mark.asyncio
async def test_no_dump_without_flag(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    import os

    monkeypatch.setattr(
        httpx.AsyncClient,
        "post",
        _bad_request_post("bad options"),
    )

    client = OllamaClient(model="e2b")

    with pytest.raises(RuntimeError):
        await client.chat(
            messages=[{"role": "user", "content": "hi"}],
        )

    assert not os.listdir(tmp_path)


@pytest.mark.asyncio
async def test_payload_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = False

    async def mock_post(
        self: httpx.AsyncClient,
        url: str,
        **kwargs: Any,
    ) -> httpx.Response:
        nonlocal called
        called = True
        raise AssertionError("must not send")

    monkeypatch.setattr(
        httpx.AsyncClient,
        "post",
        mock_post,
    )

    client = OllamaClient(model="e2b")

    with pytest.raises(ValueError):
        await client.chat(messages=[])

    with pytest.raises(ValueError):
        await client.chat(
            messages=[{"role": "nobody", "content": "hi"}],
        )

    with pytest.raises(ValueError):
        await OllamaClient(model="").chat(
            messages=[{"role": "user", "content": "hi"}],
        )

    assert called is False
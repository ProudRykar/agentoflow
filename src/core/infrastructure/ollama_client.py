from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from core.entities.models.llm import LLMResponse, LLMToolCall
from core.entities.models.llm_client import LLMClient
from core.entities.models.tool_definition import ToolDefinition

_VALID_ROLES = frozenset({
    "system",
    "user",
    "assistant",
    "tool",
})

MAX_ERROR_BODY_CHARS = 2_000


@dataclass(slots=True, frozen=True)
class OllamaOptions:
    """Per-model runtime options sent to Ollama.

    num_ctx bounds the KV cache (the main VRAM lever after
    weights). keep_alive controls how long the model stays
    resident after the request.
    """

    num_ctx: int | None = None
    num_predict: int | None = None
    num_gpu: int | None = None
    keep_alive: str | None = None
    think: bool = True

    def __post_init__(self) -> None:
        for name in (
            "num_ctx",
            "num_predict",
            "num_gpu",
        ):
            value = getattr(self, name)

            if value is not None and value <= 0:
                raise ValueError(
                    f"{name} must be greater than 0"
                )

    def to_payload(self) -> dict[str, Any]:
        options: dict[str, Any] = {}

        if self.num_ctx is not None:
            options["num_ctx"] = self.num_ctx

        if self.num_predict is not None:
            options["num_predict"] = self.num_predict

        if self.num_gpu is not None:
            options["num_gpu"] = self.num_gpu

        return options


class OllamaClient(LLMClient):
    def __init__(
        self,
        model: str,
        base_url: str = "http://127.0.0.1:11434",
        timeout: float = 120.0,
        options: OllamaOptions | None = None,
        debug_dump_dir: Path | None = None,
    ) -> None:
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._options = options or OllamaOptions()
        self._debug_dump_dir = debug_dump_dir

    @property
    def options(self) -> OllamaOptions:
        return self._options

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: tuple[ToolDefinition, ...] = (),
    ) -> LLMResponse:
        payload = self._build_payload(
            messages=messages,
            tools=tools,
            stream=False,
        )

        self._validate_payload(payload)

        async with httpx.AsyncClient(
            timeout=self._timeout,
        ) as client:
            response = await client.post(
                f"{self._base_url}/api/chat",
                json=payload,
            )

        self._raise_for_status(
            response,
            payload,
            action="chat",
        )

        data = response.json()
        message = data["message"]

        tool_calls = tuple(
            self._parse_tool_call(
                tool_call,
            )
            for tool_call in message.get(
                "tool_calls",
                [],
            )
        )

        return LLMResponse(
            content=message.get("content"),
            thinking=message.get("thinking"),
            tool_calls=tool_calls,
            raw=data,
        )

    async def chat_stream(
        self,
        messages: list[dict[str, Any]],
        tools: tuple[ToolDefinition, ...] = (),
    ) -> AsyncIterator[dict[str, Any]]:
        payload = self._build_payload(
            messages=messages,
            tools=tools,
            stream=True,
        )

        self._validate_payload(payload)

        async with httpx.AsyncClient(
            timeout=self._timeout,
        ) as client:
            async with client.stream(
                "POST",
                f"{self._base_url}/api/chat",
                json=payload,
            ) as response:
                try:
                    response.raise_for_status()
                except httpx.HTTPStatusError as exc:
                    body = await self._read_stream_body(
                        response,
                    )

                    raise self._request_error(
                        action="chat_stream",
                        status=response.status_code,
                        url=str(response.url),
                        body=body,
                        payload=payload,
                    ) from exc

                async for line in response.aiter_lines():
                    if not line:
                        continue

                    data = json.loads(line)

                    yield data

    def _build_payload(
        self,
        messages: list[dict[str, Any]],
        tools: tuple[ToolDefinition, ...],
        stream: bool,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "stream": stream,
            "think": self._options.think,
        }

        runtime_options = self._options.to_payload()

        if runtime_options:
            payload["options"] = runtime_options

        if self._options.keep_alive is not None:
            payload["keep_alive"] = self._options.keep_alive

        if tools:
            payload["tools"] = [
                self._tool_to_ollama(tool)
                for tool in tools
            ]

        return payload

    def _validate_payload(
        self,
        payload: dict[str, Any],
    ) -> None:
        if not self._model:
            raise ValueError(
                "Ollama model must not be empty",
            )

        messages = payload.get("messages")

        if not isinstance(messages, list) or not messages:
            raise ValueError(
                "Ollama payload must contain "
                "a non-empty messages list",
            )

        for index, message in enumerate(messages):
            if not isinstance(message, dict):
                raise ValueError(
                    f"Message #{index} must be an object",
                )

            role = message.get("role")

            if role not in _VALID_ROLES:
                raise ValueError(
                    f"Message #{index} has invalid role: "
                    f"{role!r}",
                )

    def _raise_for_status(
        self,
        response: httpx.Response,
        payload: dict[str, Any],
        action: str,
    ) -> None:
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise self._request_error(
                action=action,
                status=response.status_code,
                url=str(response.url),
                body=response.text,
                payload=payload,
            ) from exc

    @staticmethod
    async def _read_stream_body(
        response: httpx.Response,
    ) -> str:
        try:
            return await response.aread().decode(
                "utf-8",
                errors="replace",
            )
        except Exception:
            return "<unreadable response body>"

    def _request_error(
        self,
        action: str,
        status: int,
        url: str,
        body: str,
        payload: dict[str, Any],
    ) -> RuntimeError:
        snippet = body.strip()[:MAX_ERROR_BODY_CHARS]

        self._maybe_dump(
            action=action,
            status=status,
            url=url,
            body=body,
            payload=payload,
        )

        return RuntimeError(
            f"Ollama {action} failed: "
            f"HTTP {status} for url '{url}'. "
            f"Response body: {snippet or '<empty>'}"
        )

    def _maybe_dump(
        self,
        action: str,
        status: int,
        url: str,
        body: str,
        payload: dict[str, Any],
    ) -> None:
        if self._debug_dump_dir is None:
            return

        try:
            self._debug_dump_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            stamp = datetime.now(UTC).strftime(
                "%Y%m%dT%H%M%S%f"
            )

            dump = {
                "action": action,
                "model": self._model,
                "status": status,
                "url": url,
                "body": body,
                "payload": payload,
            }

            path = (
                self._debug_dump_dir
                / f"ollama-{action}-{status}-{stamp}.json"
            )

            path.write_text(
                json.dumps(
                    dump,
                    ensure_ascii=False,
                    indent=2,
                    default=str,
                ),
                encoding="utf-8",
            )
        except Exception:
            pass

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
            arguments=function.get(
                "arguments",
                {},
            ),
        )

    @staticmethod
    def parse_stream_tool_call(
        tool_call: dict[str, Any],
    ) -> LLMToolCall:
        return OllamaClient._parse_tool_call(
            tool_call,
        )

    async def load(self) -> None:
        async with httpx.AsyncClient(
            timeout=self._timeout,
        ) as client:
            response = await client.post(
                f"{self._base_url}/api/generate",
                json={
                    "model": self._model,
                    "prompt": "",
                    "stream": False,
                    "keep_alive": -1,
                },
            )

        response.raise_for_status()

    async def unload(self) -> None:
        async with httpx.AsyncClient(
            timeout=self._timeout,
        ) as client:
            response = await client.post(
                f"{self._base_url}/api/generate",
                json={
                    "model": self._model,
                    "prompt": "",
                    "stream": False,
                    "keep_alive": 0,
                },
            )

        response.raise_for_status()

    

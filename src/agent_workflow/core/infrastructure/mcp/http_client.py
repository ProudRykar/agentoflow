from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any

import httpx

from agent_workflow.core.infrastructure.mcp.protocol import (
    CLIENT_NAME,
    CLIENT_VERSION,
    PROTOCOL_VERSION,
    MCPError,
    MCPProtocolError,
    MCPTransportError,
    ServerInfo,
    ToolDescriptor,
    raise_for_error,
)


@dataclass(slots=True, frozen=True)
class MCPHttpConfig:
    """An MCP server reached over Streamable HTTP."""

    url: str
    headers: dict[str, str] | None = None
    startup_timeout: float = 20.0
    request_timeout: float = 60.0
    # Servers may answer with text/event-stream instead of JSON.
    accept_sse: bool = True


class MCPHttpClient:
    """Streamable HTTP transport for MCP.

    Implements the request/response half of the protocol: JSON-RPC
    is POSTed to a single endpoint and the reply is either a JSON
    body or an SSE stream. Server-initiated messages and
    notifications are read and discarded so they cannot
    desynchronise request correlation.

    The legacy HTTP+SSE split transport is not implemented.
    """

    def __init__(
        self,
        config: MCPHttpConfig,
    ) -> None:
        self._config = config
        self._client: httpx.AsyncClient | None = None
        self._next_id = 0
        self._lock = asyncio.Lock()
        self._server_info: ServerInfo | None = None
        self._session_id: str | None = None

    @property
    def connected(self) -> bool:
        return self._client is not None and self._server_info is not None

    @property
    def server_info(self) -> ServerInfo | None:
        return self._server_info

    def qualify(
        self,
        tool_name: str,
    ) -> str:
        return f"mcp_http_{_slug(tool_name)}"

    def stderr_tail(
        self,
        limit: int = 500,
    ) -> str:
        """No process, so no stderr.

        Present so the manager can report diagnostics uniformly for
        both transports; HTTP failures carry the status and body
        instead.
        """

        del limit

        return ""

    # ==================================================================
    # Lifecycle
    # ==================================================================

    async def connect(self) -> ServerInfo:
        if self.connected and self._server_info is not None:
            return self._server_info

        if not self._config.url.strip():
            raise MCPTransportError(
                "MCP HTTP server has no url",
            )

        headers = {
            "Content-Type": "application/json",
            "Accept": (
                "application/json, text/event-stream"
                if self._config.accept_sse
                else "application/json"
            ),
            **dict(self._config.headers or {}),
        }

        try:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self._config.request_timeout),
                headers=headers,
                follow_redirects=True,
            )
        except Exception as exc:  # pragma: no cover - constructor
            self._client = None

            raise MCPTransportError(
                f"Cannot create MCP HTTP client: {exc}"
            ) from exc

        try:
            await self._initialize()
        except MCPError:
            await self.close()
            raise

        return self._server_info  # type: ignore[return-value]

    async def close(self) -> None:
        client = self._client

        self._client = None
        self._server_info = None
        self._session_id = None

        if client is None:
            return

        try:
            await client.aclose()
        except Exception:
            pass

    async def __aenter__(self) -> "MCPHttpClient":
        await self.connect()

        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    # ==================================================================
    # Calls
    # ==================================================================

    async def _initialize(self) -> None:
        result = await self._request(
            "initialize",
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "clientInfo": {
                    "name": CLIENT_NAME,
                    "version": CLIENT_VERSION,
                },
            },
            timeout=self._config.startup_timeout,
        )

        # Announce readiness. Failure is not fatal: many servers
        # accept requests without the notification.
        try:
            await self._notify("notifications/initialized", {})
        except MCPError:
            pass

        info = result.get("serverInfo", {})

        self._server_info = ServerInfo(
            name=str(info.get("name", "http")),
            version=str(info.get("version", "")),
            protocol_version=str(result.get("protocolVersion", "")),
            instructions=str(result.get("instructions", "")),
        )

    async def list_tools(self) -> list[ToolDescriptor]:
        result = await self._request("tools/list", {})

        raw_tools = result.get("tools", [])

        if not isinstance(raw_tools, list):
            raise MCPProtocolError(
                "HTTP server returned a malformed tools list",
            )

        descriptors: list[ToolDescriptor] = []

        for item in raw_tools:
            if not isinstance(item, dict):
                continue

            name = item.get("name")

            if not isinstance(name, str) or not name:
                continue

            schema = item.get("inputSchema")

            descriptors.append(
                ToolDescriptor(
                    name=name,
                    description=str(item.get("description", "")),
                    input_schema=(
                        schema if isinstance(schema, dict) else {}
                    ),
                )
            )

        return descriptors

    async def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        return await self._request(
            "tools/call",
            {"name": tool_name, "arguments": arguments},
        )

    # ==================================================================
    # Transport
    # ==================================================================

    async def _request(
        self,
        method: str,
        params: dict[str, Any],
        timeout: float | None = None,
    ) -> dict[str, Any]:
        client = self._client

        if client is None:
            raise MCPTransportError(
                "MCP HTTP client is not connected",
            )

        async with self._lock:
            self._next_id += 1
            request_id = self._next_id

            message: dict[str, Any] = {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": method,
            }

            if params:
                message["params"] = params

            headers: dict[str, str] = {}

            if self._session_id:
                headers["Mcp-Session-Id"] = self._session_id

            try:
                response = await client.post(
                    self._config.url,
                    json=message,
                    headers=headers,
                    timeout=timeout or self._config.request_timeout,
                )
            except httpx.TimeoutException as exc:
                raise MCPTransportError(
                    f"MCP HTTP server timed out on {method}"
                ) from exc
            except httpx.HTTPError as exc:
                raise MCPTransportError(
                    f"MCP HTTP request failed: {exc}"
                ) from exc

            self._capture_session_id(response, headers)

            if response.status_code >= 400:
                raise MCPProtocolError(
                    f"MCP HTTP server returned "
                    f"{response.status_code} for {method}"
                )

            return self._extract_result(
                response,
                request_id,
                method,
            )

    def _capture_session_id(
        self,
        response: httpx.Response,
        sent_headers: dict[str, str],
    ) -> None:
        session = response.headers.get("mcp-session-id")

        if session:
            self._session_id = session

        # Some servers echo the session back on the request.
        del sent_headers

    def _extract_result(
        self,
        response: httpx.Response,
        request_id: int,
        method: str,
    ) -> dict[str, Any]:
        content_type = response.headers.get(
            "content-type",
            "",
        ).lower()

        if "text/event-stream" in content_type:
            payload = self._from_sse(
                response.text,
                request_id,
                method,
            )
        else:
            payload = _decode_body(response.text, method)

        raise_for_error(payload)

        result = payload.get("result")

        if result is None:
            return {}

        if not isinstance(result, dict):
            raise MCPProtocolError(
                f"MCP HTTP server returned a non-object "
                f"result for {method}",
            )

        return result

    def _from_sse(
        self,
        body: str,
        request_id: int,
        method: str,
    ) -> dict[str, Any]:
        for event in _iter_sse_data(body):
            payload = _decode_body(event, method)

            # Skip server-initiated messages.
            if "method" in payload:
                continue

            if payload.get("id") != request_id:
                continue

            return payload

        raise MCPProtocolError(
            f"MCP HTTP server sent no response for {method}",
        )

    async def _notify(
        self,
        method: str,
        params: dict[str, Any],
    ) -> None:
        client = self._client

        if client is None:
            return

        headers: dict[str, str] = {}

        if self._session_id:
            headers["Mcp-Session-Id"] = self._session_id

        try:
            await client.post(
                self._config.url,
                json={
                    "jsonrpc": "2.0",
                    "method": method,
                    "params": params,
                },
                headers=headers,
                timeout=self._config.startup_timeout,
            )
        except httpx.HTTPError as exc:
            raise MCPTransportError(
                f"MCP HTTP notification failed: {exc}"
            ) from exc


def _decode_body(
    body: str,
    method: str,
) -> dict[str, Any]:
    text = body.strip()

    if not text:
        raise MCPTransportError(
            f"MCP HTTP server returned an empty body for {method}",
        )

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise MCPTransportError(
            f"MCP HTTP server sent invalid JSON: {text[:200]!r}"
        ) from exc

    if not isinstance(payload, dict):
        raise MCPTransportError(
            "MCP HTTP response must be a JSON object",
        )

    return payload


def _iter_sse_data(body: str):
    """Yield the data payload of each SSE event."""

    data: list[str] = []

    for line in body.splitlines():
        stripped = line.strip()

        if not stripped:
            if data:
                yield "\n".join(data)
                data = []

            continue

        if stripped.startswith("data:"):
            data.append(stripped[5:].lstrip())

    if data:
        yield "\n".join(data)


def _slug(value: str) -> str:
    cleaned = "".join(
        char.lower() if char.isalnum() else "_"
        for char in str(value)
    )

    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")

    return cleaned.strip("_") or "tool"

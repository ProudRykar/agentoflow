from __future__ import annotations

import asyncio
import os
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from agent_workflow.core.infrastructure.mcp.protocol import (
    CLIENT_NAME,
    CLIENT_VERSION,
    PROTOCOL_VERSION,
    MCPError,
    MCPProtocolError,
    MCPTransportError,
    Notification,
    Request,
    ServerInfo,
    ToolDescriptor,
    parse_response,
    raise_for_error,
)


STDERR_BUFFER_LINES = 50


@dataclass(slots=True, frozen=True)
class MCPServerConfig:
    """One stdio MCP server from the config file."""

    name: str
    command: str
    args: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict)
    enabled: bool = True
    startup_timeout: float = 20.0
    request_timeout: float = 60.0
    # Tools from this server are namespaced with this prefix so two
    # servers cannot collide in a single ToolRegistry.
    prefix: str = ""


class MCPClient:
    """Minimal JSON-RPC client for a stdio MCP server.

    Only what the harness needs is implemented: the initialize
    handshake, ``tools/list`` and ``tools/call``. Notifications
    from the server are read and discarded so the pipe never fills
    and blocks the server.
    """

    def __init__(
        self,
        config: MCPServerConfig,
    ) -> None:
        self._config = config
        self._process: asyncio.subprocess.Process | None = None
        self._next_id = 0
        self._lock = asyncio.Lock()
        self._server_info: ServerInfo | None = None
        self._stderr_lines: deque[str] = deque(maxlen=STDERR_BUFFER_LINES)
        self._stderr_task: asyncio.Task[None] | None = None

    @property
    def name(self) -> str:
        return self._config.name

    @property
    def server_info(self) -> ServerInfo | None:
        return self._server_info

    @property
    def connected(self) -> bool:
        return (
            self._process is not None
            and self._process.returncode is None
        )

    def qualify(
        self,
        tool_name: str,
    ) -> str:
        prefix = self._config.prefix or self._config.name

        return f"mcp_{_slug(prefix)}_{_slug(tool_name)}"

    # ==================================================================
    # Lifecycle
    # ==================================================================

    async def connect(self) -> ServerInfo:
        if self.connected and self._server_info is not None:
            return self._server_info

        if not self._config.command.strip():
            raise MCPTransportError(
                f"MCP server '{self._config.name}' has no command",
            )

        environment = {
            **os.environ,
            **self._config.env,
        }

        try:
            self._process = await asyncio.create_subprocess_exec(
                self._config.command,
                *self._config.args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=environment,
            )
        except (OSError, ValueError) as exc:
            self._process = None

            raise MCPTransportError(
                f"Cannot start MCP server "
                f"'{self._config.name}': {exc}"
            ) from exc

        self._stderr_task = asyncio.create_task(
            self._drain_stderr()
        )

        try:
            await self._initialize()
        except MCPError:
            await self.close()
            raise

        return self._server_info  # type: ignore[return-value]

    async def close(self) -> None:
        process = self._process
        self._process = None
        self._server_info = None

        task = self._stderr_task
        self._stderr_task = None

        if task is not None and not task.done():
            task.cancel()

        if process is None:
            return

        if process.returncode is None:
            try:
                process.terminate()
            except ProcessLookupError:
                return

            try:
                await asyncio.wait_for(
                    process.wait(),
                    timeout=5.0,
                )
            except asyncio.TimeoutError:
                try:
                    process.kill()
                except ProcessLookupError:
                    pass

        for stream in (process.stdin, process.stdout, process.stderr):
            if stream is None:
                continue

            try:
                stream.close()
            except Exception:
                pass

    async def __aenter__(self) -> "MCPClient":
        await self.connect()

        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    # ==================================================================
    # Handshake and calls
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

        # The handshake is only complete once the server has been
        # told we are ready.
        await self._notify("notifications/initialized", {})

        info = result.get("serverInfo", {})

        self._server_info = ServerInfo(
            name=str(info.get("name", self._config.name)),
            version=str(info.get("version", "")),
            protocol_version=str(
                result.get("protocolVersion", "")
            ),
            instructions=str(result.get("instructions", "")),
        )

    async def list_tools(self) -> list[ToolDescriptor]:
        result = await self._request(
            "tools/list",
            {},
            timeout=self._config.request_timeout,
        )

        raw_tools = result.get("tools", [])

        if not isinstance(raw_tools, list):
            raise MCPProtocolError(
                f"Server '{self._config.name}' returned a "
                "malformed tools list",
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
                    description=str(
                        item.get("description", "")
                    ),
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
            timeout=self._config.request_timeout,
        )

    # ==================================================================
    # Transport
    # ==================================================================

    async def _request(
        self,
        method: str,
        params: dict[str, Any],
        timeout: float,
    ) -> dict[str, Any]:
        async with self._lock:
            process = self._process

            if process is None or process.stdin is None:
                raise MCPTransportError(
                    f"MCP server '{self._config.name}' is not running"
                )

            self._next_id += 1
            request_id = self._next_id

            request = Request(
                method=method,
                params=params,
                id=request_id,
            )

            try:
                process.stdin.write(request.encode())
                await process.stdin.drain()
            except (BrokenPipeError, ConnectionResetError) as exc:
                raise MCPTransportError(
                    f"MCP server '{self._config.name}' closed "
                    "the connection"
                ) from exc

            return await self._read_response(
                process,
                request_id,
                method,
                timeout,
            )

    async def _read_response(
        self,
        process: asyncio.subprocess.Process,
        request_id: int,
        method: str,
        timeout: float,
    ) -> dict[str, Any]:
        assert process.stdout is not None

        try:
            return await asyncio.wait_for(
                self._await_response(
                    process,
                    request_id,
                    method,
                ),
                timeout=timeout,
            )
        except asyncio.TimeoutError as exc:
            raise MCPTransportError(
                f"MCP server '{self._config.name}' timed out on "
                f"{method} after {timeout:g}s"
            ) from exc

    async def _await_response(
        self,
        process: asyncio.subprocess.Process,
        request_id: int,
        method: str,
    ) -> dict[str, Any]:
        assert process.stdout is not None

        while True:
            line = await process.stdout.readline()

            if not line:
                raise MCPTransportError(
                    f"MCP server '{self._config.name}' exited "
                    f"while handling {method}"
                )

            text = line.decode("utf-8", errors="replace").strip()

            if not text:
                continue

            payload = parse_response(text)

            # Server-initiated notifications and requests carry no
            # matching id; ignore them so they cannot desynchronise
            # the request/response pairing.
            if "method" in payload:
                continue

            if payload.get("id") != request_id:
                continue

            raise_for_error(payload)

            result = payload.get("result")

            if result is None:
                return {}

            if not isinstance(result, dict):
                raise MCPProtocolError(
                    f"Server '{self._config.name}' returned a "
                    "non-object result",
                )

            return result

    async def _notify(
        self,
        method: str,
        params: dict[str, Any],
    ) -> None:
        process = self._process

        if process is None or process.stdin is None:
            return

        try:
            process.stdin.write(
                Notification(
                    method=method,
                    params=params,
                ).encode()
            )
            await process.stdin.drain()
        except (BrokenPipeError, ConnectionResetError):
            # The server may exit before we finish initialising;
            # the next call reports the real problem.
            pass

    def stderr_tail(self, limit: int = 500) -> str:
        """Recent server diagnostics, for error messages.

        A server that prints a stack trace to stderr is the usual
        cause of a failed handshake, so the tail is kept in a
        bounded buffer by a background reader.
        """

        return "".join(self._stderr_lines)[-limit:].strip()

    async def _drain_stderr(self) -> None:
        process = self._process

        if process is None or process.stderr is None:
            return

        try:
            while True:
                raw = await process.stderr.readline()

                if not raw:
                    break

                line = raw.decode("utf-8", errors="replace")

                self._stderr_lines.append(line)

                while len(self._stderr_lines) > STDERR_BUFFER_LINES:
                    self._stderr_lines.pop(0)
        except Exception:
            return


def _slug(value: str) -> str:
    cleaned = "".join(
        char.lower() if char.isalnum() else "_"
        for char in str(value)
    )

    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")

    return cleaned.strip("_") or "tool"

from __future__ import annotations

import asyncio
import os
from collections import deque
from collections.abc import Sequence
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
        # Serialises writes only. Reads are demultiplexed by a single
        # background task, so several calls can be in flight at once
        # without their responses being read by the wrong waiter.
        self._write_lock = asyncio.Lock()
        self._pending: dict[int, asyncio.Future[dict[str, Any]]] = {}
        self._reader_task: asyncio.Task[None] | None = None
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

        self._reader_task = asyncio.create_task(
            self._read_loop(self._process)
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

        # Release anyone still waiting before the pipe is torn down,
        # so a shutdown shows them a closed connection immediately
        # rather than after their own timeout.
        self._fail_pending(
            MCPTransportError(
                f"MCP server '{self._config.name}' was closed"
            )
        )

        for attribute in ("_reader_task", "_stderr_task"):
            task = getattr(self, attribute)
            setattr(self, attribute, None)

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

    async def call_tools_concurrently(
        self,
        calls: Sequence[tuple[str, dict[str, Any]]],
        *,
        limit: int = 4,
    ) -> list[Any]:
        """Run several tool calls at once, keeping every outcome.

        A bulk edit is N calls that each take a round trip, and issued
        one per assistant turn they also cost N model turns and N
        approvals. Overlapping them turns the wall clock into the
        slowest call instead of the sum of all of them.

        Failures are returned, not raised: one rejected item in a batch
        of fifty is a normal outcome and must not discard the other
        forty-nine, which are already done on the far side and cannot
        be un-done by an exception here.
        """

        if limit < 1:
            raise ValueError("limit must be >= 1")

        if not calls:
            return []

        semaphore = asyncio.Semaphore(min(limit, len(calls)))

        async def one(
            tool_name: str,
            arguments: dict[str, Any],
        ):
            async with semaphore:
                return await self.call_tool(tool_name, arguments)

        return await asyncio.gather(
            *(one(name, args) for name, args in calls),
            return_exceptions=True,
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
        process = self._process

        if process is None or process.stdin is None:
            raise MCPTransportError(
                f"MCP server '{self._config.name}' is not running"
            )

        # The id is handed out and the line written under one lock so
        # that the order of messages on the pipe always matches the
        # order of their ids. Without that, two concurrent writes can
        # interleave and a response can be attributed to the wrong
        # request.
        async with self._write_lock:
            self._next_id += 1
            request_id = self._next_id

            request = Request(
                method=method,
                params=params,
                id=request_id,
            )

            waiter: asyncio.Future[dict[str, Any]] = (
                asyncio.get_running_loop().create_future()
            )

            self._pending[request_id] = waiter

            try:
                process.stdin.write(request.encode())
                await process.stdin.drain()
            except (BrokenPipeError, ConnectionResetError) as exc:
                self._pending.pop(request_id, None)
                raise MCPTransportError(
                    f"MCP server '{self._config.name}' closed "
                    "the connection"
                ) from exc

        try:
            return await asyncio.wait_for(waiter, timeout=timeout)
        except asyncio.TimeoutError as exc:
            # Dropped rather than left registered: a late reply to an
            # abandoned request must not resolve whatever future
            # happens to reuse this slot.
            self._pending.pop(request_id, None)
            raise MCPTransportError(
                f"MCP server '{self._config.name}' timed out on "
                f"{method} after {timeout:g}s"
            ) from exc

    async def _read_loop(
        self,
        process: asyncio.subprocess.Process,
    ) -> None:
        """Route every response to the request that is waiting for it.

        One reader for the whole process. Each waiter used to read the
        pipe itself and discard anything whose id did not match, which
        meant a reply to a queued request was thrown away and the
        server appeared to have answered the wrong thing. With N calls
        in flight there is no ordering to fall back on, so the routing
        has to be exact.
        """

        stdout = process.stdout

        if stdout is None:
            self._fail_pending(
                MCPTransportError(
                    f"MCP server '{self._config.name}' has no stdout"
                )
            )
            return

        try:
            while True:
                line = await stdout.readline()

                if not line:
                    self._fail_pending(
                        MCPTransportError(
                            f"MCP server '{self._config.name}' exited"
                        )
                    )
                    return

                text = line.decode("utf-8", errors="replace").strip()

                if not text:
                    continue

                self._dispatch(text)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._fail_pending(
                MCPTransportError(
                    f"MCP server '{self._config.name}' output failed: "
                    f"{exc}"
                )
            )

    def _dispatch(self, text: str) -> None:
        try:
            payload = parse_response(text)
        except Exception:
            # Unparseable line: nothing is waiting on it by name, and
            # failing every in-flight request over one bad line would
            # turn a stray server log into an outage.
            return

        # Server-initiated notifications and requests carry no id we
        # issued; they are not answers to anything.
        if "method" in payload:
            return

        waiter = self._pending.get(payload.get("id"))

        if waiter is None or waiter.done():
            # A reply to a request that already timed out.
            return

        try:
            raise_for_error(payload)
        except Exception as exc:
            waiter.set_exception(exc)
            return

        result = payload.get("result")

        if result is None:
            waiter.set_result({})
            return

        if not isinstance(result, dict):
            waiter.set_exception(
                MCPProtocolError(
                    f"Server '{self._config.name}' returned a "
                    "non-object result"
                )
            )
            return

        waiter.set_result(result)

    def _fail_pending(
        self,
        error: Exception,
    ) -> None:
        """Release everyone waiting when the pipe dies.

        Left unawaited, each waiter would sit until its own timeout,
        turning one connection failure into N slow timeouts.
        """

        pending = list(self._pending.items())
        self._pending.clear()

        for _request_id, waiter in pending:
            if not waiter.done():
                waiter.set_exception(error)

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

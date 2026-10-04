from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from agent_workflow.core.entities.models.tool import (
    Tool,
    ToolPolicy,
)
from agent_workflow.core.entities.models.tool_registry import (
    ToolRegistry,
    ToolRegistryError,
)
from agent_workflow.core.infrastructure.config import MCPConfig
from agent_workflow.core.infrastructure.mcp.client import (
    MCPClient,
    MCPServerConfig,
)
from agent_workflow.core.infrastructure.mcp.protocol import (
    MCPError,
    MCPToolError,
    ToolDescriptor,
    tool_error,
    tool_result_text,
)
from agent_workflow.core.infrastructure.mcp.schema import (
    build_input_dataclass,
    property_names,
)


class MCPServerState(StrEnum):
    DISABLED = "disabled"
    PENDING = "pending"
    CONNECTED = "connected"
    FAILED = "failed"
    CLOSED = "closed"


@dataclass(slots=True)
class MCPToolInfo:
    """A registered MCP tool, for observability."""

    server: str
    remote_name: str
    qualified_name: str
    description: str
    parameters: tuple[str, ...] = ()
    required: tuple[str, ...] = ()
    permissions: frozenset[str] = frozenset()
    requires_approval: bool = True


@dataclass(slots=True)
class MCPServerInfo:
    name: str
    command: str = ""
    transport: str = "stdio"
    url: str = ""
    args: tuple[str, ...] = ()
    enabled: bool = True
    state: MCPServerState = MCPServerState.PENDING
    error: str = ""
    server_version: str = ""
    protocol_version: str = ""
    instructions: str = ""
    tools: list[str] = field(default_factory=list)
    connected_at: float = 0.0
    duration_seconds: float = 0.0


class MCPManager:
    """Connects configured MCP servers and publishes their tools.

    Registration goes through the ordinary ``ToolRegistry``, so an
    MCP tool is indistinguishable downstream from a builtin: the
    same permissions, approval flow, guardrails and events apply.
    """

    def __init__(
        self,
        config: MCPConfig,
        registry: ToolRegistry,
        permissions: frozenset[str] | None = None,
    ) -> None:
        self._config = config
        self._registry = registry
        self._permissions = permissions or frozenset()

        self._clients: dict[str, MCPClient] = {}
        self._servers: dict[str, MCPServerInfo] = {}
        self._tools: dict[str, MCPToolInfo] = {}
        self._schemas: dict[str, dict[str, Any]] = {}
        self._timeouts: dict[str, float] = {}
        self._entries: dict[str, Any] = {}

        for entry in config.servers:
            self._timeouts[entry.name] = entry.request_timeout
            self._entries[entry.name] = entry

            self._servers[entry.name] = MCPServerInfo(
                name=entry.name,
                command=entry.command,
                transport=entry.transport,
                url=entry.url,
                args=entry.args,
                enabled=entry.enabled,
                state=(
                    MCPServerState.DISABLED
                    if not entry.enabled
                    else MCPServerState.PENDING
                ),
            )

    @property
    def enabled(self) -> bool:
        return self._config.enabled

    def granted_permissions(self) -> frozenset[str]:
        return frozenset(self._config.default_permissions)

    def servers(self) -> list[MCPServerInfo]:
        return [self._servers[name] for name in sorted(self._servers)]

    def tools(self) -> list[MCPToolInfo]:
        return [
            self._tools[key]
            for key in sorted(self._tools)
        ]

    def tool(
        self,
        qualified_name: str,
    ) -> MCPToolInfo | None:
        return self._tools.get(qualified_name)

    def owns(
        self,
        tool_name: str,
    ) -> bool:
        return tool_name in self._tools

    def describe(
        self,
        qualified_name: str,
    ) -> str:
        """System-prompt blurb listing available MCP tools."""

        if not self._tools:
            return ""

        lines = [
            "MCP TOOLS:",
            "",
            "These tools are provided by connected Model Context "
            "Protocol servers. Use them when the task matches the "
            "server's purpose.",
            "",
        ]

        for name in sorted(self._tools):
            info = self._tools[name]

            detail = info.description.strip().splitlines()

            summary = detail[0] if detail else ""

            lines.append(
                f"- {info.qualified_name} (server: {info.server}): "
                f"{summary}"
            )

        return "\n".join(lines)

    # ==================================================================
    # Lifecycle
    # ==================================================================

    def server_names(self) -> list[str]:
        return sorted(self._entries)

    def known(
        self,
        name: str,
    ) -> bool:
        return name in self._entries

    def is_connected(
        self,
        name: str,
    ) -> bool:
        client = self._clients.get(name)

        return client is not None and client.connected

    async def apply_config(
        self,
        config: MCPConfig,
    ) -> list[str]:
        """Adopt a new configuration without restarting the session.

        Returns the names that were connected. Servers that vanished
        or were disabled are disconnected and their tools removed;
        servers that appeared or changed are (re)connected. Failures
        stay recorded per server instead of propagating, so one bad
        entry cannot block the rest.
        """

        self._config = config

        wanted = {entry.name: entry for entry in config.servers}

        for name in list(self._entries):
            entry = wanted.get(name)

            if entry is not None and entry.enabled:
                continue

            await self.disconnect(name)

            self._entries.pop(name, None)
            self._timeouts.pop(name, None)
            self._servers.pop(name, None)

        connected: list[str] = []

        if not config.enabled:
            return connected

        for name, entry in wanted.items():
            if not entry.enabled:
                continue

            previous = self._entries.get(name)

            if previous is not None and _same_entry(previous, entry):
                continue

            if previous is not None:
                # A changed definition must not keep the old client.
                await self.disconnect(name)

            self._entries[name] = entry
            self._timeouts[name] = entry.request_timeout
            self._servers[name] = MCPServerInfo(
                name=entry.name,
                command=entry.command,
                transport=entry.transport,
                url=entry.url,
                args=entry.args,
                enabled=entry.enabled,
                state=MCPServerState.PENDING,
            )

            try:
                await self.connect_server(entry)
            except Exception:
                # connect_server records the failure on the server.
                continue

            connected.append(name)

        return connected

    async def connect_all(self) -> None:
        """Connect every server marked enabled in the config.

        Servers are independent: one failing does not stop the
        others or the runtime.
        """

        if not self._config.enabled:
            return

        for entry in self._config.servers:
            if not entry.enabled:
                continue

            await self.connect(entry.name)

    async def connect(
        self,
        name: str,
    ) -> MCPServerInfo:
        """Connect one server by name, replacing any prior session."""

        entry = self._entries.get(name)

        if entry is None:
            raise MCPError(f"Unknown MCP server: {name}")

        if not self._config.enabled:
            raise MCPError("MCP is disabled in the configuration")

        # Fully detach first: leaving stale tools registered would
        # make the fresh registration collide on name.
        await self.disconnect(name)

        return await self.connect_server(entry)

    async def disconnect(
        self,
        name: str,
    ) -> bool:
        """Stop a server and remove the tools it contributed.

        Removing the tools keeps the registry honest: a stale entry
        would let the model call a server that is no longer there,
        and would collide when the server is connected again.
        """

        info = self._servers.get(name)

        client = self._clients.pop(name, None)

        if client is not None:
            await client.close()

        if info is not None:
            info.state = MCPServerState.CLOSED
            info.tools = []

        for tool_name in self._owned_tools(name):
            self._unregister(tool_name)

        return client is not None

    def _owned_tools(
        self,
        server: str,
    ) -> list[str]:
        return [
            qualified
            for qualified, info in self._tools.items()
            if info.server == server
        ]

    def _unregister(self, qualified: str) -> None:
        self._tools.pop(qualified, None)
        self._schemas.pop(qualified, None)

        try:
            self._registry.unregister(qualified)
        except ToolRegistryError:
            # Already gone; nothing to do.
            return

    async def reload(
        self,
        name: str,
    ) -> MCPServerInfo:
        """Disconnect and reconnect, picking up tool changes."""

        await self.disconnect(name)

        return await self.connect(name)

    async def connect_server(self, entry: Any) -> MCPServerInfo:
        info = self._servers[entry.name]

        info.state = MCPServerState.PENDING
        info.error = ""
        info.transport = getattr(
            entry, "transport", "stdio"
        )
        info.url = getattr(entry, "url", "")
        info.server_version = ""
        info.protocol_version = ""
        info.instructions = ""
        info.tools = []
        info.connected_at = 0.0
        info.duration_seconds = 0.0

        client: Any

        if getattr(entry, "transport", "stdio") == "http":
            from agent_workflow.core.infrastructure.mcp.http_client import (
                MCPHttpClient,
                MCPHttpConfig,
            )

            client = MCPHttpClient(
                MCPHttpConfig(
                    url=entry.url,
                    headers=dict(getattr(entry, "headers", {}) or {}),
                    startup_timeout=entry.startup_timeout,
                    request_timeout=entry.request_timeout,
                )
            )
        else:
            client = MCPClient(
                MCPServerConfig(
                    name=entry.name,
                    command=entry.command,
                    args=tuple(entry.args),
                    env=dict(entry.env),
                    enabled=entry.enabled,
                    startup_timeout=entry.startup_timeout,
                    request_timeout=entry.request_timeout,
                    prefix=entry.prefix,
                )
            )

        started = time.monotonic()

        try:
            server_info = await client.connect()
            descriptors = await client.list_tools()

        except Exception as exc:
            detail = str(exc)

            tail = client.stderr_tail()

            if tail:
                detail = f"{detail} | server stderr: {tail}"

            info.state = MCPServerState.FAILED
            info.error = detail

            await client.close()

            return info

        self._clients[entry.name] = client

        info.state = MCPServerState.CONNECTED
        info.server_version = server_info.version
        info.protocol_version = server_info.protocol_version
        info.instructions = server_info.instructions
        info.connected_at = time.time()
        info.tools = []

        for descriptor in descriptors:
            self._register(info.name, client, descriptor)

        info.duration_seconds = time.monotonic() - started

        return info

    def _register(
        self,
        server: str,
        client: MCPClient,
        descriptor: ToolDescriptor,
    ) -> None:
        qualified = client.qualify(descriptor.name)

        schema = descriptor.input_schema

        input_type = build_input_dataclass(
            qualified,
            schema,
        )

        permissions = frozenset(self._config.default_permissions)

        policy = ToolPolicy(
            permissions=permissions,
            timeout=self._timeouts.get(
                server,
                DEFAULT_REQUEST_TIMEOUT,
            ),
            max_output_size=MAX_OUTPUT_CHARS,
            # Remote tools are third-party code, so they are gated
            # by the approval flow unless the operator opts out.
            requires_approval=self._config.require_approval,
        )

        remote_names = property_names(schema)
        required = _required_names(schema)

        info = MCPToolInfo(
            server=server,
            remote_name=descriptor.name,
            qualified_name=qualified,
            description=descriptor.description,
            parameters=tuple(remote_names),
            required=tuple(required),
            permissions=permissions,
            requires_approval=policy.requires_approval,
        )

        try:
            self._registry.register(
                Tool(
                    name=qualified,
                    description=_description_with_prefix(
                        descriptor,
                        server,
                    ),
                    input_type=input_type,
                    handler=_build_handler(
                        client,
                        descriptor.name,
                        remote_names,
                    ),
                    policy=policy,
                )
            )
        except ToolRegistryError:
            # A name clash must not break the other tools.
            return

        self._tools[qualified] = info
        self._schemas[qualified] = schema

        server_info = self._servers.get(server)

        if server_info is not None:
            server_info.tools.append(qualified)

    async def close(self) -> None:
        for name in list(self._clients):
            await self.disconnect(name)

        for info in self._servers.values():
            if info.state is MCPServerState.CONNECTED:
                info.state = MCPServerState.CLOSED

    async def __aenter__(self) -> "MCPManager":
        await self.connect_all()

        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()


MAX_OUTPUT_CHARS = 200_000

DEFAULT_REQUEST_TIMEOUT = 60.0


def _same_entry(
    left: Any,
    right: Any,
) -> bool:
    """Compare two server definitions field by field."""

    if left is None or right is None:
        return False

    return all(
        getattr(left, field, None) == getattr(right, field, None)
        for field in (
            "transport",
            "command",
            "url",
            "args",
            "env",
            "headers",
            "enabled",
            "startup_timeout",
            "request_timeout",
            "prefix",
        )
    )


def _required_names(
    schema: dict[str, Any] | None,
) -> list[str]:
    if not isinstance(schema, dict):
        return []

    required = schema.get("required")

    if not isinstance(required, list):
        return []

    return [str(item) for item in required]


def _description_with_prefix(
    descriptor: ToolDescriptor,
    server: str,
) -> str:
    prefix = f"[MCP:{server}] "

    return prefix + (descriptor.description.strip() or descriptor.name)


def _build_handler(
    client: MCPClient,
    remote_name: str,
    remote_names: list[str],
):
    """Adapt a decoded dataclass instance back to MCP arguments.

    The dataclass field names may have been sanitised, so the
    original schema names are restored here.
    """

    async def handler(
        arguments: Any,
        context: Any,
    ) -> str:
        del context

        payload = _to_arguments(arguments, remote_names)

        try:
            response = await client.call_tool(
                remote_name,
                payload,
            )
        except Exception as exc:
            return _error_text(str(exc))

        try:
            return tool_result_text(response)
        except MCPToolError as exc:
            # A server-reported failure is data, not a transport
            # error: the model should see it and adapt.
            return f"Error: {exc}"

    return handler


def _error_text(message: str) -> str:
    return f"Error: {message}"


def _to_arguments(
    arguments: Any,
    remote_names: list[str],
) -> dict[str, Any]:
    from dataclasses import asdict, is_dataclass

    if not is_dataclass(arguments):
        if isinstance(arguments, dict):
            return dict(arguments)

        return {}

    values = asdict(arguments)

    if not remote_names:
        return values

    # Map sanitised field names back onto the server's names.
    ordered = list(values)

    payload: dict[str, Any] = {}

    for index, remote in enumerate(remote_names):
        if index < len(ordered):
            payload[remote] = values[ordered[index]]

    return payload

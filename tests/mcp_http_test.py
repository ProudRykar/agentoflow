from __future__ import annotations

import asyncio
import socket
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Iterator
from urllib.parse import urlparse

import pytest

from agent_workflow.core.entities.models.tool import ToolContext
from agent_workflow.core.entities.models.tool_executor import (
    ToolExecutor,
)
from agent_workflow.core.entities.models.tool_registry import (
    ToolRegistry,
)
from agent_workflow.core.infrastructure.config import (
    MCPConfig,
    MCPServerConfigEntry,
)
from agent_workflow.core.infrastructure.mcp.http_client import (
    MCPHttpClient,
    MCPHttpConfig,
)
from agent_workflow.core.infrastructure.mcp.manager import MCPManager
from agent_workflow.core.infrastructure.mcp.protocol import (
    MCPProtocolError,
    MCPTransportError,
    tool_result_text,
)


SERVER_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "mcp_http_server.py"
)


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))

        return int(probe.getsockname()[1])


@pytest.fixture
def http_server() -> Iterator[str]:
    """Run the example HTTP MCP server in a background thread."""

    port = free_port()

    server = ThreadingHTTPServer(
        ("127.0.0.1", port),
        _load_handler(),
    )

    thread = threading.Thread(
        target=server.serve_forever,
        daemon=True,
    )

    thread.start()

    try:
        yield f"http://127.0.0.1:{port}/mcp"
    finally:
        server.shutdown()
        server.server_close()


def _load_handler():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "mcp_http_server_example",
        SERVER_SCRIPT,
    )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]

    return module.Handler


# ======================================================================
# SSE parsing
# ======================================================================


def test_iter_sse_data() -> None:
    from agent_workflow.core.infrastructure.mcp.http_client import (
        _iter_sse_data,
    )

    body = (
        ": ping\n\n"
        "event: message\n"
        'data: {"a": 1}\n'
        "\n"
        "event: message\n"
        'data: {"b": 2}\n\n'
    )

    assert list(_iter_sse_data(body)) == [
        '{"a": 1}',
        '{"b": 2}',
    ]


def test_iter_sse_data_multiline() -> None:
    from agent_workflow.core.infrastructure.mcp.http_client import (
        _iter_sse_data,
    )

    assert list(_iter_sse_data("data: one\ndata: two\n\n")) == [
        "one\ntwo",
    ]


# ======================================================================
# Client
# ======================================================================


@pytest.mark.asyncio
async def test_http_handshake_and_list(http_server: str) -> None:
    async with MCPHttpClient(MCPHttpConfig(url=http_server)) as client:
        info = client.connected

        assert info is True

        tools = await client.list_tools()

        assert {tool.name for tool in tools} == {"echo", "add"}


@pytest.mark.asyncio
async def test_http_call_tool(http_server: str) -> None:
    async with MCPHttpClient(MCPHttpConfig(url=http_server)) as client:
        response = await client.call_tool(
            "add",
            {"a": 20, "b": 22},
        )

        assert tool_result_text(response) == "42"


@pytest.mark.asyncio
async def test_http_sse_transport(http_server: str) -> None:
    """The server answers with SSE when the client accepts it."""

    async with MCPHttpClient(
        MCPHttpConfig(url=http_server, accept_sse=True)
    ) as client:
        assert client.server_info is not None
        assert client.server_info.name == "agentoflow-http-test"

        response = await client.call_tool("echo", {"text": "sse"})

        assert tool_result_text(response) == "sse"


@pytest.mark.asyncio
async def test_http_server_error_is_raised(
    http_server: str,
) -> None:
    async with MCPHttpClient(MCPHttpConfig(url=http_server)) as client:
        with pytest.raises(MCPProtocolError, match="Unknown tool"):
            await client.call_tool("nope", {})


@pytest.mark.asyncio
async def test_http_empty_url_fails() -> None:
    client = MCPHttpClient(MCPHttpConfig(url=""))

    with pytest.raises(MCPTransportError):
        await client.connect()


@pytest.mark.asyncio
async def test_http_unreachable_fails() -> None:
    client = MCPHttpClient(
        MCPHttpConfig(
            url=f"http://127.0.0.1:{free_port()}/mcp",
            startup_timeout=2.0,
        )
    )

    with pytest.raises(MCPTransportError):
        await client.connect()


@pytest.mark.asyncio
async def test_http_call_after_close_fails(http_server: str) -> None:
    client = MCPHttpClient(MCPHttpConfig(url=http_server))

    await client.connect()
    await client.close()

    with pytest.raises(MCPTransportError):
        await client.call_tool("echo", {})


@pytest.mark.asyncio
async def test_http_qualifies_tool_names() -> None:
    client = MCPHttpClient(MCPHttpConfig(url="http://x"))

    assert client.qualify("read file") == "mcp_http_read_file"


@pytest.mark.asyncio
async def test_http_concurrent_calls_are_correlated(
    http_server: str,
) -> None:
    async with MCPHttpClient(MCPHttpConfig(url=http_server)) as client:
        responses = await asyncio.gather(
            *(
                client.call_tool(
                    "add",
                    {"a": index, "b": 1},
                )
                for index in range(6)
            )
        )

        assert [
            tool_result_text(response)
            for response in responses
        ] == ["1", "2", "3", "4", "5", "6"]


@pytest.mark.asyncio
async def test_http_headers_are_sent(
    http_server: str,
) -> None:
    async with MCPHttpClient(
        MCPHttpConfig(
            url=http_server,
            headers={"X-Test": "1"},
        )
    ) as client:
        assert await client.list_tools()


# ======================================================================
# Manager integration
# ======================================================================


@pytest.mark.asyncio
async def test_manager_connects_http_server(
    http_server: str,
) -> None:
    registry = ToolRegistry()

    manager = MCPManager(
        MCPConfig(
            enabled=True,
            require_approval=False,
            servers=(
                MCPServerConfigEntry(
                    name="remote",
                    transport="http",
                    url=http_server,
                ),
            ),
        ),
        registry,
    )

    try:
        await manager.connect_all()

        [server] = manager.servers()

        assert server.state.value == "connected"
        assert server.transport == "http"
        assert server.server_version == "2.0.0"

        names = sorted(
            tool.name for tool in registry.all()
        )

        assert names == ["mcp_http_add", "mcp_http_echo"]
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_manager_executes_http_tool(
    http_server: str,
) -> None:
    registry = ToolRegistry()

    manager = MCPManager(
        MCPConfig(
            enabled=True,
            require_approval=False,
            servers=(
                MCPServerConfigEntry(
                    name="remote",
                    transport="http",
                    url=http_server,
                ),
            ),
        ),
        registry,
    )

    try:
        await manager.connect_all()

        result = await ToolExecutor(registry).execute(
            "mcp_http_echo",
            {"text": "over http"},
            ToolContext(
                working_directory=Path.cwd(),
                environment={},
                allowed_path=(Path.cwd(),),
                permissions=frozenset({"mcp.execute"}),
            ),
        )

        assert result.output == "over http"
        assert result.error is None
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_manager_disconnects_http_server(
    http_server: str,
) -> None:
    registry = ToolRegistry()

    manager = MCPManager(
        MCPConfig(
            enabled=True,
            servers=(
                MCPServerConfigEntry(
                    name="remote",
                    transport="http",
                    url=http_server,
                ),
            ),
        ),
        registry,
    )

    try:
        await manager.connect_all()

        assert await manager.disconnect("remote") is True
        assert registry.all() == ()
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_manager_http_failure_is_recorded() -> None:
    manager = MCPManager(
        MCPConfig(
            enabled=True,
            servers=(
                MCPServerConfigEntry(
                    name="dead",
                    transport="http",
                    url=f"http://127.0.0.1:{free_port()}/mcp",
                    startup_timeout=2.0,
                ),
            ),
        ),
        ToolRegistry(),
    )

    try:
        await manager.connect_all()

        [server] = manager.servers()

        assert server.state.value == "failed"
        assert server.error
    finally:
        await manager.close()


# ======================================================================
# Config
# ======================================================================


def test_config_parses_http_server(tmp_path: Path) -> None:
    from agent_workflow.core.infrastructure.config import ConfigLoader

    path = tmp_path / "config.toml"
    path.write_text(
        """
[[mcp.servers]]
name = "remote"
transport = "http"
url = "https://example.com/mcp"
headers = { Authorization = "Bearer x" }
""",
        encoding="utf-8",
    )

    [server] = ConfigLoader().load(path).mcp.servers

    assert server.transport == "http"
    assert server.url == "https://example.com/mcp"
    assert server.headers == {"Authorization": "Bearer x"}
    assert server.command == ""


def test_config_skips_http_without_url(tmp_path: Path) -> None:
    from agent_workflow.core.infrastructure.config import ConfigLoader

    path = tmp_path / "config.toml"
    path.write_text(
        """
[[mcp.servers]]
name = "broken"
transport = "http"
""",
        encoding="utf-8",
    )

    assert ConfigLoader().load(path).mcp.servers == ()


def test_config_stdio_is_default(tmp_path: Path) -> None:
    from agent_workflow.core.infrastructure.config import ConfigLoader

    path = tmp_path / "config.toml"
    path.write_text(
        """
[[mcp.servers]]
name = "local"
command = "npx"
""",
        encoding="utf-8",
    )

    [server] = ConfigLoader().load(path).mcp.servers

    assert server.transport == "stdio"
    assert server.url == ""


def test_url_of_stdlib_entry_is_blank() -> None:
    entry = MCPServerConfigEntry(name="s", command="x")

    assert entry.transport == "stdio"
    assert entry.url == ""
    assert urlparse("").path == ""

from __future__ import annotations

import asyncio
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from agent_workflow.core.entities.models.arguments import (
    ArgumentDecoder,
)
from agent_workflow.core.entities.models.schema_generator import (
    SchemaGenerator,
)
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
from agent_workflow.core.infrastructure.mcp.client import (
    MCPClient,
    MCPServerConfig,
)
from agent_workflow.core.infrastructure.mcp.manager import (
    MCPManager,
    MCPServerState,
)
from agent_workflow.core.infrastructure.mcp.protocol import (
    MCPTransportError,
    MCPToolError,
    parse_response,
    raise_for_error,
    tool_result_text,
)
from agent_workflow.core.infrastructure.mcp.schema import (
    build_input_dataclass,
    sanitize_field_name,
)


SERVER = str(
    (Path(__file__).resolve().parents[1]
     / "examples"
     / "mcp_stdio_server.py")
)


def entry(**overrides: Any) -> MCPServerConfigEntry:
    values: dict[str, Any] = {
        "name": "test",
        "command": sys.executable,
        "args": (SERVER,),
    }

    values.update(overrides)

    return MCPServerConfigEntry(**values)


def config(*servers: MCPServerConfigEntry, **kwargs: Any) -> MCPConfig:
    values: dict[str, Any] = {
        "enabled": True,
        "servers": servers or (entry(),),
        "require_approval": False,
    }

    values.update(kwargs)

    return MCPConfig(**values)


def context() -> ToolContext:
    return ToolContext(
        working_directory=Path.cwd(),
        environment={},
        allowed_path=(Path.cwd(),),
        permissions=frozenset({"mcp.execute"}),
    )


# ======================================================================
# Protocol framing
# ======================================================================


def test_request_encoding_includes_id() -> None:
    from agent_workflow.core.infrastructure.mcp.protocol import Request

    raw = Request(
        method="tools/call",
        params={"name": "x"},
        id=7,
    ).encode()

    assert raw.endswith(b"\n")
    assert b'"jsonrpc":"2.0"' in raw
    assert b'"id":7' in raw


def test_notification_has_no_id() -> None:
    from agent_workflow.core.infrastructure.mcp.protocol import (
        Notification,
    )

    raw = Notification(method="notifications/initialized").encode()

    assert b'"id"' not in raw


def test_parse_response_rejects_garbage() -> None:
    with pytest.raises(MCPTransportError):
        parse_response("not json")


def test_parse_response_rejects_blank() -> None:
    with pytest.raises(MCPTransportError):
        parse_response("   ")


def test_parse_response_rejects_non_object() -> None:
    with pytest.raises(MCPTransportError):
        parse_response("[1, 2]")


def test_raise_for_error_surfaces_message() -> None:
    from agent_workflow.core.infrastructure.mcp.protocol import (
        MCPProtocolError,
    )

    with pytest.raises(MCPProtocolError, match="boom"):
        raise_for_error(
            {"error": {"code": -32000, "message": "boom"}}
        )


def test_raise_for_error_passes_through_result() -> None:
    raise_for_error({"result": {"ok": True}})


def test_tool_result_text_joins_blocks() -> None:
    text = tool_result_text(
        {
            "content": [
                {"type": "text", "text": "one"},
                {"type": "text", "text": "two"},
            ]
        }
    )

    assert text == "one\ntwo"


def test_tool_result_text_describes_non_text_blocks() -> None:
    text = tool_result_text(
        {
            "content": [
                {"type": "text", "text": "see image"},
                {"type": "image", "data": "..."},
            ]
        }
    )

    assert "see image" in text
    assert "image omitted" in text


def test_tool_result_text_raises_on_server_error() -> None:
    with pytest.raises(MCPToolError, match="nope"):
        tool_result_text(
            {
                "isError": True,
                "content": [{"type": "text", "text": "nope"}],
            }
        )


def test_tool_result_text_handles_empty() -> None:
    assert tool_result_text({}) == ""


# ======================================================================
# Schema mapping
# ======================================================================


def test_sanitize_field_name() -> None:
    taken: set[str] = set()

    assert sanitize_field_name("a-b", taken) == "a_b"
    assert sanitize_field_name("1abc", taken) == "field_1abc"
    assert sanitize_field_name("a-b", taken) == "a_b_2"
    assert sanitize_field_name("class", taken) == "class"


def test_build_dataclass_from_schema() -> None:
    schema = {
        "type": "object",
        "properties": {
            "text": {"type": "string"},
            "count": {"type": "integer"},
        },
        "required": ["text"],
    }

    cls = build_input_dataclass("Echo", schema)

    generated = SchemaGenerator().generate(cls)

    assert generated["properties"]["text"] == {"type": "string"}
    assert generated["required"] == ["text"]

    decoded = ArgumentDecoder().decode(
        {"text": "hi", "count": 2},
        cls,
    )

    assert decoded.text == "hi"
    assert decoded.count == 2


def test_build_dataclass_without_properties() -> None:
    cls = build_input_dataclass("NoArgs", {})

    assert SchemaGenerator().generate(cls)["properties"] == {}


def test_build_dataclass_handles_enum_hint() -> None:
    cls = build_input_dataclass(
        "WithEnum",
        {
            "properties": {
                "mode": {"enum": ["a", "b"]},
            }
        },
    )

    decoded = ArgumentDecoder().decode({"mode": "a"}, cls)

    assert decoded.mode == "a"


def test_build_dataclass_rejects_unknown_argument() -> None:
    from agent_workflow.core.entities.models.arguments import (
        ArgumentDecoderError,
    )

    cls = build_input_dataclass(
        "Echo",
        {"properties": {"text": {"type": "string"}}},
    )

    with pytest.raises(ArgumentDecoderError):
        ArgumentDecoder().decode({"nope": 1}, cls)


# ======================================================================
# Client lifecycle
# ======================================================================


@pytest.mark.asyncio
async def test_client_handshake_and_list() -> None:
    client = MCPClient(
        MCPServerConfig(
            name="test",
            command=sys.executable,
            args=(SERVER,),
        )
    )

    try:
        info = await client.connect()

        assert info.name == "agentoflow-test-server"
        assert info.version == "1.2.3"

        tools = await client.list_tools()

        assert {tool.name for tool in tools} == {
            "echo",
            "add",
            "explode",
        }
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_client_call_tool() -> None:
    client = MCPClient(
        MCPServerConfig(
            name="test",
            command=sys.executable,
            args=(SERVER,),
        )
    )

    try:
        await client.connect()

        response = await client.call_tool(
            "add",
            {"a": 2, "b": 5},
        )

        assert tool_result_text(response) == "7"
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_client_missing_command_fails() -> None:
    client = MCPClient(
        MCPServerConfig(name="test", command="")
    )

    with pytest.raises(MCPTransportError):
        await client.connect()


@pytest.mark.asyncio
async def test_client_bad_command_fails() -> None:
    client = MCPClient(
        MCPServerConfig(
            name="test",
            command="definitely-not-a-real-binary-xyz",
        )
    )

    with pytest.raises(MCPTransportError):
        await client.connect()


@pytest.mark.asyncio
async def test_client_qualifies_tool_names() -> None:
    client = MCPClient(
        MCPServerConfig(name="my files", command="x")
    )

    assert client.qualify("read file") == "mcp_my_files_read_file"


@pytest.mark.asyncio
async def test_client_call_after_close_fails() -> None:
    client = MCPClient(
        MCPServerConfig(
            name="test",
            command=sys.executable,
            args=(SERVER,),
        )
    )

    await client.connect()
    await client.close()

    with pytest.raises(MCPTransportError):
        await client.call_tool("echo", {})


@pytest.mark.asyncio
async def test_client_context_manager() -> None:
    async with MCPClient(
        MCPServerConfig(
            name="test",
            command=sys.executable,
            args=(SERVER,),
        )
    ) as client:
        assert client.connected is True

    assert client.connected is False


# ======================================================================
# Manager
# ======================================================================


@pytest.mark.asyncio
async def test_manager_registers_qualified_tools() -> None:
    registry = ToolRegistry()

    manager = MCPManager(config(), registry)

    try:
        await manager.connect_all()

        names = sorted(
            tool.name for tool in registry.all()
        )

        assert names == [
            "mcp_test_add",
            "mcp_test_echo",
            "mcp_test_explode",
        ]
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_manager_reports_server_state() -> None:
    manager = MCPManager(config(), ToolRegistry())

    try:
        await manager.connect_all()

        [server] = manager.servers()

        assert server.state is MCPServerState.CONNECTED
        assert server.server_version == "1.2.3"
        assert server.protocol_version == "2025-06-18"
        assert server.error == ""
        assert len(server.tools) == 3
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_manager_records_failure_without_raising() -> None:
    """One broken server must not stop the good one."""

    registry = ToolRegistry()

    manager = MCPManager(
        config(
            entry(name="good"),
            entry(
                name="bad",
                command="definitely-not-a-real-binary-xyz",
            ),
        ),
        registry,
    )

    try:
        await manager.connect_all()

        states = {
            server.name: server.state
            for server in manager.servers()
        }

        assert states["good"] is MCPServerState.CONNECTED
        assert states["bad"] is MCPServerState.FAILED

        bad = next(
            server
            for server in manager.servers()
            if server.name == "bad"
        )

        assert bad.error

        assert registry.has("mcp_good_echo")
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_manager_skips_disabled_servers() -> None:
    manager = MCPManager(
        config(entry(enabled=False)),
        ToolRegistry(),
    )

    try:
        await manager.connect_all()

        [server] = manager.servers()

        assert server.state is MCPServerState.DISABLED
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_manager_disabled_config_does_nothing() -> None:
    registry = ToolRegistry()

    manager = MCPManager(
        MCPConfig(enabled=False, servers=(entry(),)),
        registry,
    )

    await manager.connect_all()

    assert registry.all() == ()


@pytest.mark.asyncio
async def test_manager_tools_carry_metadata() -> None:
    manager = MCPManager(config(), ToolRegistry())

    try:
        await manager.connect_all()

        info = manager.tool("mcp_test_add")

        assert info is not None
        assert info.server == "test"
        assert info.remote_name == "add"
        assert info.parameters == ("a", "b")
        assert info.required == ("a", "b")
        assert info.permissions == frozenset({"mcp.execute"})
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_manager_requires_approval_by_default() -> None:
    registry = ToolRegistry()

    manager = MCPManager(
        config(require_approval=True),
        registry,
    )

    try:
        await manager.connect_all()

        tool = registry.get("mcp_test_echo")

        assert tool.policy.requires_approval is True
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_manager_executed_tool_returns_text() -> None:
    registry = ToolRegistry()

    manager = MCPManager(config(), registry)

    try:
        await manager.connect_all()

        executor = ToolExecutor(registry)

        result = await executor.execute(
            "mcp_test_echo",
            {"text": "round trip"},
            context(),
        )

        assert result.output == "round trip"
        assert result.error is None
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_manager_server_error_is_data_not_exception() -> None:
    registry = ToolRegistry()

    manager = MCPManager(config(), registry)

    try:
        await manager.connect_all()

        result = await ToolExecutor(registry).execute(
            "mcp_test_explode",
            {},
            context(),
        )

        assert result.output is not None
        assert "deliberate failure" in result.output
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_manager_tool_denied_without_permission() -> None:
    registry = ToolRegistry()

    manager = MCPManager(config(), registry)

    try:
        await manager.connect_all()

        result = await ToolExecutor(registry).execute(
            "mcp_test_echo",
            {"text": "x"},
            ToolContext(
                working_directory=Path.cwd(),
                environment={},
                allowed_path=(Path.cwd(),),
                permissions=frozenset(),
            ),
        )

        assert result.error is not None
        assert result.error.code == "permission_denied"
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_manager_two_servers_do_not_collide() -> None:
    registry = ToolRegistry()

    manager = MCPManager(
        config(entry(name="alpha"), entry(name="beta")),
        registry,
    )

    try:
        await manager.connect_all()

        names = sorted(
            tool.name for tool in registry.all()
        )

        assert "mcp_alpha_echo" in names
        assert "mcp_beta_echo" in names
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_manager_owns_reports_membership() -> None:
    manager = MCPManager(config(), ToolRegistry())

    try:
        await manager.connect_all()

        assert manager.owns("mcp_test_echo") is True
        assert manager.owns("read_file") is False
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_manager_describe_lists_tools() -> None:
    manager = MCPManager(config(), ToolRegistry())

    try:
        await manager.connect_all()

        described = manager.describe("x")

        assert "MCP TOOLS" in described
        assert "mcp_test_echo" in described
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_manager_describe_empty_when_no_tools() -> None:
    manager = MCPManager(
        MCPConfig(enabled=False, servers=()),
        ToolRegistry(),
    )

    assert manager.describe("x") == ""


@pytest.mark.asyncio
async def test_manager_close_marks_servers_closed() -> None:
    manager = MCPManager(config(), ToolRegistry())

    await manager.connect_all()
    await manager.close()

    assert all(
        server.state is MCPServerState.CLOSED
        for server in manager.servers()
    )


@pytest.mark.asyncio
async def test_manager_concurrent_calls_are_serialised() -> None:
    """Requests must not interleave on the single stdio pipe."""

    registry = ToolRegistry()

    manager = MCPManager(config(), registry)

    try:
        await manager.connect_all()

        executor = ToolExecutor(registry)

        results = await asyncio.gather(
            *(
                executor.execute(
                    "mcp_test_add",
                    {"a": index, "b": 1},
                    context(),
                )
                for index in range(6)
            )
        )

        assert [
            result.output for result in results
        ] == ["1", "2", "3", "4", "5", "6"]
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_manager_permission_grant() -> None:
    manager = MCPManager(
        config(
            default_permissions=("mcp.execute", "mcp.write"),
        ),
        ToolRegistry(),
    )

    assert manager.granted_permissions() == frozenset({
        "mcp.execute",
        "mcp.write",
    })


# ======================================================================
# Config parsing
# ======================================================================


def write_config(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "config.toml"
    path.write_text(body, encoding="utf-8")

    return path


def load_mcp(tmp_path: Path, body: str):
    from agent_workflow.core.infrastructure.config import ConfigLoader

    return ConfigLoader().load(
        write_config(tmp_path, body)
    ).mcp


def test_config_defaults_when_absent(tmp_path: Path) -> None:
    mcp = load_mcp(tmp_path, "[agent]\nmax_iterations = 3\n")

    assert mcp.enabled is False
    assert mcp.servers == ()
    assert mcp.require_approval is True
    assert mcp.default_permissions == ("mcp.execute",)


def test_config_parses_array_form(tmp_path: Path) -> None:
    mcp = load_mcp(
        tmp_path,
        """
[mcp]
enabled = true

[[mcp.servers]]
name = "files"
command = "npx"
args = ["-y", "server-filesystem", "/tmp"]
env = { TOKEN = "x" }
""",
    )

    assert mcp.enabled is True

    [server] = mcp.servers

    assert server.name == "files"
    assert server.command == "npx"
    assert server.args == ("-y", "server-filesystem", "/tmp")
    assert server.env == {"TOKEN": "x"}
    assert server.enabled is True


def test_config_parses_keyed_form(tmp_path: Path) -> None:
    mcp = load_mcp(
        tmp_path,
        """
[mcp.servers.files]
command = "npx"
""",
    )

    [server] = mcp.servers

    assert server.name == "files"
    assert server.command == "npx"


def test_config_skips_entries_without_command(
    tmp_path: Path,
) -> None:
    mcp = load_mcp(
        tmp_path,
        """
[[mcp.servers]]
name = "broken"

[[mcp.servers]]
name = "fine"
command = "npx"
""",
    )

    assert [s.name for s in mcp.servers] == ["fine"]


def test_config_skips_entries_without_name(
    tmp_path: Path,
) -> None:
    mcp = load_mcp(
        tmp_path,
        """
[[mcp.servers]]
command = "npx"
""",
    )

    assert mcp.servers == ()


def test_config_reads_timeouts_and_prefix(
    tmp_path: Path,
) -> None:
    mcp = load_mcp(
        tmp_path,
        """
[[mcp.servers]]
name = "s"
command = "npx"
startup_timeout = 5.5
request_timeout = 90.0
prefix = "custom"
enabled = false
""",
    )

    [server] = mcp.servers

    assert server.startup_timeout == 5.5
    assert server.request_timeout == 90.0
    assert server.prefix == "custom"
    assert server.enabled is False


def test_config_reads_permissions_and_approval(
    tmp_path: Path,
) -> None:
    mcp = load_mcp(
        tmp_path,
        """
[mcp]
default_permissions = ["mcp.execute", "mcp.net"]
require_approval = false
""",
    )

    assert mcp.default_permissions == ("mcp.execute", "mcp.net")
    assert mcp.require_approval is False


def test_config_enables_when_servers_present(
    tmp_path: Path,
) -> None:
    mcp = load_mcp(
        tmp_path,
        """
[[mcp.servers]]
name = "s"
command = "npx"
""",
    )

    assert mcp.enabled is True


# ======================================================================
# Runtime server control
# ======================================================================


@pytest.mark.asyncio
async def test_connect_by_name() -> None:
    manager = MCPManager(config(), ToolRegistry())

    try:
        info = await manager.connect("test")

        assert info.state is MCPServerState.CONNECTED
        assert manager.is_connected("test") is True
        assert len(manager.tools()) == 3
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_connect_unknown_server_fails() -> None:
    from agent_workflow.core.infrastructure.mcp.protocol import MCPError

    manager = MCPManager(config(), ToolRegistry())

    with pytest.raises(MCPError, match="Unknown MCP server"):
        await manager.connect("nope")


@pytest.mark.asyncio
async def test_disconnect_removes_tools_from_registry() -> None:
    registry = ToolRegistry()

    manager = MCPManager(config(), registry)

    await manager.connect_all()

    assert registry.has("mcp_test_echo") is True

    assert await manager.disconnect("test") is True

    assert registry.has("mcp_test_echo") is False
    assert manager.tools() == []
    assert manager.is_connected("test") is False


@pytest.mark.asyncio
async def test_disconnect_leaves_other_servers_alone() -> None:
    registry = ToolRegistry()

    manager = MCPManager(
        config(entry(name="alpha"), entry(name="beta")),
        registry,
    )

    try:
        await manager.connect_all()

        await manager.disconnect("alpha")

        assert registry.has("mcp_alpha_echo") is False
        assert registry.has("mcp_beta_echo") is True
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_disconnect_unknown_returns_false() -> None:
    manager = MCPManager(config(), ToolRegistry())

    assert await manager.disconnect("nope") is False


@pytest.mark.asyncio
async def test_reconnect_after_disconnect() -> None:
    registry = ToolRegistry()

    manager = MCPManager(config(), registry)

    try:
        await manager.connect_all()
        await manager.disconnect("test")

        await manager.connect("test")

        assert registry.has("mcp_test_echo") is True
        assert len(manager.servers()) == 1

        server = manager.servers()[0]
        assert server.state is MCPServerState.CONNECTED
        assert len(server.tools) == 3
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_reload_clears_stale_state() -> None:
    manager = MCPManager(config(), ToolRegistry())

    try:
        await manager.connect_all()
        await manager.reload("test")

        server = manager.servers()[0]

        assert server.state is MCPServerState.CONNECTED
        assert server.error == ""
        assert len(server.tools) == 3
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_connect_twice_does_not_duplicate_tools() -> None:
    registry = ToolRegistry()

    manager = MCPManager(config(), registry)

    try:
        await manager.connect("test")
        await manager.connect("test")

        assert len(manager.tools()) == 3

        server = manager.servers()[0]
        assert len(server.tools) == 3
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_reconnect_records_failure_state() -> None:
    registry = ToolRegistry()

    manager = MCPManager(
        config(entry(command="definitely-not-a-real-binary-xyz")),
        registry,
    )

    try:
        await manager.connect_all()

        [server] = manager.servers()

        assert server.state is MCPServerState.FAILED
        assert server.error
        assert registry.all() == ()
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_server_names_and_known() -> None:
    manager = MCPManager(
        config(entry(name="a"), entry(name="b")),
        ToolRegistry(),
    )

    assert manager.server_names() == ["a", "b"]
    assert manager.known("a") is True
    assert manager.known("zzz") is False


@pytest.mark.asyncio
async def test_connect_refused_when_disabled() -> None:
    from agent_workflow.core.infrastructure.mcp.protocol import MCPError

    manager = MCPManager(
        MCPConfig(enabled=False, servers=(entry(),)),
        ToolRegistry(),
    )

    with pytest.raises(MCPError, match="disabled"):
        await manager.connect("test")


# ======================================================================
# Container runtime environment
# ======================================================================


def _warnings_for(tmp_path: Path, body: str) -> tuple[str, ...]:
    from agent_workflow.core.infrastructure.config import (
        ConfigLoader,
        server_configuration_warnings,
    )

    config = ConfigLoader().load(write_config(tmp_path, body))
    entry = config.mcp.servers[0]

    return server_configuration_warnings(
        entry.command,
        entry.args,
        entry.env,
    )


def test_container_runtime_without_env_flag_warns(
    tmp_path: Path,
) -> None:
    # The variable reaches podman but never the container, so the
    # server silently falls back to its own default. This exact
    # mismatch is a silent failure, which is why it is checked here.
    warnings = _warnings_for(
        tmp_path,
        """
        [mcp]
        enabled = true

        [mcp.servers.Stash]
        command = "podman"
        args = ["run", "-i", "--rm", "stash-mcp:local"]

        [mcp.servers.Stash.env]
        STASH_ENDPOINT = "http://host.containers.internal:9999"
        """,
    )

    assert len(warnings) == 1
    assert "container runtime" in warnings[0]
    assert "STASH_ENDPOINT" in warnings[0]
    assert "-e" in warnings[0]


def test_forwarded_env_produces_no_warning(tmp_path: Path) -> None:
    warnings = _warnings_for(
        tmp_path,
        """
        [mcp]
        enabled = true

        [mcp.servers.Stash]
        command = "podman"
        args = ["run", "-i", "--rm", "-e", "STASH_ENDPOINT", "img"]

        [mcp.servers.Stash.env]
        STASH_ENDPOINT = "http://host.containers.internal:9999"
        """,
    )

    assert warnings == ()


def test_inline_env_value_produces_no_warning(tmp_path: Path) -> None:
    warnings = _warnings_for(
        tmp_path,
        """
        [mcp]
        enabled = true

        [mcp.servers.Stash]
        command = "docker"
        args = ["run", "-i", "--env", "STASH_ENDPOINT=http://x", "img"]

        [mcp.servers.Stash.env]
        STASH_ENDPOINT = "http://host.containers.internal:9999"
        """,
    )

    assert warnings == ()


def test_no_warning_without_env_vars(tmp_path: Path) -> None:
    warnings = _warnings_for(
        tmp_path,
        """
        [mcp]
        enabled = true

        [mcp.servers.Stash]
        command = "podman"
        args = ["run", "-i", "--rm", "img"]
        """,
    )

    assert warnings == ()


def test_no_warning_for_a_plain_command(tmp_path: Path) -> None:
    warnings = _warnings_for(
        tmp_path,
        """
        [mcp]
        enabled = true

        [mcp.servers.Stash]
        command = "uvx"
        args = ["run", "stash-mcp"]

        [mcp.servers.Stash.env]
        STASH_ENDPOINT = "http://localhost:9999"
        """,
    )

    assert warnings == ()


@pytest.mark.asyncio
async def test_warning_reaches_the_server_info() -> None:
    manager = MCPManager(
        MCPConfig(
            enabled=True,
            servers=(
                replace(
                    entry(),
                    command="podman",
                    args=("run", "-i", "img"),
                    env={"STASH_ENDPOINT": "http://x"},
                ),
            ),
        ),
        ToolRegistry(),
    )

    info = manager.servers()[0]

    assert len(info.warnings) == 1
    assert "STASH_ENDPOINT" in info.warnings[0]

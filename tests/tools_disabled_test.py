"""Hiding tools from the model.

A narrow model picks badly when every option is on display, and two
tools that do the same thing is worse than one: `get_tags` and
`stash_graphql` both list tags, so the choice came down to whichever
the model felt like using.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import pytest

import agent_workflow.core.infrastructure.paths as paths_module
from agent_workflow.core.entities.models.tool_registry import ToolRegistry
from agent_workflow.core.entities.models.tool import (
    Tool,
    ToolPolicy,
)
from agent_workflow.core.infrastructure.config import (
    ConfigError,
    ConfigLoader,
    ToolsConfig,
)


def _tool(name: str) -> Tool:
    from dataclasses import dataclass

    @dataclass(slots=True, frozen=True)
    class _Input:
        value: str = ""

    async def handler(arguments, context):
        del arguments, context

        return "ok"

    return Tool(
        name=name,
        description="d",
        input_type=_Input,
        handler=handler,
        policy=ToolPolicy(
            permissions=frozenset(),
            timeout=1.0,
            max_output_size=10,
        ),
    )


# ======================================================================
# Config
# ======================================================================


def test_disabled_defaults_to_nothing() -> None:
    assert ToolsConfig().disabled == ()


def test_disabled_parses(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"

    path.write_text(
        '[tools]\ndisabled = ["mcp_stash_get_tags", "x"]\n',
        encoding="utf-8",
    )

    assert ConfigLoader().load(path).tools.disabled == (
        "mcp_stash_get_tags",
        "x",
    )


def test_disabled_must_be_an_array(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"

    path.write_text(
        '[tools]\ndisabled = "mcp_stash_get_tags"\n',
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="array"):
        ConfigLoader().load(path)


def test_a_blank_name_is_refused() -> None:
    # Otherwise the entry can never match and looks like it worked.
    with pytest.raises(ValueError, match="blank"):
        ToolsConfig(disabled=("  ",))


# ======================================================================
# Application
# ======================================================================


def _toggle(*names: str, disabled: tuple[str, ...] = ()):
    from agent_workflow.core.entities.models.tool_toggle import (
        ToolToggle,
    )

    registry = ToolRegistry()

    for name in names:
        registry.register(_tool(name))

    return ToolToggle(registry, ToolsConfig(disabled=disabled))


def test_apply_removes_the_named_tools() -> None:
    toggle = _toggle("a", "b", "c", disabled=("b",))

    toggle.apply()

    assert toggle._registry.has("a") is True
    assert toggle._registry.has("b") is False
    assert toggle._registry.has("c") is True


def test_disabling_then_enabling_restores_the_tool() -> None:
    """The point of retaining hidden tools.

    A one-way filter can hide a tool but never bring it back, so a
    mistake in the config needed a restart to undo.
    """

    toggle = _toggle("a")

    assert toggle.disable("a") is True
    assert toggle._registry.has("a") is False

    assert toggle.enable("a") is True
    assert toggle._registry.has("a") is True
    assert toggle.disabled_names() == ()


def test_a_tool_disabled_by_config_stays_hidden_when_it_arrives() -> None:
    """MCP tools appear only after their server connects.

    A name in [tools] disabled refers to a tool that does not exist
    yet, so the intent has to be remembered rather than applied once.
    """

    toggle = _toggle("a", disabled=("mcp_late",))

    assert toggle.is_disabled("mcp_late") is True

    # The server connects and registers its tool.
    toggle._registry.register(_tool("mcp_late"))

    toggle.apply()

    assert toggle._registry.has("mcp_late") is False

    # And a reconnect must not bring it back.
    toggle._registry.register(_tool("mcp_late"))

    toggle.apply()

    assert toggle._registry.has("mcp_late") is False


def test_unknown_names_are_reported(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A typo would otherwise look like it worked."""

    toggle = _toggle("a")

    with caplog.at_level(logging.WARNING):
        assert toggle.disable("typo") is False

    assert "typo" in caplog.text

    assert toggle.enable("typo") is False


def test_disabling_an_unknown_tool_leaves_the_rest_alone(
    caplog: pytest.LogCaptureFixture,
) -> None:
    toggle = _toggle("a", "b")

    with caplog.at_level(logging.WARNING):
        toggle.disable("nope")

    assert toggle._registry.has("a") is True
    assert toggle._registry.has("b") is True


def test_known_tools_include_hidden_ones() -> None:
    toggle = _toggle("a", "b", disabled=("b",))

    toggle.apply()

    assert toggle.enabled_names() == ("a",)
    assert toggle.known_tools_by_name().keys() == {"a", "b"}


def test_config_reflects_the_current_state() -> None:
    toggle = _toggle("a", "b")

    toggle.disable("b")
    toggle.disable("a")

    assert toggle.config().disabled == ("a", "b")


def test_blocks_reports_membership() -> None:
    config = ToolsConfig(disabled=("a",))

    assert config.blocks("a") is True
    assert config.blocks("b") is False


# ======================================================================
# End to end through the runtime
# ======================================================================


async def test_disabled_tools_are_absent_from_a_real_session() -> None:
    """The filter has to run after MCP servers connect.

    MCP tools only exist once their servers have connected, so a
    filter placed next to the builtin registration silently did
    nothing for them -- which is most of the tools anyone would want
    to hide.
    """

    from agent_workflow.core.application.runtime import create_runtime

    home = Path(tempfile.mkdtemp())

    config = home / "config.toml"

    config.write_text(
        paths_module.DEFAULT_CONFIG
        + '\n[tools]\ndisabled = ["read_file", "list_directory"]\n',
        encoding="utf-8",
    )

    original = paths_module.AgentWorkflowPaths.__init__

    def patched(self, home_override=None):
        original(self, home_override or home)

    paths_module.AgentWorkflowPaths.__init__ = patched

    try:
        from pathlib import Path as P

        runtime = await create_runtime(
            lambda **kwargs: True,
            working_directory=P(tempfile.mkdtemp()),
        )

        assert runtime.agent.registry.has("read_file") is False
        assert runtime.agent.registry.has("list_directory") is False
        assert runtime.agent.registry.has("execute_shell") is True
    finally:
        paths_module.AgentWorkflowPaths.__init__ = original


# ======================================================================
# REST
# ======================================================================


def _factory_with_tools(base_factory, tmp_path):
    """Wrap the transport stub so it carries two toggleable tools.

    Built here rather than in the shared conftest: registering stub
    tools globally would change what every other transport test
    sees.
    """

    from dataclasses import dataclass

    from agent_workflow.core.entities.models.tool import (
        Tool,
        ToolPolicy,
    )
    from agent_workflow.core.entities.models.tool_toggle import (
        ToolToggle,
    )

    @dataclass(slots=True, frozen=True)
    class _Input:
        """A real input type; None breaks the schema generator."""

        value: str = ""

    async def factory(*args, **kwargs):
        runtime = await base_factory(*args, **kwargs)

        registry = runtime.agent.registry

        async def noop(arguments, context):
            del arguments, context

            return "ok"

        for name in ("alpha_tool", "beta_tool"):
            registry.register(
                Tool(
                    name=name,
                    description=f"stub {name}",
                    input_type=_Input,
                    handler=noop,
                    policy=ToolPolicy(
                        permissions=frozenset(),
                        timeout=1.0,
                        max_output_size=10,
                    ),
                )
            )

        # The stub toggle was built when the registry was empty, so
        # it has to learn about the new tools.
        runtime.tool_toggle = ToolToggle(registry)

        return runtime

    del tmp_path

    return factory


@pytest.fixture
def tool_client(client, stub_factory, tmp_path, monkeypatch):
    """A client whose sessions carry two toggleable tools."""

    from agent_workflow.core.application import session as session_module

    monkeypatch.setattr(
        session_module,
        "create_runtime",
        _factory_with_tools(stub_factory, tmp_path),
    )

    return client


def _session_id(client) -> str:
    response = client.post("/api/sessions")

    assert response.status_code == 200

    return response.json()["session_id"]


def test_disable_hides_the_tool_for_the_run(tool_client) -> None:
    session_id = _session_id(tool_client)

    response = tool_client.post(
        f"/api/tools/{session_id}/tools/alpha_tool",
        json={"enabled": False},
    )

    assert response.status_code == 200

    body = response.json()

    assert body["tool"] == "alpha_tool"
    assert body["enabled"] is False
    assert "alpha_tool" in body["disabled"]

    listed = tool_client.get(
        f"/api/tools/{session_id}"
    ).json()

    by_name = {item["name"]: item for item in listed["tools"]}

    # A disabled tool is still listed, so it can be switched back on.
    assert by_name["alpha_tool"]["enabled"] is False
    assert by_name["beta_tool"]["enabled"] is True


def test_enable_puts_the_tool_back(tool_client) -> None:
    session_id = _session_id(tool_client)

    tool_client.post(
        f"/api/tools/{session_id}/tools/alpha_tool",
        json={"enabled": False},
    )

    response = tool_client.post(
        f"/api/tools/{session_id}/tools/alpha_tool",
        json={"enabled": True},
    )

    assert response.status_code == 200
    assert response.json()["disabled"] == []

    listed = tool_client.get(
        f"/api/tools/{session_id}"
    ).json()

    by_name = {item["name"]: item for item in listed["tools"]}

    assert by_name["alpha_tool"]["enabled"] is True


def test_unknown_tool_is_a_404(tool_client) -> None:
    """A typo must not look like it worked."""

    session_id = _session_id(tool_client)

    response = tool_client.post(
        f"/api/tools/{session_id}/tools/no_such_tool",
        json={"enabled": False},
    )

    assert response.status_code == 404
    assert "no_such_tool" in response.json()["detail"]


def test_the_registry_actually_loses_the_tool(tool_client) -> None:
    """The listing can report anything; the registry cannot lie."""

    session_id = _session_id(tool_client)

    tool_client.post(
        f"/api/tools/{session_id}/tools/alpha_tool",
        json={"enabled": False},
    )

    session = tool_client.app.state.session_manager._sessions[
        session_id
    ]

    registry = session.agent.registry

    assert registry.has("alpha_tool") is False
    assert registry.has("beta_tool") is True


def test_registered_count_excludes_disabled(tool_client) -> None:
    session_id = _session_id(tool_client)

    before = tool_client.get(
        f"/api/tools/{session_id}"
    ).json()["summary"]["registered_tools"]

    tool_client.post(
        f"/api/tools/{session_id}/tools/alpha_tool",
        json={"enabled": False},
    )

    after = tool_client.get(
        f"/api/tools/{session_id}"
    ).json()["summary"]["registered_tools"]

    assert after == before - 1


def test_the_choice_is_persisted(tool_client) -> None:
    """A switch that resets on restart looks like it never worked."""

    session_id = _session_id(tool_client)

    tool_client.post(
        f"/api/tools/{session_id}/tools/alpha_tool",
        json={"enabled": False},
    )

    from agent_workflow.core.application.settings_service import (
        CONFIG_FILE,
    )
    from agent_workflow.web.api.tools import _settings_of

    class _Request:
        def __init__(self, app) -> None:
            self.app = app

    service = _settings_of(
        _Request(tool_client.app)  # type: ignore[arg-type]
    )

    document = service.read(CONFIG_FILE)

    assert "alpha_tool" in document.data.get("tools", {}).get(
        "disabled",
        [],
    )


# The other half -- a config entry hiding a tool in a freshly built
# runtime -- is covered by
# test_a_tool_disabled_by_config_stays_hidden_when_it_arrives and needs
# a real runtime, which the transport stub deliberately is not.

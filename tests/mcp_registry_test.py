"""MCP server registry: create, update, delete and enable from the UI."""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from agent_workflow.core.infrastructure.config import MCPConfig
from agent_workflow.core.infrastructure.mcp.manager import MCPManager


def _attach_manager(
    client: TestClient,
    session_id: str,
    config: MCPConfig | None = None,
) -> None:
    """Give the session a live MCP manager.

    The stub runtime ships without one, and the registry endpoints
    only reconcile a session that actually has an MCP runtime.
    """

    session = client.portal.call(  # type: ignore[attr-defined]
        client.app.state.session_manager.get_session,
        session_id,
    )

    manager = MCPManager(
        config or MCPConfig(enabled=False),
        session.agent.registry,
    )

    session.runtime.mcp_manager = manager


def _config_path() -> Path:
    from agent_workflow.core.infrastructure.paths import (
        AgentWorkflowPaths,
    )

    return AgentWorkflowPaths().config


def _stored_mcp() -> dict:
    path = _config_path()

    if not path.exists():
        return {}

    with path.open("rb") as file:
        return tomllib.load(file).get("mcp", {})


def _session(client: TestClient) -> str:
    return client.post("/api/sessions").json()["session_id"]


# ======================================================================
# Creating a server
# ======================================================================


def test_create_stdio_server_persists_config(
    stubbed: TestClient,
) -> None:
    session_id = _session(stubbed)
    _attach_manager(stubbed, session_id)

    response = stubbed.post(
        f"/api/mcp/{session_id}/servers",
        json={
            "name": "stash",
            "transport": "stdio",
            "command": "npx",
            "args": ["-y", "stash-mcp"],
            "env": {"STASH_URL": "http://127.0.0.1:9999"},
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert body["enabled"] is True
    assert [server["name"] for server in body["servers"]] == ["stash"]

    stored = _stored_mcp()

    assert stored["enabled"] is True
    assert stored["servers"]["stash"]["command"] == "npx"
    assert stored["servers"]["stash"]["args"] == ["-y", "stash-mcp"]
    assert stored["servers"]["stash"]["env"] == {
        "STASH_URL": "http://127.0.0.1:9999"
    }


def test_creating_a_server_unblocks_the_disabled_panel(
    stubbed: TestClient,
) -> None:
    """Regression: the panel refused to render without an [mcp]
    section, so the UI that creates the section was unreachable."""

    session_id = _session(stubbed)
    _attach_manager(stubbed, session_id)

    assert stubbed.get(f"/api/mcp/{session_id}").json()["enabled"] is False

    stubbed.post(
        f"/api/mcp/{session_id}/servers",
        json={
            "name": "demo",
            "command": sys.executable,
            "args": ["-c", "pass"],
        },
    )

    body = stubbed.get(f"/api/mcp/{session_id}").json()

    assert body["enabled"] is True
    assert body["servers"]


def test_create_http_server(stubbed: TestClient) -> None:
    session_id = _session(stubbed)
    _attach_manager(stubbed, session_id)

    response = stubbed.post(
        f"/api/mcp/{session_id}/servers",
        json={
            "name": "remote",
            "transport": "http",
            "url": "http://127.0.0.1:8931/mcp",
            "headers": {"Authorization": "Bearer x"},
        },
    )

    assert response.status_code == 200

    stored = _stored_mcp()

    assert stored["servers"]["remote"]["transport"] == "http"
    assert stored["servers"]["remote"]["url"] == (
        "http://127.0.0.1:8931/mcp"
    )


def test_updating_a_server_replaces_its_definition(
    stubbed: TestClient,
) -> None:
    session_id = _session(stubbed)
    _attach_manager(stubbed, session_id)

    stubbed.post(
        f"/api/mcp/{session_id}/servers",
        json={"name": "demo", "command": "one"},
    )

    stubbed.post(
        f"/api/mcp/{session_id}/servers",
        json={"name": "demo", "command": "two"},
    )

    stored = _stored_mcp()

    assert list(stored["servers"]) == ["demo"]
    assert stored["servers"]["demo"]["command"] == "two"


def test_adding_a_server_keeps_the_other_servers(
    stubbed: TestClient,
) -> None:
    session_id = _session(stubbed)
    _attach_manager(stubbed, session_id)

    stubbed.post(
        f"/api/mcp/{session_id}/servers",
        json={"name": "first", "command": "a"},
    )

    stubbed.post(
        f"/api/mcp/{session_id}/servers",
        json={"name": "second", "command": "b"},
    )

    stored = _stored_mcp()

    assert sorted(stored["servers"]) == ["first", "second"]


def test_config_written_by_the_registry_loads(
    stubbed: TestClient,
) -> None:
    """The loader is the validator: what we persist must parse."""

    from agent_workflow.core.infrastructure.config import ConfigLoader

    session_id = _session(stubbed)
    _attach_manager(stubbed, session_id)

    stubbed.post(
        f"/api/mcp/{session_id}/servers",
        json={
            "name": "demo",
            "command": "npx",
            "args": ["-y", "pkg"],
            "env": {"A": "1"},
        },
    )

    config = ConfigLoader().load(_config_path())

    assert config.mcp.enabled is True
    assert config.mcp.require_approval is True

    [entry] = config.mcp.servers

    assert entry.name == "demo"
    assert entry.command == "npx"
    assert entry.args == ("-y", "pkg")
    assert entry.env == {"A": "1"}


# ======================================================================
# Validation
# ======================================================================


@pytest.mark.parametrize(
    "payload",
    [
        {"name": "", "command": "x"},
        {"name": "bad name", "command": "x"},
        {"name": "demo"},
        {"name": "demo", "transport": "carrier-pigeon", "command": "x"},
        {"name": "demo", "transport": "http"},
        {"name": "demo", "transport": "stdio", "command": "  "},
    ],
)
def test_invalid_server_is_rejected(
    stubbed: TestClient,
    payload: dict,
) -> None:
    session_id = _session(stubbed)
    _attach_manager(stubbed, session_id)

    response = stubbed.post(
        f"/api/mcp/{session_id}/servers",
        json=payload,
    )

    assert response.status_code == 422

    # Nothing was written.
    assert _stored_mcp() == {}


# ======================================================================
# Deleting
# ======================================================================


def test_delete_server_removes_it_from_config(
    stubbed: TestClient,
) -> None:
    session_id = _session(stubbed)
    _attach_manager(stubbed, session_id)

    stubbed.post(
        f"/api/mcp/{session_id}/servers",
        json={"name": "demo", "command": "x"},
    )

    response = stubbed.delete(f"/api/mcp/{session_id}/servers/demo")

    assert response.status_code == 200
    assert response.json()["servers"] == []

    # The empty table is dropped rather than left behind.
    assert "servers" not in _stored_mcp()


def test_delete_keeps_other_servers(stubbed: TestClient) -> None:
    session_id = _session(stubbed)
    _attach_manager(stubbed, session_id)

    stubbed.post(
        f"/api/mcp/{session_id}/servers",
        json={"name": "keep", "command": "a"},
    )
    stubbed.post(
        f"/api/mcp/{session_id}/servers",
        json={"name": "drop", "command": "b"},
    )

    stubbed.delete(f"/api/mcp/{session_id}/servers/drop")

    assert list(_stored_mcp()["servers"]) == ["keep"]


def test_delete_unknown_server(stubbed: TestClient) -> None:
    session_id = _session(stubbed)
    _attach_manager(stubbed, session_id)

    stubbed.post(
        f"/api/mcp/{session_id}/servers",
        json={"name": "demo", "command": "x"},
    )

    response = stubbed.delete(f"/api/mcp/{session_id}/servers/ghost")

    assert response.status_code == 404
    assert "ghost" in response.json()["detail"]


def test_delete_on_unknown_session(stubbed: TestClient) -> None:
    assert stubbed.delete("/api/mcp/ghost/servers/demo").status_code == 404


# ======================================================================
# Enable / disable
# ======================================================================


def test_enable_writes_the_section(stubbed: TestClient) -> None:
    session_id = _session(stubbed)
    _attach_manager(stubbed, session_id)

    response = stubbed.post(
        f"/api/mcp/{session_id}/enabled",
        json={"enabled": True},
    )

    assert response.status_code == 200
    assert response.json()["enabled"] is True
    assert _stored_mcp()["enabled"] is True


def test_disable_removes_servers_from_config(
    stubbed: TestClient,
) -> None:
    """Disabling must not leave dead entries behind that would
    silently reconnect on the next session."""

    session_id = _session(stubbed)
    _attach_manager(stubbed, session_id)

    stubbed.post(
        f"/api/mcp/{session_id}/servers",
        json={"name": "demo", "command": "x"},
    )

    response = stubbed.post(
        f"/api/mcp/{session_id}/enabled",
        json={"enabled": False},
    )

    assert response.status_code == 200

    body = response.json()

    assert body["enabled"] is False
    assert body["servers"] == []

    stored = _stored_mcp()

    assert stored["enabled"] is False
    assert "servers" not in stored

def test_editing_args_preserves_env(stubbed: TestClient) -> None:
    """Env values never reach the browser, so an edit that omits them
    must not be read as "clear the environment"."""

    session_id = _session(stubbed)
    _attach_manager(stubbed, session_id)

    stubbed.post(
        f"/api/mcp/{session_id}/servers",
        json={
            "name": "stash",
            "command": "podman",
            "args": ["run", "old"],
            "env": {"STASH_URL": "http://127.0.0.1:9999"},
            "headers": {"X-Token": "secret"},
        },
    )

    stubbed.post(
        f"/api/mcp/{session_id}/servers",
        json={"name": "stash", "command": "podman", "args": ["run", "new"]},
    )

    stored = _stored_mcp()["servers"]["stash"]

    assert stored["args"] == ["run", "new"]
    assert stored["env"] == {"STASH_URL": "http://127.0.0.1:9999"}
    assert stored["headers"] == {"X-Token": "secret"}


def test_env_can_be_cleared_explicitly(stubbed: TestClient) -> None:
    session_id = _session(stubbed)
    _attach_manager(stubbed, session_id)

    stubbed.post(
        f"/api/mcp/{session_id}/servers",
        json={"name": "stash", "command": "x", "env": {"A": "1"}},
    )

    stubbed.post(
        f"/api/mcp/{session_id}/servers",
        json={"name": "stash", "command": "x", "env": {}},
    )

    assert "env" not in _stored_mcp()["servers"]["stash"]


def test_env_is_never_returned_to_the_browser(
    stubbed: TestClient,
) -> None:
    session_id = _session(stubbed)
    _attach_manager(stubbed, session_id)

    stubbed.post(
        f"/api/mcp/{session_id}/servers",
        json={
            "name": "stash",
            "command": "x",
            "env": {"SECRET": "value"},
            "headers": {"Authorization": "Bearer secret"},
        },
    )

    body = stubbed.get(f"/api/mcp/{session_id}").text

    assert "SECRET" not in body
    assert "secret" not in body

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient


# ======================================================================
# Sessions CRUD
# ======================================================================


def test_create_session(stubbed: TestClient) -> None:
    response = stubbed.post("/api/sessions")

    assert response.status_code == 200

    body = response.json()

    assert body["session_id"]
    assert body["state"] == "idle"


def test_create_session_accepts_metadata(
    stubbed: TestClient,
) -> None:
    response = stubbed.post(
        "/api/sessions",
        json={"metadata": {"project": "demo"}},
    )

    assert response.status_code == 200

    session_id = response.json()["session_id"]

    info = stubbed.get(f"/api/sessions/{session_id}").json()

    assert info["metadata"] == {"project": "demo"}


def test_list_sessions_grows(stubbed: TestClient) -> None:
    before = len(stubbed.get("/api/sessions").json())

    stubbed.post("/api/sessions")
    stubbed.post("/api/sessions")

    assert len(stubbed.get("/api/sessions").json()) == before + 2


def test_get_session(stubbed: TestClient) -> None:
    session_id = stubbed.post(
        "/api/sessions"
    ).json()["session_id"]

    response = stubbed.get(f"/api/sessions/{session_id}")

    assert response.status_code == 200

    body = response.json()

    assert body["session_id"] == session_id
    assert body["model"] == "stub-model"
    assert body["session"]["state"] == "idle"
    assert body["latest_seq"] == 0
    # Internal observers (statistics) are always attached.
    assert body["subscriber_count"] >= 1


def test_get_unknown_session(stubbed: TestClient) -> None:
    assert stubbed.get("/api/sessions/missing").status_code == 404


def test_close_session(stubbed: TestClient) -> None:
    session_id = stubbed.post(
        "/api/sessions"
    ).json()["session_id"]

    assert (
        stubbed.delete(f"/api/sessions/{session_id}").status_code
        == 200
    )
    assert (
        stubbed.get(f"/api/sessions/{session_id}").status_code == 404
    )


def test_close_unknown_session(stubbed: TestClient) -> None:
    assert stubbed.delete("/api/sessions/missing").status_code == 404


def test_actions_on_unknown_session_are_404(
    stubbed: TestClient,
) -> None:
    assert (
        stubbed.post(
            "/api/sessions/missing/run",
            json={"prompt": "x"},
        ).status_code
        == 404
    )
    assert (
        stubbed.post(
            "/api/sessions/missing/continue",
            json={"prompt": "x"},
        ).status_code
        == 404
    )
    assert (
        stubbed.post("/api/sessions/missing/resume").status_code == 404
    )
    assert (
        stubbed.post("/api/sessions/missing/cancel").status_code == 404
    )
    assert (
        stubbed.post(
            "/api/sessions/missing/approvals/a/allow"
        ).status_code
        == 404
    )
    assert (
        stubbed.post(
            "/api/sessions/missing/approvals/a/deny"
        ).status_code
        == 404
    )


# ======================================================================
# Execution
# ======================================================================


def test_run_accepts_prompt(stubbed: TestClient) -> None:
    session_id = stubbed.post(
        "/api/sessions"
    ).json()["session_id"]

    response = stubbed.post(
        f"/api/sessions/{session_id}/run",
        json={"prompt": "hello"},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "started"


def test_run_rejects_empty_prompt(
    stubbed: TestClient,
) -> None:
    session_id = stubbed.post(
        "/api/sessions"
    ).json()["session_id"]

    response = stubbed.post(
        f"/api/sessions/{session_id}/run",
        json={"prompt": ""},
    )

    assert response.status_code == 422


def test_run_twice_conflicts_while_running(
    stubbed: TestClient,
    stub_factory,
    portal,
) -> None:
    """A second run must be refused while one is in flight."""

    import asyncio

    from agent_workflow.core.application import session as session_module

    original = session_module.create_runtime
    base = stub_factory

    gate = asyncio.Event()

    async def gated_create_runtime(
        approval_handler,
        working_directory=None,
        stats_store=None,
    ):
        runtime = await base(
            approval_handler,
            working_directory,
        )

        class GatedLLM(type(runtime.agent._llm)):  # type: ignore[attr-defined]
            async def chat(self, messages, tools=()):
                await gate.wait()

                return await super().chat(messages, tools)

        runtime.agent._llm = GatedLLM()

        return runtime

    session_module.create_runtime = gated_create_runtime

    try:
        session_id = stubbed.post(
            "/api/sessions"
        ).json()["session_id"]

        first = stubbed.post(
            f"/api/sessions/{session_id}/run",
            json={"prompt": "one"},
        )

        assert first.status_code == 200

        second = stubbed.post(
            f"/api/sessions/{session_id}/run",
            json={"prompt": "two"},
        )

        assert second.status_code == 409

        portal.call(gate.set)

        stubbed.post(f"/api/sessions/{session_id}/cancel")

    finally:
        session_module.create_runtime = original


def test_continue_is_available(
    stubbed: TestClient,
) -> None:
    session_id = stubbed.post(
        "/api/sessions"
    ).json()["session_id"]

    assert (
        stubbed.post(
            f"/api/sessions/{session_id}/run",
            json={"prompt": "one"},
        ).status_code
        == 200
    )
    assert (
        stubbed.post(
            f"/api/sessions/{session_id}/continue",
            json={"prompt": "two"},
        ).status_code
        == 200
    )


def test_resume_without_prepared_task(
    stubbed: TestClient,
) -> None:
    session_id = stubbed.post(
        "/api/sessions"
    ).json()["session_id"]

    response = stubbed.post(f"/api/sessions/{session_id}/resume")

    assert response.status_code in (200, 409)


def test_cancel_without_run_reports_false(
    stubbed: TestClient,
) -> None:
    session_id = stubbed.post(
        "/api/sessions"
    ).json()["session_id"]

    response = stubbed.post(f"/api/sessions/{session_id}/cancel")

    assert response.status_code == 200
    assert response.json()["cancelled"] is False


# ======================================================================
# Approval
# ======================================================================


def test_no_approval_pending_returns_none(
    stubbed: TestClient,
) -> None:
    session_id = stubbed.post(
        "/api/sessions"
    ).json()["session_id"]

    response = stubbed.get(f"/api/sessions/{session_id}/approval")

    assert response.status_code == 200
    assert response.json() is None


def test_approval_without_pending_is_conflict(
    stubbed: TestClient,
) -> None:
    session_id = stubbed.post(
        "/api/sessions"
    ).json()["session_id"]

    assert (
        stubbed.post(
            f"/api/sessions/{session_id}/approvals/x/allow"
        ).status_code
        == 409
    )
    assert (
        stubbed.post(
            f"/api/sessions/{session_id}/approvals/x/deny"
        ).status_code
        == 409
    )


def test_allow_resolves_pending_request(
    stubbed: TestClient,
    portal,
) -> None:
    import asyncio

    from agent_workflow.core.entities.models.approval import (
        ApprovalRequest,
    )

    session_id = stubbed.post(
        "/api/sessions"
    ).json()["session_id"]

    session = stubbed.portal.call(  # type: ignore[attr-defined]
        stubbed.app.state.session_manager.get_session,
        session_id,
    )

    request = ApprovalRequest(
        tool_name="execute_shell",
        arguments={"command": "curl https://x.dev"},
        permission="shell.network",
        reason="network",
    )

    async def scenario() -> None:
        task = asyncio.create_task(session.approval(request))

        await asyncio.sleep(0.02)

        pending = await asyncio.to_thread(
            stubbed.get,
            f"/api/sessions/{session_id}/approval",
        )

        assert pending.status_code == 200
        assert pending.json()["permission"] == "shell.network"

        approval_id = pending.json()["approval_id"]

        stale = await asyncio.to_thread(
            stubbed.post,
            f"/api/sessions/{session_id}"
            f"/approvals/wrong-id/allow",
        )

        assert stale.status_code == 409

        accepted = await asyncio.to_thread(
            stubbed.post,
            f"/api/sessions/{session_id}"
            f"/approvals/{approval_id}/allow",
        )

        assert accepted.status_code == 200
        assert await task is True

    portal.call(scenario)


def test_deny_resolves_pending_request(
    stubbed: TestClient,
    portal,
) -> None:
    import asyncio

    from agent_workflow.core.entities.models.approval import (
        ApprovalRequest,
    )

    session_id = stubbed.post(
        "/api/sessions"
    ).json()["session_id"]

    session = stubbed.portal.call(  # type: ignore[attr-defined]
        stubbed.app.state.session_manager.get_session,
        session_id,
    )

    request = ApprovalRequest(
        tool_name="execute_shell",
        arguments={"command": "rm -rf build"},
        permission="shell.destructive",
        reason="destructive",
    )

    async def scenario() -> None:
        task = asyncio.create_task(session.approval(request))

        await asyncio.sleep(0.02)

        approval_id = session.approval.approval_id

        denied = await asyncio.to_thread(
            stubbed.post,
            f"/api/sessions/{session_id}"
            f"/approvals/{approval_id}/deny",
        )

        assert denied.status_code == 200
        assert await task is False

    portal.call(scenario)


# ======================================================================
# Read-only observability
# ======================================================================


def test_skills_endpoint(stubbed: TestClient) -> None:
    session_id = stubbed.post(
        "/api/sessions"
    ).json()["session_id"]

    response = stubbed.get(f"/api/skills/{session_id}")

    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_skills_unknown_session(stubbed: TestClient) -> None:
    assert stubbed.get("/api/skills/missing").status_code == 404


def test_plugins_endpoint(stubbed: TestClient) -> None:
    session_id = stubbed.post(
        "/api/sessions"
    ).json()["session_id"]

    response = stubbed.get(f"/api/plugins/{session_id}")

    assert response.status_code == 200
    assert response.json() == []


def test_plugins_unknown_session(stubbed: TestClient) -> None:
    assert stubbed.get("/api/plugins/missing").status_code == 404


def test_system_info(stubbed: TestClient) -> None:
    session_id = stubbed.post(
        "/api/sessions"
    ).json()["session_id"]

    response = stubbed.get(f"/api/system/{session_id}")

    assert response.status_code == 200

    body = response.json()

    assert body["model"] == "stub-model"
    assert body["version"]
    assert body["working_directory"]


def test_system_unknown_session(stubbed: TestClient) -> None:
    assert stubbed.get("/api/system/missing").status_code == 404


# ======================================================================
# Per-session working directory
# ======================================================================


def test_create_session_with_working_directory(
    stubbed: TestClient,
    tmp_path: Path,
) -> None:
    response = stubbed.post(
        "/api/sessions",
        json={"working_directory": str(tmp_path)},
    )

    assert response.status_code == 200

    session_id = response.json()["session_id"]

    info = stubbed.get(f"/api/sessions/{session_id}").json()

    assert info["working_directory"] == str(tmp_path.resolve())


def test_working_directory_accepts_tilde(
    stubbed: TestClient,
) -> None:
    response = stubbed.post(
        "/api/sessions",
        json={"working_directory": "~"},
    )

    assert response.status_code == 200


def test_missing_working_directory_is_rejected(
    stubbed: TestClient,
) -> None:
    response = stubbed.post(
        "/api/sessions",
        json={"working_directory": "/definitely/not/here/xyz"},
    )

    assert response.status_code == 400
    assert "not accessible" in response.json()["detail"]


def test_file_as_working_directory_is_rejected(
    stubbed: TestClient,
    tmp_path: Path,
) -> None:
    target = tmp_path / "a-file.txt"
    target.write_text("x", encoding="utf-8")

    response = stubbed.post(
        "/api/sessions",
        json={"working_directory": str(target)},
    )

    assert response.status_code == 400
    assert "not a directory" in response.json()["detail"]


def test_blank_working_directory_uses_default(
    stubbed: TestClient,
) -> None:
    response = stubbed.post(
        "/api/sessions",
        json={"working_directory": "   "},
    )

    assert response.status_code == 200


# ======================================================================
# POST /messages
# ======================================================================


def test_message_auto_starts_run_first_turn(
    stubbed: TestClient,
) -> None:
    session_id = stubbed.post(
        "/api/sessions"
    ).json()["session_id"]

    response = stubbed.post(
        f"/api/sessions/{session_id}/messages",
        json={"content": "hello", "mode": "auto"},
    )

    assert response.status_code == 200
    assert response.json()["mode"] == "run"


def test_message_auto_continues_second_turn(
    stubbed: TestClient,
) -> None:
    session_id = stubbed.post(
        "/api/sessions"
    ).json()["session_id"]

    first = stubbed.post(
        f"/api/sessions/{session_id}/messages",
        json={"content": "one"},
    )

    assert first.json()["mode"] == "run"

    second = stubbed.post(
        f"/api/sessions/{session_id}/messages",
        json={"content": "two"},
    )

    assert second.status_code == 200
    assert second.json()["mode"] == "continue"


def test_message_explicit_run_mode(
    stubbed: TestClient,
) -> None:
    session_id = stubbed.post(
        "/api/sessions"
    ).json()["session_id"]

    response = stubbed.post(
        f"/api/sessions/{session_id}/messages",
        json={"content": "x", "mode": "run"},
    )

    assert response.json()["mode"] == "run"


def test_message_resume_mode_has_no_mode_label(
    stubbed: TestClient,
) -> None:
    session_id = stubbed.post(
        "/api/sessions"
    ).json()["session_id"]

    response = stubbed.post(
        f"/api/sessions/{session_id}/messages",
        json={"content": "x", "mode": "resume"},
    )

    assert response.status_code in (200, 409)
    assert response.json()["mode"] is None


def test_message_rejects_empty_content(
    stubbed: TestClient,
) -> None:
    session_id = stubbed.post(
        "/api/sessions"
    ).json()["session_id"]

    assert (
        stubbed.post(
            f"/api/sessions/{session_id}/messages",
            json={"content": ""},
        ).status_code
        == 422
    )
    assert (
        stubbed.post(
            f"/api/sessions/{session_id}/messages",
            json={"content": "   "},
        ).status_code
        == 422
    )


def test_message_rejects_unknown_mode(
    stubbed: TestClient,
) -> None:
    session_id = stubbed.post(
        "/api/sessions"
    ).json()["session_id"]

    assert (
        stubbed.post(
            f"/api/sessions/{session_id}/messages",
            json={"content": "x", "mode": "nonsense"},
        ).status_code
        == 422
    )


def test_message_on_unknown_session(stubbed: TestClient) -> None:
    assert (
        stubbed.post(
            "/api/sessions/missing/messages",
            json={"content": "x"},
        ).status_code
        == 404
    )


def test_message_conflicts_while_running(
    stubbed: TestClient,
    stub_factory,
    portal,
) -> None:
    import asyncio

    from agent_workflow.core.application import session as session_module

    original = session_module.create_runtime
    base = stub_factory

    gate = asyncio.Event()

    async def gated(
        approval_handler,
        working_directory=None,
        stats_store=None,
    ):
        runtime = await base(
            approval_handler,
            working_directory,
        )

        class GatedLLM(type(runtime.agent._llm)):  # type: ignore[attr-defined]
            async def chat(self, messages, tools=()):
                await gate.wait()

                return await super().chat(messages, tools)

        runtime.agent._llm = GatedLLM()

        return runtime

    session_module.create_runtime = gated

    try:
        session_id = stubbed.post(
            "/api/sessions"
        ).json()["session_id"]

        assert (
            stubbed.post(
                f"/api/sessions/{session_id}/messages",
                json={"content": "one"},
            ).status_code
            == 200
        )

        assert (
            stubbed.post(
                f"/api/sessions/{session_id}/messages",
                json={"content": "two"},
            ).status_code
            == 409
        )

        portal.call(gate.set)

        stubbed.post(f"/api/sessions/{session_id}/cancel")

    finally:
        session_module.create_runtime = original


# ======================================================================
# Tool inventory
# ======================================================================


def test_tools_endpoint_lists_registered_tools(
    stubbed: TestClient,
) -> None:
    session_id = stubbed.post(
        "/api/sessions"
    ).json()["session_id"]

    response = stubbed.get(f"/api/tools/{session_id}")

    assert response.status_code == 200

    body = response.json()

    assert body["tools"] == []
    assert body["summary"]["total_calls"] == 0


def test_tools_endpoint_unknown_session(
    stubbed: TestClient,
) -> None:
    assert stubbed.get("/api/tools/missing").status_code == 404


def test_tools_endpoint_describes_builtin_tool(
    stubbed: TestClient,
    tmp_path: Path,
) -> None:
    """The stub runtime has no tools; use a real registry instead."""

    from agent_workflow.core.entities.models.builtin.registry import (
        create_builtin_registry,
    )

    session_id = stubbed.post(
        "/api/sessions"
    ).json()["session_id"]

    session = stubbed.portal.call(  # type: ignore[attr-defined]
        stubbed.app.state.session_manager.get_session,
        session_id,
    )

    for tool in create_builtin_registry().all():
        session.agent.registry.register(tool)

    body = stubbed.get(f"/api/tools/{session_id}").json()

    names = {tool["name"] for tool in body["tools"]}

    assert "read_file" in names

    read_file = next(
        tool
        for tool in body["tools"]
        if tool["name"] == "read_file"
    )

    assert read_file["source"] == "builtin"
    assert read_file["description"]
    assert read_file["permissions"] == ["filesystem.read"]
    assert read_file["requires_approval"] is False
    assert read_file["timeout"] > 0

    params = {
        parameter["name"]
        for parameter in read_file["parameters"]
    }

    assert "path" in params


def test_tools_endpoint_reports_missing_permissions(
    stubbed: TestClient,
) -> None:
    from agent_workflow.core.entities.models.builtin.registry import (
        create_builtin_registry,
    )

    session_id = stubbed.post(
        "/api/sessions"
    ).json()["session_id"]

    session = stubbed.portal.call(  # type: ignore[attr-defined]
        stubbed.app.state.session_manager.get_session,
        session_id,
    )

    for tool in create_builtin_registry().all():
        session.agent.registry.register(tool)

    body = stubbed.get(f"/api/tools/{session_id}").json()

    # The stub context grants nothing, so every permission is
    # reported as missing.
    read_file = next(
        tool
        for tool in body["tools"]
        if tool["name"] == "read_file"
    )

    assert read_file["missing_permissions"] == ["filesystem.read"]


def test_tools_endpoint_includes_usage_stats(
    stubbed: TestClient,
) -> None:
    session_id = stubbed.post(
        "/api/sessions"
    ).json()["session_id"]

    session = stubbed.portal.call(  # type: ignore[attr-defined]
        stubbed.app.state.session_manager.get_session,
        session_id,
    )

    from agent_workflow.core.entities.models.agent_trace import (
        ToolFinished,
        ToolStarted,
    )

    stubbed.portal.call(  # type: ignore[attr-defined]
        session.events.publish,
        ToolStarted(
            iteration=1,
            tool_call_id="c1",
            tool_name="read_file",
            arguments={},
        ),
    )
    stubbed.portal.call(  # type: ignore[attr-defined]
        session.events.publish,
        ToolFinished(
            iteration=1,
            tool_call_id="c1",
            tool_name="read_file",
            output="ok",
            error_code=None,
            error_message=None,
            duration_seconds=0.5,
        ),
    )

    body = stubbed.get(f"/api/tools/{session_id}").json()

    assert body["summary"]["total_calls"] == 1
    assert body["summary"]["total_successes"] == 1
    assert body["summary"]["error_rate"] == 0.0


# ======================================================================
# MCP
# ======================================================================


def test_mcp_disabled_by_default(stubbed: TestClient) -> None:
    session_id = stubbed.post(
        "/api/sessions"
    ).json()["session_id"]

    response = stubbed.get(f"/api/mcp/{session_id}")

    assert response.status_code == 200

    body = response.json()

    assert body["enabled"] is False
    assert body["servers"] == []
    assert body["total_tools"] == 0


def test_mcp_unknown_session(stubbed: TestClient) -> None:
    assert stubbed.get("/api/mcp/missing").status_code == 404


def test_mcp_reports_failed_server(
    stubbed: TestClient,
) -> None:
    """A server that cannot start is reported, not raised."""

    from agent_workflow.core.infrastructure.config import (
        MCPConfig,
        MCPServerConfigEntry,
    )
    from agent_workflow.core.infrastructure.mcp.manager import MCPManager

    session_id = stubbed.post(
        "/api/sessions"
    ).json()["session_id"]

    session = stubbed.portal.call(  # type: ignore[attr-defined]
        stubbed.app.state.session_manager.get_session,
        session_id,
    )

    manager = MCPManager(
        MCPConfig(
            enabled=True,
            servers=(
                MCPServerConfigEntry(
                    name="broken",
                    command="definitely-not-a-real-binary-xyz",
                ),
            ),
        ),
        session.agent.registry,
    )

    stubbed.portal.call(  # type: ignore[attr-defined]
        manager.connect_all
    )

    session.runtime.mcp_manager = manager

    body = stubbed.get(f"/api/mcp/{session_id}").json()

    assert body["enabled"] is True
    assert body["connected_servers"] == 0
    assert body["total_tools"] == 0
    assert body["permissions"] == ["mcp.execute"]

    [server] = body["servers"]

    assert server["name"] == "broken"
    assert server["state"] == "failed"
    assert server["error"]


def test_mcp_reports_connected_server_and_tools(
    stubbed: TestClient,
) -> None:
    from pathlib import Path as _Path

    import sys

    from agent_workflow.core.infrastructure.config import (
        MCPConfig,
        MCPServerConfigEntry,
    )
    from agent_workflow.core.infrastructure.mcp.manager import MCPManager

    server_script = str(
        (
            _Path(__file__).resolve().parents[1]
            / "examples"
            / "mcp_stdio_server.py"
        )
    )

    session_id = stubbed.post(
        "/api/sessions"
    ).json()["session_id"]

    session = stubbed.portal.call(  # type: ignore[attr-defined]
        stubbed.app.state.session_manager.get_session,
        session_id,
    )

    manager = MCPManager(
        MCPConfig(
            enabled=True,
            require_approval=False,
            servers=(
                MCPServerConfigEntry(
                    name="demo",
                    command=sys.executable,
                    args=(server_script,),
                ),
            ),
        ),
        session.agent.registry,
    )

    stubbed.portal.call(  # type: ignore[attr-defined]
        manager.connect_all
    )

    session.runtime.mcp_manager = manager

    body = stubbed.get(f"/api/mcp/{session_id}").json()

    assert body["connected_servers"] == 1
    assert body["total_tools"] == 3

    [server] = body["servers"]

    assert server["state"] == "connected"
    assert server["server_version"] == "1.2.3"

    names = {tool["qualified_name"] for tool in server["tools"]}

    assert names == {
        "mcp_demo_add",
        "mcp_demo_echo",
        "mcp_demo_explode",
    }

    # The same tools must appear in the tool inventory.
    tools = stubbed.get(f"/api/tools/{session_id}").json()["tools"]

    mcp_tools = [
        tool for tool in tools if tool["source"] == "mcp"
    ]

    assert len(mcp_tools) == 3

    add = next(
        tool for tool in mcp_tools if tool["name"] == "mcp_demo_add"
    )

    assert add["required_parameters"] == ["a", "b"]
    assert add["permissions"] == ["mcp.execute"]
    assert add["requires_approval"] is False


# ======================================================================
# MCP runtime control
# ======================================================================


def _attach_mcp(stubbed: TestClient, tmp_path: Path, **config_kwargs):
    """Give a session a real MCPManager backed by the test server."""

    import sys

    from agent_workflow.core.infrastructure.config import (
        MCPConfig,
        MCPServerConfigEntry,
    )
    from agent_workflow.core.infrastructure.mcp.manager import MCPManager

    script = str(
        (
            Path(__file__).resolve().parents[1]
            / "examples"
            / "mcp_stdio_server.py"
        )
    )

    session_id = stubbed.post(
        "/api/sessions",
        json={"working_directory": str(tmp_path)},
    ).json()["session_id"]

    session = stubbed.portal.call(  # type: ignore[attr-defined]
        stubbed.app.state.session_manager.get_session,
        session_id,
    )

    manager = MCPManager(
        MCPConfig(
            enabled=True,
            servers=(
                MCPServerConfigEntry(
                    name="demo",
                    command=sys.executable,
                    args=(script,),
                ),
            ),
            **config_kwargs,
        ),
        session.agent.registry,
    )

    session.runtime.mcp_manager = manager

    return session_id, session, manager


def test_mcp_connect_server(stubbed: TestClient, tmp_path: Path) -> None:
    session_id, _session, manager = _attach_mcp(
        stubbed, tmp_path
    )

    response = stubbed.post(
        f"/api/mcp/{session_id}/servers/demo/connect"
    )

    assert response.status_code == 200

    body = response.json()

    assert body["state"] == "connected"
    assert body["total_tools"] == 3
    assert manager.is_connected("demo") is True


def test_mcp_disconnect_removes_tools(
    stubbed: TestClient,
    tmp_path: Path,
) -> None:
    session_id, session, manager = _attach_mcp(
        stubbed, tmp_path
    )

    stubbed.post(f"/api/mcp/{session_id}/servers/demo/connect")

    assert session.agent.registry.has("mcp_demo_echo") is True

    response = stubbed.post(
        f"/api/mcp/{session_id}/servers/demo/disconnect"
    )

    assert response.status_code == 200
    assert response.json()["state"] == "closed"
    assert session.agent.registry.has("mcp_demo_echo") is False
    assert manager.tools() == []


def test_mcp_reload_server(stubbed: TestClient, tmp_path: Path) -> None:
    session_id, _session, manager = _attach_mcp(
        stubbed, tmp_path
    )

    stubbed.post(f"/api/mcp/{session_id}/servers/demo/connect")

    response = stubbed.post(
        f"/api/mcp/{session_id}/servers/demo/reload"
    )

    assert response.status_code == 200
    assert response.json()["state"] == "connected"
    assert manager.is_connected("demo") is True


def test_mcp_reload_all(stubbed: TestClient, tmp_path: Path) -> None:
    session_id, _session, manager = _attach_mcp(
        stubbed, tmp_path
    )

    response = stubbed.post(f"/api/mcp/{session_id}/reload")

    assert response.status_code == 200
    assert response.json()["total_tools"] == 3
    assert manager.is_connected("demo") is True


def test_mcp_permissions_refresh_on_connect(
    stubbed: TestClient,
    tmp_path: Path,
) -> None:
    """A connected server's permissions must reach ToolContext."""

    session_id, session, _manager = _attach_mcp(
        stubbed, tmp_path, default_permissions=("mcp.execute",)
    )

    assert "mcp.execute" not in session.context.permissions

    stubbed.post(f"/api/mcp/{session_id}/servers/demo/connect")

    assert "mcp.execute" in session.context.permissions


def test_mcp_unknown_server_action(stubbed: TestClient) -> None:
    session_id = stubbed.post(
        "/api/sessions"
    ).json()["session_id"]

    response = stubbed.post(
        f"/api/mcp/{session_id}/servers/ghost/connect"
    )

    assert response.status_code == 409


def test_mcp_unknown_server_is_404_when_enabled(
    stubbed: TestClient,
    tmp_path: Path,
) -> None:
    session_id, _session, _manager = _attach_mcp(
        stubbed, tmp_path
    )

    response = stubbed.post(
        f"/api/mcp/{session_id}/servers/ghost/connect"
    )

    assert response.status_code == 404


def test_mcp_unknown_action_is_404(
    stubbed: TestClient,
    tmp_path: Path,
) -> None:
    session_id, _session, _manager = _attach_mcp(
        stubbed, tmp_path
    )

    response = stubbed.post(
        f"/api/mcp/{session_id}/servers/demo/explode"
    )

    assert response.status_code == 404


def test_mcp_action_on_unknown_session(
    stubbed: TestClient,
) -> None:
    assert (
        stubbed.post(
            "/api/mcp/missing/servers/demo/connect"
        ).status_code
        == 404
    )
    assert (
        stubbed.post("/api/mcp/missing/reload").status_code == 404
    )


def test_mcp_control_requires_enabled(
    stubbed: TestClient,
) -> None:
    session_id = stubbed.post(
        "/api/sessions"
    ).json()["session_id"]

    assert (
        stubbed.post(
            f"/api/mcp/{session_id}/servers/demo/connect"
        ).status_code
        == 409
    )
    assert (
        stubbed.post(f"/api/mcp/{session_id}/reload").status_code
        == 409
    )


def test_mcp_connect_failure_reports_state(
    stubbed: TestClient,
    tmp_path: Path,
) -> None:
    """A server that cannot start is reported, not raised."""

    session_id, session, _manager = _attach_mcp(
        stubbed, tmp_path
    )

    manager = session.runtime.mcp_manager

    # Point the server at a binary that cannot start.
    entry = manager._entries["demo"]
    object.__setattr__(
        entry, "command", "definitely-not-real-xyz"
    )

    response = stubbed.post(
        f"/api/mcp/{session_id}/servers/demo/connect"
    )

    assert response.status_code == 200
    assert response.json()["state"] == "failed"
    assert response.json()["total_tools"] == 0
    assert session.agent.registry.has("mcp_demo_echo") is False

    detail = stubbed.get(f"/api/mcp/{session_id}").json()

    assert detail["servers"][0]["error"]

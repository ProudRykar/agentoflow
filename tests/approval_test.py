from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from agent_workflow.core.entities.models.agent_trace import AgentStarted
from agent_workflow.core.entities.models.approval import (
    ApprovalController,
    ApprovalDeniedError,
    ApprovalRequest,
    ApprovalRequested,
    ApprovalResolved,
)


def make_request() -> ApprovalRequest:
    return ApprovalRequest(
        tool_name="execute_shell",
        arguments={"command": "ls"},
        permission="shell.execute",
        reason="Shell command needs approval",
    )


async def test_allow_resumes_execution() -> None:
    controller = ApprovalController()

    task = asyncio.create_task(controller(make_request()))

    await asyncio.sleep(0.01)

    assert controller.active
    assert controller.request is not None

    assert controller.allow() is True
    assert await task is True
    assert controller.active is False
    assert controller.request is None


async def test_deny_resumes_with_false() -> None:
    controller = ApprovalController()

    task = asyncio.create_task(controller(make_request()))

    await asyncio.sleep(0.01)

    # deny() reports whether the decision was accepted.
    assert controller.deny() is True
    assert await task is False


async def test_each_request_gets_an_id() -> None:
    controller = ApprovalController()

    seen: list[str] = []

    async def on_event(event) -> None:
        if isinstance(event, ApprovalRequested):
            seen.append(event.approval_id)

    controller.set_on_event(on_event)

    for _ in range(2):
        task = asyncio.create_task(controller(make_request()))
        await asyncio.sleep(0.01)
        controller.allow()
        await task

    assert len(seen) == 2
    assert seen[0] != seen[1]


async def test_events_report_request_and_resolution() -> None:
    controller = ApprovalController()

    events: list[object] = []

    async def on_event(event) -> None:
        events.append(event)

    controller.set_on_event(on_event)

    task = asyncio.create_task(controller(make_request()))
    await asyncio.sleep(0.01)

    requested = events[0]
    assert isinstance(requested, ApprovalRequested)
    assert requested.tool_name == "execute_shell"
    assert requested.permission == "shell.execute"
    assert requested.arguments == {"command": "ls"}

    controller.allow(approval_id=requested.approval_id)
    await task
    await asyncio.sleep(0.01)

    resolved = events[-1]
    assert isinstance(resolved, ApprovalResolved)
    assert resolved.approval_id == requested.approval_id
    assert resolved.approved is True


async def test_concurrent_request_is_rejected() -> None:
    controller = ApprovalController()

    task = asyncio.create_task(controller(make_request()))
    await asyncio.sleep(0.01)

    with pytest.raises(
        RuntimeError,
        match="already active",
    ):
        await controller(make_request())

    controller.allow()
    await task


async def test_stale_approval_id_is_ignored() -> None:
    controller = ApprovalController()

    task = asyncio.create_task(controller(make_request()))
    await asyncio.sleep(0.01)

    current = controller.approval_id

    assert controller.allow(approval_id="not-the-current") is False
    assert controller.active is True

    assert controller.allow(approval_id=current) is True
    await task


async def test_decision_without_active_request_is_noop() -> None:
    controller = ApprovalController()

    assert controller.allow() is False
    assert controller.deny() is False


async def test_cancel_unblocks_waiting_agent() -> None:
    controller = ApprovalController()

    task = asyncio.create_task(controller(make_request()))
    await asyncio.sleep(0.01)

    assert await controller.cancel() is True
    assert await task is False
    assert controller.active is False


async def test_cancel_without_request_is_noop() -> None:
    controller = ApprovalController()

    assert await controller.cancel() is False


async def test_read_only_shell_needs_no_approval() -> None:
    """'ls' is read-only, so the hook must not block on it."""

    from pathlib import Path

    from agent_workflow.core.entities.models.approval_hook import (
        ApprovalHook,
    )
    from agent_workflow.core.entities.models.llm import LLMToolCall
    from agent_workflow.core.entities.models.path_policy import (
        PathPolicy,
    )
    from agent_workflow.core.entities.models.shell_policy import (
        ShellPolicy,
    )
    from agent_workflow.core.entities.models.tool import ToolContext

    controller = ApprovalController()

    hook = ApprovalHook(
        handler=controller,
        shell_policy=ShellPolicy(
            path_policy=PathPolicy(
                allowed_paths=(Path.cwd(),),
            ),
        ),
    )

    await hook.before_tool(
        iteration=1,
        tool_call=LLMToolCall(
            id="c1",
            name="execute_shell",
            arguments={"command": "ls"},
        ),
        context=ToolContext(
            working_directory=Path.cwd(),
            environment={},
            allowed_path=(Path.cwd(),),
            permissions=frozenset({"shell.execute"}),
        ),
    )

    assert controller.active is False


async def test_denial_raises_in_agent_flow() -> None:
    """ApprovalHook turns a False decision into ApprovalDeniedError."""

    from pathlib import Path

    from agent_workflow.core.entities.models.approval_hook import (
        ApprovalHook,
    )
    from agent_workflow.core.entities.models.llm import LLMToolCall
    from agent_workflow.core.entities.models.path_policy import (
        PathPolicy,
    )
    from agent_workflow.core.entities.models.shell_policy import (
        ShellPolicy,
    )
    from agent_workflow.core.entities.models.tool import ToolContext

    controller = ApprovalController()

    hook = ApprovalHook(
        handler=controller,
        shell_policy=ShellPolicy(
            path_policy=PathPolicy(
                allowed_paths=(Path.cwd(),),
            ),
        ),
    )

    # curl is a network command, so it is approvable.
    context = ToolContext(
        working_directory=Path.cwd(),
        environment={},
        allowed_path=(Path.cwd(),),
        permissions=frozenset({"shell.execute"}),
    )

    task = asyncio.create_task(
        hook.before_tool(
            iteration=1,
            tool_call=LLMToolCall(
                id="c1",
                name="execute_shell",
                arguments={"command": "curl https://x.dev"},
            ),
            context=context,
        )
    )

    await asyncio.sleep(0.01)

    assert controller.active is True

    controller.deny()

    with pytest.raises(ApprovalDeniedError):
        await task


async def test_allow_records_approved_permission() -> None:
    from pathlib import Path

    from agent_workflow.core.entities.models.approval_hook import (
        ApprovalHook,
    )
    from agent_workflow.core.entities.models.llm import LLMToolCall
    from agent_workflow.core.entities.models.path_policy import (
        PathPolicy,
    )
    from agent_workflow.core.entities.models.shell_policy import (
        ShellPolicy,
    )
    from agent_workflow.core.entities.models.tool import ToolContext

    controller = ApprovalController()

    hook = ApprovalHook(
        handler=controller,
        shell_policy=ShellPolicy(
            path_policy=PathPolicy(
                allowed_paths=(Path.cwd(),),
            ),
        ),
    )

    context = ToolContext(
        working_directory=Path.cwd(),
        environment={},
        allowed_path=(Path.cwd(),),
        permissions=frozenset({"shell.execute"}),
    )

    task = asyncio.create_task(
        hook.before_tool(
            iteration=1,
            tool_call=LLMToolCall(
                id="c1",
                name="execute_shell",
                arguments={"command": "curl https://x.dev"},
            ),
            context=context,
        )
    )

    await asyncio.sleep(0.01)

    controller.allow()

    await task

    assert "shell.network" in context.approved_permissions


async def test_on_change_is_notified() -> None:
    controller = ApprovalController()

    calls: list[int] = []

    controller.set_on_change(lambda: calls.append(1))

    task = asyncio.create_task(controller(make_request()))
    await asyncio.sleep(0.01)

    controller.allow()
    await task

    assert len(calls) == 2


async def test_approval_is_independent_of_transport() -> None:
    """The controller only needs a callable sink, no UI imports."""

    controller = ApprovalController()

    events: list[object] = []

    async def collect(event) -> None:
        events.append(event)

    controller.set_on_event(collect)

    task = asyncio.create_task(controller(make_request()))
    await asyncio.sleep(0.01)
    controller.allow()
    assert await task is True

    assert any(
        isinstance(event, ApprovalRequested)
        for event in events
    )


# ======================================================================
# A granted permission must stop being asked again
# ======================================================================


def _mcp_hook(remember: bool = True):
    """A hook over one tool that declares requires_approval.

    Mirrors an MCP tool: its declared permission is already held, so
    the only thing left to approve is the synthetic per-tool grant.
    """

    from agent_workflow.core.entities.models.approval_hook import (
        ApprovalHook,
    )
    from agent_workflow.core.entities.models.path_policy import (
        PathPolicy,
    )
    from agent_workflow.core.entities.models.shell_policy import (
        ShellPolicy,
    )
    from agent_workflow.core.entities.models.tool import (
        Tool,
        ToolContext,
        ToolPolicy,
    )
    from agent_workflow.core.entities.models.tool_registry import (
        ToolRegistry,
    )

    async def handler(arguments, context):
        del arguments, context

        return "ok"

    registry = ToolRegistry()

    registry.register(
        Tool(
            name="mcp_stash_update_tag_description",
            description="Update a tag description.",
            input_type=None,
            handler=handler,
            policy=ToolPolicy(
                permissions=frozenset({"mcp.execute"}),
                requires_approval=True,
                timeout=5.0,
                max_output_size=1_000,
            ),
        )
    )

    controller = ApprovalController()

    hook = ApprovalHook(
        handler=controller,
        shell_policy=ShellPolicy(
            path_policy=PathPolicy(
                allowed_paths=(Path.cwd(),),
            ),
        ),
        registry=registry,
        remember_approvals=remember,
    )

    context = ToolContext(
        working_directory=Path.cwd(),
        environment={},
        allowed_path=(Path.cwd(),),
        permissions=frozenset({"mcp.execute"}),
    )

    return hook, controller, context


async def _call(hook, context, name: str) -> None:
    from agent_workflow.core.entities.models.llm import LLMToolCall

    await hook.before_tool(
        iteration=1,
        tool_call=LLMToolCall(id="c1", name=name, arguments={}),
        context=context,
    )


async def test_bulk_work_asks_once_not_once_per_item() -> None:
    """The reason this matters: ten tags used to mean ten prompts.

    Approving once and then being asked again trains the operator to
    click allow without reading, which is worse than asking once.
    """

    from agent_workflow.core.entities.models.approval_hook import (
        ApprovalHook,
    )
    from agent_workflow.core.entities.models.path_policy import (
        PathPolicy,
    )
    from agent_workflow.core.entities.models.shell_policy import (
        ShellPolicy,
    )
    from agent_workflow.core.entities.models.tool import (
        Tool,
        ToolContext,
        ToolPolicy,
    )
    from agent_workflow.core.entities.models.tool_registry import (
        ToolRegistry,
    )

    asked: list[str] = []

    async def counting_handler(request):
        asked.append(request.permission)

        return True

    async def noop(arguments, context):
        del arguments, context

        return "ok"

    registry = ToolRegistry()

    registry.register(
        Tool(
            name="mcp_stash_update_tag_description",
            description="d",
            input_type=None,
            handler=noop,
            policy=ToolPolicy(
                permissions=frozenset({"mcp.execute"}),
                requires_approval=True,
                timeout=5.0,
                max_output_size=1_000,
            ),
        )
    )

    hook = ApprovalHook(
        handler=counting_handler,
        shell_policy=ShellPolicy(
            path_policy=PathPolicy(allowed_paths=(Path.cwd(),)),
        ),
        registry=registry,
    )

    context = ToolContext(
        working_directory=Path.cwd(),
        environment={},
        allowed_path=(Path.cwd(),),
        permissions=frozenset({"mcp.execute"}),
    )

    for index in range(10):
        await _call(hook, context, "mcp_stash_update_tag_description")

    assert len(asked) == 1
    assert asked[0] == "tool:mcp_stash_update_tag_description"


async def test_a_granted_permission_is_recorded_and_honoured() -> None:
    hook, controller, context = _mcp_hook()

    task = asyncio.create_task(
        _call(hook, context, "mcp_stash_update_tag_description")
    )

    await asyncio.sleep(0.01)
    controller.allow()
    await task

    assert (
        "tool:mcp_stash_update_tag_description"
        in context.approved_permissions
    )

    # The second call must not open a prompt at all.
    await _call(hook, context, "mcp_stash_update_tag_description")

    assert controller.active is False


async def test_remembering_is_per_tool() -> None:
    from agent_workflow.core.entities.models.approval_hook import (
        ApprovalHook,
    )
    from agent_workflow.core.entities.models.path_policy import (
        PathPolicy,
    )
    from agent_workflow.core.entities.models.shell_policy import (
        ShellPolicy,
    )
    from agent_workflow.core.entities.models.tool import (
        Tool,
        ToolContext,
        ToolPolicy,
    )
    from agent_workflow.core.entities.models.tool_registry import (
        ToolRegistry,
    )

    async def handler(arguments, context):
        del arguments, context

        return "ok"

    registry = ToolRegistry()

    for name in ("tool_a", "tool_b"):
        registry.register(
            Tool(
                name=name,
                description="d",
                input_type=None,
                handler=handler,
                policy=ToolPolicy(
                    permissions=frozenset({"mcp.execute"}),
                    requires_approval=True,
                    timeout=5.0,
                    max_output_size=1_000,
                ),
            )
        )

    controller = ApprovalController()

    hook = ApprovalHook(
        handler=controller,
        shell_policy=ShellPolicy(
            path_policy=PathPolicy(allowed_paths=(Path.cwd(),)),
        ),
        registry=registry,
    )

    context = ToolContext(
        working_directory=Path.cwd(),
        environment={},
        allowed_path=(Path.cwd(),),
        permissions=frozenset({"mcp.execute"}),
    )

    task = asyncio.create_task(_call(hook, context, "tool_a"))

    await asyncio.sleep(0.01)
    controller.allow()
    await task

    # A different tool is a different grant and must still ask.
    other = asyncio.create_task(_call(hook, context, "tool_b"))

    await asyncio.sleep(0.01)

    assert controller.active is True
    assert controller.request.tool_name == "tool_b"

    controller.deny()

    with pytest.raises(ApprovalDeniedError):
        await other


async def test_remembering_can_be_switched_off() -> None:
    """An operator who wants every single call prompted keeps that."""

    from agent_workflow.core.entities.models.approval_hook import (
        ApprovalHook,
    )
    from agent_workflow.core.entities.models.path_policy import (
        PathPolicy,
    )
    from agent_workflow.core.entities.models.shell_policy import (
        ShellPolicy,
    )
    from agent_workflow.core.entities.models.tool import (
        Tool,
        ToolContext,
        ToolPolicy,
    )
    from agent_workflow.core.entities.models.tool_registry import (
        ToolRegistry,
    )

    asked: list[str] = []

    async def counting_handler(request):
        asked.append(request.permission)

        return True

    async def noop(arguments, context):
        del arguments, context

        return "ok"

    registry = ToolRegistry()

    registry.register(
        Tool(
            name="mcp_stash_update_tag_description",
            description="d",
            input_type=None,
            handler=noop,
            policy=ToolPolicy(
                permissions=frozenset({"mcp.execute"}),
                requires_approval=True,
                timeout=5.0,
                max_output_size=1_000,
            ),
        )
    )

    hook = ApprovalHook(
        handler=counting_handler,
        shell_policy=ShellPolicy(
            path_policy=PathPolicy(allowed_paths=(Path.cwd(),)),
        ),
        registry=registry,
        remember_approvals=False,
    )

    context = ToolContext(
        working_directory=Path.cwd(),
        environment={},
        allowed_path=(Path.cwd(),),
        permissions=frozenset({"mcp.execute"}),
    )

    for _ in range(3):
        await _call(hook, context, "mcp_stash_update_tag_description")

    assert len(asked) == 3


async def test_a_denial_does_not_unlock_the_tool() -> None:
    hook, controller, context = _mcp_hook()

    task = asyncio.create_task(
        _call(hook, context, "mcp_stash_update_tag_description")
    )

    await asyncio.sleep(0.01)
    controller.deny()

    with pytest.raises(ApprovalDeniedError):
        await task

    assert context.approved_permissions == frozenset()

    # The next call must ask again.
    second = asyncio.create_task(
        _call(hook, context, "mcp_stash_update_tag_description")
    )

    await asyncio.sleep(0.01)

    assert controller.active is True

    controller.deny()

    with pytest.raises(ApprovalDeniedError):
        await second

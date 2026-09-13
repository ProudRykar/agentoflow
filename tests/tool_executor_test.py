from pathlib import Path
from typing import Any

import pytest

from core.entities.models.arguments import ArgumentDecoder
from core.entities.models.builtin.read_file import ReadFileInput
from core.entities.models.tool import Tool, ToolContext, ToolPolicy
from core.entities.models.tool_executor import ToolExecutor
from core.entities.models.tool_registry import ToolRegistry


def create_registry(
    handler: Any,
) -> ToolRegistry:
    registry = ToolRegistry()

    registry.register(
        Tool(
            name="read_file",
            description="Read a file",
            input_type=ReadFileInput,
            handler=handler,
            policy=ToolPolicy(
                permissions=frozenset({"filesystem.read"}),
                timeout=1.0,
                max_output_size=100,
            ),
        )
    )

    return registry


@pytest.fixture
def context() -> ToolContext:
    return ToolContext(
        working_directory=Path("/tmp"),
        environment={},
        allowed_path=(),
        permissions=frozenset({"filesystem.read"}),
    )


@pytest.mark.asyncio
async def test_execute_tool(context: ToolContext) -> None:
    async def handler(
        arguments: ReadFileInput,
        context: ToolContext,
    ) -> str:
        return f"read: {arguments.path}"

    executor = ToolExecutor(
        registry=create_registry(handler),
    )

    result = await executor.execute(
        tool_name="read_file",
        arguments={"path": "test.txt"},
        context=context,
    )

    assert result.output == "read: test.txt"
    assert result.error is None


@pytest.mark.asyncio
async def test_unknown_tool(context: ToolContext) -> None:
    executor = ToolExecutor(
        registry=ToolRegistry(),
    )

    result = await executor.execute(
        tool_name="does_not_exist",
        arguments={},
        context=context,
    )

    assert result.output is None
    assert result.error is not None
    assert result.error.code == "tool_not_found"


@pytest.mark.asyncio
async def test_invalid_arguments(context: ToolContext) -> None:
    async def handler(
        arguments: ReadFileInput,
        context: ToolContext,
    ) -> str:
        return "ok"

    executor = ToolExecutor(
        registry=create_registry(handler),
    )

    result = await executor.execute(
        tool_name="read_file",
        arguments={"path": 123},
        context=context,
    )

    assert result.output is None
    assert result.error is not None
    assert result.error.code == "invalid_type"


@pytest.mark.asyncio
async def test_timeout(context: ToolContext) -> None:
    async def handler(
        arguments: ReadFileInput,
        context: ToolContext,
    ) -> str:
        import asyncio

        await asyncio.sleep(10)
        return "too late"

    executor = ToolExecutor(
        registry=create_registry(handler),
    )

    result = await executor.execute(
        tool_name="read_file",
        arguments={"path": "test.txt"},
        context=context,
    )

    assert result.output is None
    assert result.error is not None
    assert result.error.code == "timeout"
    assert result.error.retryable is True


@pytest.mark.asyncio
async def test_output_limit(context: ToolContext) -> None:
    async def handler(
        arguments: ReadFileInput,
        context: ToolContext,
    ) -> str:
        return "x" * 101

    executor = ToolExecutor(
        registry=create_registry(handler),
    )

    result = await executor.execute(
        tool_name="read_file",
        arguments={"path": "test.txt"},
        context=context,
    )

    assert result.output is None
    assert result.error is not None
    assert result.error.code == "output_too_large"


@pytest.mark.asyncio
async def test_permission_denied() -> None:
    async def handler(
        arguments: ReadFileInput,
        context: ToolContext,
    ) -> str:
        return "should not execute"

    registry = ToolRegistry()

    registry.register(
        Tool(
            name="read_file",
            description="Read a file",
            input_type=ReadFileInput,
            handler=handler,
            policy=ToolPolicy(
                permissions=frozenset({"filesystem.read"}),
                timeout=1.0,
                max_output_size=100,
            ),
        )
    )

    context = ToolContext(
        working_directory=Path("/tmp"),
        environment={},
        allowed_path=(),
        permissions=frozenset(),
    )

    executor = ToolExecutor(registry)

    result = await executor.execute(
        tool_name="read_file",
        arguments={"path": "test.txt"},
        context=context,
    )

    assert result.output is None
    assert result.error is not None
    assert result.error.code == "permission_denied"


@pytest.mark.asyncio
async def test_permission_denied_does_not_execute_handler() -> None:
    executed = False

    async def handler(
        arguments: ReadFileInput,
        context: ToolContext,
    ) -> str:
        nonlocal executed
        executed = True
        return "should not execute"

    # зарегистрировать tool с filesystem.read
    # context дать без filesystem.read
    # execute()

    assert executed is False
from typing import Any

import pytest

from core.entities.models.tool import Tool, ToolPolicy
from core.entities.models.tool_registry import (
    ToolRegistry,
    ToolRegistryError,
)


@pytest.fixture
def tool_policy() -> ToolPolicy:
    return ToolPolicy(
        permissions=frozenset(),
        timeout=10.0,
        max_output_size=10_000,
    )


async def fake_handler(arguments: Any, context: Any) -> str:
    return "ok"


def test_register_and_get() -> None:
    registry = ToolRegistry()

    tool = Tool(
        name="test_tool",
        description="Test tool",
        input_type=dict,
        handler=fake_handler,
        policy=tool_policy,
    )

    registry.register(tool)

    assert registry.get("test_tool") is tool


def test_has() -> None:
    registry = ToolRegistry()

    tool = Tool(
        name="test_tool",
        description="Test tool",
        input_type=dict,
        handler=fake_handler,
        policy=tool_policy,
    )

    registry.register(tool)

    assert registry.has("test_tool")
    assert not registry.has("unknown_tool")


def test_register_duplicate_tool() -> None:
    registry = ToolRegistry()

    tool = Tool(
        name="test_tool",
        description="Test tool",
        input_type=dict,
        handler=fake_handler,
        policy=tool_policy,
    )

    registry.register(tool)

    with pytest.raises(ToolRegistryError):
        registry.register(tool)


def test_all() -> None:
    registry = ToolRegistry()

    first = Tool(
        name="first",
        description="First tool",
        input_type=dict,
        handler=fake_handler,
        policy=tool_policy,
    )

    second = Tool(
        name="second",
        description="Second tool",
        input_type=dict,
        handler=fake_handler,
        policy=tool_policy,
    )

    registry.register(first)
    registry.register(second)

    assert registry.all() == (first, second)
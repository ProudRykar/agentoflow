from typing import Any

from core.entities.models.builtin.read_file import ReadFileInput
from core.entities.models.tool import Tool, ToolPolicy
from core.entities.models.tool_definition import ToolDefinition
from core.entities.models.tool_definition_builder import (
    ToolDefinitionBuilder,
)


async def fake_handler(
    arguments: Any,
    context: Any,
) -> str:
    return "ok"


def test_build_tool_definition() -> None:
    tool = Tool(
        name="read_file",
        description="Read a file from the working directory",
        input_type=ReadFileInput,
        handler=fake_handler,
        policy=ToolPolicy(
            permissions=frozenset(),
            timeout=10.0,
            max_output_size=10_000,
        ),
    )

    definition = ToolDefinitionBuilder().build(tool)

    assert definition == ToolDefinition(
        name="read_file",
        description="Read a file from the working directory",
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
            },
            "required": ["path"],
        },
    )
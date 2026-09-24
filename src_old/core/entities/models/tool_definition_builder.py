from typing import Any

from core.entities.models.schema_generator import SchemaGenerator
from core.entities.models.tool import Tool
from core.entities.models.tool_definition import ToolDefinition


class ToolDefinitionBuilder:
    def __init__(
        self,
        schema_generator: SchemaGenerator | None = None,
    ) -> None:
        self._schema_generator = schema_generator or SchemaGenerator()

    def build(
        self,
        tool: Tool[Any, Any],
    ) -> ToolDefinition:
        return ToolDefinition(
            name=tool.name,
            description=tool.description,
            input_schema=self._schema_generator.generate(
                tool.input_type
            ),
        )
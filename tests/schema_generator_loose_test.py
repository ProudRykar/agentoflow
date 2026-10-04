"""Schema generation for tools with unconstrained parameters.

An MCP tool whose JSON Schema fragment has no recognisable ``type``
is mapped onto ``typing.Any``. The generator has to describe that
instead of rejecting it: one such tool used to answer
``GET /api/tools`` with a 500 and take out the whole Tools tab.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from agent_workflow.core.entities.models.schema_generator import (
    SchemaGenerator,
)
from agent_workflow.core.entities.models.tool_definition_builder import (
    ToolDefinitionBuilder,
)
from agent_workflow.core.infrastructure.mcp.schema import (
    build_input_dataclass,
)


@dataclass
class Loose:
    anything: Any
    label: str


@dataclass
class LooseContainers:
    items: list[Any]
    mapping: dict[str, Any]


@dataclass
class LooseOptional:
    maybe: Any | None


class Weird:
    """A plain class used as an annotation."""


@dataclass
class WeirdHolder:
    value: Weird


# ======================================================================
# Any is permissive
# ======================================================================


def test_any_field_is_permissive() -> None:
    schema = SchemaGenerator().generate(Loose)

    assert schema["properties"]["anything"] == {}
    assert schema["properties"]["label"] == {"type": "string"}


def test_any_inside_containers() -> None:
    schema = SchemaGenerator().generate(LooseContainers)

    assert schema["properties"]["items"] == {
        "type": "array",
        "items": {},
    }
    assert schema["properties"]["mapping"] == {
        "type": "object",
        "additionalProperties": {},
    }


def test_optional_any_is_permissive() -> None:
    schema = SchemaGenerator().generate(LooseOptional)

    assert schema["properties"]["maybe"] == {}


def test_unknown_annotation_still_raises() -> None:
    """Permissive for Any, strict for genuinely unknown types."""

    with pytest.raises(TypeError):
        SchemaGenerator().generate(WeirdHolder)


# ======================================================================
# The real MCP path
# ======================================================================


def test_typeless_property_maps_to_any_and_still_builds() -> None:
    """A property with no ``type`` is what a real MCP server sends."""

    input_type = build_input_dataclass(
        "mcp_stash_get_performers",
        {
            "type": "object",
            "properties": {
                "query": {"description": "free text"},
                "performer_name": {"type": "string"},
            },
            "required": ["query"],
        },
    )

    schema = SchemaGenerator().generate(input_type)

    assert schema["properties"]["query"] == {}

    # An optional field is marked nullable; that is existing
    # behaviour for non-required MCP parameters.
    assert schema["properties"]["performer_name"] == {
        "type": "string",
        "nullable": True,
    }


def test_tool_definition_builds_for_a_realistic_mcp_tool() -> None:
    """Regression: this raised "Unsupported field type: typing.Any"
    and surfaced as 500 on GET /api/tools."""

    from agent_workflow.core.entities.models.tool import (
        Tool,
        ToolPolicy,
    )

    input_type = build_input_dataclass(
        "mcp_stash_search",
        {
            "type": "object",
            "properties": {
                "term": {"type": "string"},
                "limit": {"type": "integer"},
                "filters": {"type": "object"},
                "anything": {},
                "either": {"type": ["string", "integer"]},
            },
            "required": ["term"],
        },
    )

    async def handler(arguments: Any, context: Any) -> str:
        del arguments, context

        return "ok"

    definition = ToolDefinitionBuilder().build(
        Tool(
            name="mcp_stash_search",
            description="Search.",
            input_type=input_type,
            handler=handler,
            policy=ToolPolicy(
                permissions=frozenset({"mcp.execute"}),
                timeout=30.0,
                max_output_size=65_536,
            ),
        )
    )

    properties = definition.input_schema["properties"]

    assert definition.name == "mcp_stash_search"
    assert properties["term"] == {"type": "string"}
    assert properties["limit"] == {
        "type": "integer",
        "nullable": True,
    }
    assert properties["filters"]["type"] == "object"
    assert properties["anything"] == {}
    assert definition.input_schema["required"] == ["term"]
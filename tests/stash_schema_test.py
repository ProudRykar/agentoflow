"""A rejected query must not read as a broken tool.

The failure being guarded against is an agent writing a plausible
query, being refused for naming a field that does not exist, and
concluding the GraphQL tool is unusable. Almost always the field exists
on the *filter* and not the object, or the other way round, so the two
lists have to be readable side by side.
"""

from __future__ import annotations

import json

import pytest

from agent_workflow.core.entities.models.stash_schema import (
    EntitySchema,
    SchemaInput,
    StashSchemaError,
    StashSchemaReader,
    create_stash_entity_schema_tool,
    format_schema,
)


PERFORMER = [
    "id", "name", "urls", "alias_list", "favorite", "career_start",
    "career_end", "measurements", "scene_count",
]

PERFORMER_FILTER = [
    "AND", "OR", "NOT", "name", "aliases", "career_length",
    "career_start", "career_end", "measurements", "filter_favorites",
    "scene_count", "url",
]


class FakeClient:
    def __init__(self, types: dict[str, list[str]], *, truncate: bool = False) -> None:
        self.types = types
        self.truncate = truncate
        self.asked: list[str] = []
        self._policy = type("P", (), {"timeout": 30.0})()

    @property
    def policy(self):
        return self._policy

    async def execute(self, query: str, variables: str = "") -> str:
        name = json.loads(variables)["name"]

        self.asked.append(name)

        if name not in self.types:
            return json.dumps({"__type": None})

        if self.truncate:
            return (
                '{"__type": {"fields": [{"name": "id"}]}}'
                "\\n... [truncated by the runtime limit]"
            )

        return json.dumps({"__type": {"fields": [{"name": field} for field in self.types[name]]}})


async def test_it_reads_both_lists() -> None:
    reader = StashSchemaReader(FakeClient({
        "Performer": PERFORMER,
        "PerformerFilterType": PERFORMER_FILTER,
    }))

    schema = await reader.read("Performer", "PerformerFilterType")

    assert "alias_list" in schema.entity_fields
    assert "aliases" in schema.filter_fields


async def test_it_names_the_fields_that_exist_on_only_one_side() -> None:
    reader = StashSchemaReader(FakeClient({
        "Performer": PERFORMER,
        "PerformerFilterType": PERFORMER_FILTER,
    }))

    schema = await reader.read("Performer", "PerformerFilterType")

    # The trap the model falls into: career_length is filterable, not
    # selectable, so asking for both fails wholesale.
    assert "career_length" in schema.only_on_filter
    assert "career_length" not in schema.only_on_object

    assert "alias_list" in schema.only_on_object
    assert "alias_list" not in schema.only_on_filter

    assert "scene_count" in schema.shared


async def test_an_unknown_entity_is_an_error_not_an_empty_list() -> None:
    reader = StashSchemaReader(FakeClient({}))

    with pytest.raises(StashSchemaError, match="not a type Stash knows"):
        await reader.read("Nonsense")


async def test_asking_for_nothing_is_rejected() -> None:
    reader = StashSchemaReader(FakeClient({}))

    with pytest.raises(StashSchemaError, match="entity or a filter type"):
        await reader.read("")


async def test_a_truncated_schema_is_an_error() -> None:
    reader = StashSchemaReader(
        FakeClient({"Performer": PERFORMER}, truncate=True)
    )

    with pytest.raises(StashSchemaError, match="truncated"):
        await reader.read("Performer")


async def test_a_non_json_body_is_an_error() -> None:
    class Broken(FakeClient):
        async def execute(self, query: str, variables: str = "") -> str:
            return "<html>502</html>"

    with pytest.raises(StashSchemaError, match="not JSON"):
        await StashSchemaReader(Broken({})).read("Performer")


def test_rendering_shows_both_lists_and_the_warning() -> None:
    rendered = format_schema(
        EntitySchema(
            entity="Performer",
            entity_fields=tuple(PERFORMER),
            filter_name="PerformerFilterType",
            filter_fields=tuple(PERFORMER_FILTER),
        )
    )

    assert "Performer fields:" in rendered
    assert "PerformerFilterType fields:" in rendered
    assert "career_length" in rendered
    assert "fails as a whole" in rendered


def test_rendering_a_lone_object_does_not_mention_a_filter() -> None:
    rendered = format_schema(EntitySchema(entity="Scene", entity_fields=("id",)))

    assert "Scene fields:" in rendered
    assert "Filterable" not in rendered


async def test_the_tool_asks_for_the_permission_it_needs() -> None:
    tool = create_stash_entity_schema_tool(StashSchemaReader(FakeClient({})))

    assert tool.name == "stash_entity_schema"
    assert tool.policy.permissions == frozenset({"graphql.execute"})
    assert isinstance(tool.input_type, type)


async def test_the_tool_returns_the_rendering() -> None:
    reader = StashSchemaReader(FakeClient({"Performer": PERFORMER}))

    class Context:
        pass

    tool = create_stash_entity_schema_tool(reader)

    rendered = await tool.handler(
        tool.input_type(entity="Performer"),
        Context(),  # type: ignore[arg-type]
    )

    assert "Performer fields:" in rendered
    assert "alias_list" in rendered


def test_defaults_are_empty_so_a_lone_object_can_be_asked_for() -> None:
    arguments = SchemaInput(entity="Scene")

    assert arguments.filter_type == ""

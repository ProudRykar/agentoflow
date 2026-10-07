"""Ask Stash what a field is actually called.

The failure this prevents: an agent writes a plausible query, gets a
rejection, and concludes the GraphQL tool is broken. The rejections
usually come from a field that exists on the *filter* but not on the
object, or the reverse, and the two lists share only some names:

| On the object | On the filter |
| --- | --- |
| `alias_list` | `aliases` |
| `measurements` (one string) | `measurements` (a criterion) |
| `career_start`, `career_end` | `career_length`, `career_start` |
| no `scrap_count` | no `scrap_count` |

So "performer career length" is a real filter field and a missing
object field, and asking for both in one query fails wholesale. This
tool answers the question instead of leaving it to be guessed at.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from typing import Any

from agent_workflow.core.entities.models.graphql_query import (
    StashGraphQLClient,
)
from agent_workflow.core.entities.models.tool import (
    Tool,
    ToolContext,
    ToolPolicy,
)


class StashSchemaError(RuntimeError):
    """The schema could not be read."""


@dataclass(slots=True, frozen=True)
class SchemaInput:
    """Arguments for ``stash_entity_schema``."""

    entity: str = ""
    """Object name, e.g. Performer, Scene, Tag, Image, Gallery."""

    filter_type: str = ""
    """Filter input name, e.g. PerformerFilterType. Empty skips it."""


@dataclass(slots=True, frozen=True)
class EntitySchema:
    """Field names for one object and its filter."""

    entity: str
    entity_fields: tuple[str, ...] = ()
    filter_name: str = ""
    filter_fields: tuple[str, ...] = ()

    @property
    def only_on_object(self) -> tuple[str, ...]:
        return tuple(
            name
            for name in self.entity_fields
            if name not in self.filter_fields
        )

    @property
    def only_on_filter(self) -> tuple[str, ...]:
        return tuple(
            name
            for name in self.filter_fields
            if name not in self.entity_fields
        )

    @property
    def shared(self) -> tuple[str, ...]:
        both = set(self.entity_fields) & set(self.filter_fields)

        return tuple(name for name in self.entity_fields if name in both)


_QUERY = """
query($name: String!) {
  __type(name: $name) {
    kind
    name
    fields { name }
    inputFields { name }
  }
}
"""


class StashSchemaReader:
    """Reads field names out of the live schema."""

    def __init__(self, client: StashGraphQLClient) -> None:
        self._client = client

    @property
    def client(self) -> StashGraphQLClient:
        return self._client

    async def _type_fields(self, name: str) -> tuple[str, ...]:
        if not name.strip():
            return ()

        raw = await self._client.execute(
            _QUERY,
            json.dumps({"name": name.strip()}),
        )

        body = raw if isinstance(raw, str) else str(raw)

        if "[truncated by the runtime limit]" in body:
            raise StashSchemaError(
                f"the schema response for {name!r} was truncated"
            )

        try:
            payload = json.loads(raw) if isinstance(raw, str) else raw
        except json.JSONDecodeError as exc:
            raise StashSchemaError(
                "Stash returned something that is not JSON"
            ) from exc

        node = (payload or {}).get("__type")

        if not isinstance(node, dict):
            return ()

        entries = node.get("fields") or node.get("inputFields") or []

        return tuple(
            str(entry.get("name"))
            for entry in entries
            if entry.get("name")
        )

    async def read(
        self,
        entity: str,
        filter_type: str = "",
    ) -> EntitySchema:
        """Field names for an object and, optionally, its filter."""

        if not entity.strip() and not filter_type.strip():
            raise StashSchemaError("give an entity or a filter type")

        schema = EntitySchema(entity=entity.strip())

        if entity.strip():
            schema = replace(
                schema,
                entity_fields=await self._type_fields(entity),
            )

        if filter_type.strip():
            schema = replace(
                schema,
                filter_name=filter_type.strip(),
                filter_fields=await self._type_fields(filter_type),
            )

        if schema.entity and not schema.entity_fields:
            raise StashSchemaError(
                f"{entity!r} is not a type Stash knows about"
            )

        return schema


def format_schema(schema: EntitySchema) -> str:
    """List the fields, and point out where the two lists diverge."""

    lines: list[str] = []

    if schema.entity:
        lines.append(f"{schema.entity} fields:")
        lines.append(f"  {', '.join(schema.entity_fields)}")
        lines.append("")

    if schema.filter_name:
        lines.append(f"{schema.filter_name} fields:")
        lines.append(f"  {', '.join(schema.filter_fields)}")
        lines.append("")

    if schema.entity and schema.filter_name:
        only_object = schema.only_on_object
        only_filter = schema.only_on_filter

        if only_object:
            lines.append(
                "Selectable on the object, absent from the filter: "
                f"{', '.join(only_object)}"
            )

        if only_filter:
            lines.append(
                "Filterable, absent from the object: "
                f"{', '.join(only_filter)}"
            )

        lines.append("")
        lines.append(
            "A query that filters on one and selects the other in a "
            "single document fails as a whole, so check both lists "
            "before writing it."
        )

    return "\n".join(lines)


def create_stash_entity_schema_tool(reader: StashSchemaReader) -> Tool:
    """The tool to reach for instead of guessing a field name."""

    async def handler(
        arguments: SchemaInput,
        context: ToolContext,
    ) -> str:
        del context

        schema = await reader.read(
            arguments.entity,
            arguments.filter_type,
        )

        return format_schema(schema)

    return Tool(
        name="stash_entity_schema",
        description=(
            "List the fields Stash actually has for an entity and, "
            "optionally, for its filter type. Use it before writing a "
            "query whose fields you are not certain about, and "
            "whenever a query is rejected for naming a field that does "
            "not exist.\n"
            "\n"
            "The two lists usually disagree, which is why a plausible "
            "query fails: a performer has alias_list and career_start, "
            "while the filter has aliases and career_length, so asking "
            "for career_length in a selection is rejected. It also "
            "reports which names exist on only one of the two.\n"
            "\n"
            "Give entity as the object name (Performer, Scene, Tag, "
            "Image, Gallery, Studio, Group) and filter_type as the "
            "matching filter input (PerformerFilterType, "
            "SceneFilterType) when the query needs both."
        ),
        input_type=SchemaInput,
        handler=handler,
        policy=ToolPolicy(
            permissions=frozenset({
                "graphql.execute",
            }),
            timeout=reader.client.policy.timeout,
            max_output_size=40_000,
        ),
    )


__all__: list[Any] = [
    "EntitySchema",
    "SchemaInput",
    "StashSchemaError",
    "StashSchemaReader",
    "create_stash_entity_schema_tool",
    "format_schema",
]
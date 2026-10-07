"""A tool whose input contains objects of its own.

Nested dataclasses were unsupported in both halves at once, and neither
half failed usefully. The schema generator raised ``TypeError:
Unsupported field type`` from the tools endpoint, which returns 500 for
*every* tool rather than for the one that cannot be described, so the
page looked broken for an unrelated reason. The argument decoder had no
branch at all: it returned the raw dict, so a tool could be listed and
scheduled and then fail in its handler on an attribute that was never
there.

These tests pin both halves, plus the case that makes recursion a
non-obvious risk: a type that contains itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from agent_workflow.core.entities.models.arguments import (
    ArgumentDecoder,
    ArgumentDecoderError,
)
from agent_workflow.core.entities.models.schema_generator import (
    SchemaGenerator,
)


@dataclass
class Line:
    tag_id: str
    note: str = ""


@dataclass
class Patch:
    calls: list[Line]
    limit: int = 4
    dry_run: bool = False


@dataclass
class Nested:
    patch: Patch
    note: str = ""


@dataclass
class Plain:
    query: str
    limit: int = 10


@dataclass
class CarriesArguments:
    """A field named like an argument envelope."""

    arguments: dict[str, object]


@dataclass
class Item:
    arguments: dict[str, object]


@dataclass
class Batch:
    items: list[Item]


@dataclass
class Tree:
    """Self-referential, to check the cycle is broken and not followed."""

    name: str
    child: "Tree | None" = None


@dataclass
class WithTree:
    root: Tree
    others: dict[str, Tree] = field(default_factory=dict)


class TestSchema:
    def test_a_list_of_objects_is_described_in_full(self) -> None:
        schema = SchemaGenerator().generate(Patch)

        calls = schema["properties"]["calls"]

        assert calls["type"] == "array"

        items = calls["items"]

        assert items["type"] == "object"
        assert items["properties"]["tag_id"] == {"type": "string"}
        assert items["required"] == ["tag_id"]

    def test_required_still_works_alongside_optionals(self) -> None:
        schema = SchemaGenerator().generate(Patch)

        assert sorted(schema["required"]) == ["calls"]
        assert schema["properties"]["limit"]["type"] == "integer"
        assert schema["properties"]["dry_run"]["type"] == "boolean"

    def test_two_levels_deep(self) -> None:
        schema = SchemaGenerator().generate(Nested)

        inner = (
            schema["properties"]["patch"]["properties"]["calls"]["items"]
        )

        assert inner["properties"]["tag_id"] == {"type": "string"}

    def test_an_optional_object_is_nullable(self) -> None:
        schema = SchemaGenerator().generate(
            WithTree
        )

        assert schema["properties"]["root"]["type"] == "object"

    def test_a_self_referential_type_terminates(self) -> None:
        """Recursing into a cycle would never return.

        The nesting is cut with a free-form object, which keeps a
        recursive input type from taking down every tool page it is
        registered on.
        """

        schema = SchemaGenerator().generate(WithTree)

        child = schema["properties"]["root"]["properties"]["child"]

        assert child["type"] == "object"
        assert child["additionalProperties"] is True
        # Cut here, so the description is finite.
        assert "properties" in child

    def test_a_repeated_type_is_still_described_everywhere(self) -> None:
        """Cutting a cycle is not a licence to under-describe.

        ``Tree`` appears as both root and a dict value; neither is on a
        cycle back to itself, so both must be spelled out.
        """

        schema = SchemaGenerator().generate(WithTree)

        others = schema["properties"]["others"]

        assert others["additionalProperties"]["type"] == "object"
        assert others["additionalProperties"]["properties"]


class TestDecode:
    def test_objects_are_built_not_left_as_dicts(self) -> None:
        decoded = ArgumentDecoder().decode(
            {
                "calls": [
                    {"tag_id": "a"},
                    {"tag_id": "b", "note": "why"},
                ]
            },
            Patch,
        )

        assert isinstance(decoded, Patch)
        assert all(isinstance(line, Line) for line in decoded.calls)
        assert decoded.calls[0].note == ""
        assert decoded.calls[1].note == "why"

    def test_defaults_are_applied(self) -> None:
        decoded = ArgumentDecoder().decode(
            {"calls": [{"tag_id": "a"}]},
            Patch,
        )

        assert decoded.limit == 4
        assert decoded.dry_run is False

    def test_two_levels_deep(self) -> None:
        decoded = ArgumentDecoder().decode(
            {"patch": {"calls": [{"tag_id": "a"}]}},
            Nested,
        )

        assert isinstance(decoded.patch, Patch)
        assert isinstance(decoded.patch.calls[0], Line)

    def test_an_already_built_object_is_accepted(self) -> None:
        """A caller holding decoded objects is not punished for it."""

        line = Line(tag_id="a")

        decoded = ArgumentDecoder().decode(
            {"calls": [line]},
            Patch,
        )

        assert decoded.calls[0] is line

    def test_a_non_object_where_an_object_belongs_is_refused(self) -> None:
        with pytest.raises(ArgumentDecoderError):
            ArgumentDecoder().decode(
                {"calls": ["not an object"]},
                Patch,
            )

    def test_optional_object_may_be_absent(self) -> None:
        decoded = ArgumentDecoder().decode(
            {"root": {"name": "x"}},
            WithTree,
        )

        assert decoded.root.name == "x"
        assert decoded.root.child is None


class TestErrorPaths:
    """The message has to say *which* object, not just which field.

    A model that passed twenty objects needs to be told the index. "Unknown
    field 'tag_id'" leaves it guessing.
    """

    def decode_error(self, data: dict) -> str:
        with pytest.raises(ArgumentDecoderError) as caught:
            ArgumentDecoder().decode(data, Patch)

        return str(caught.value)

    def test_an_unknown_field_names_the_index(self) -> None:
        message = self.decode_error(
            {"calls": [{"tag_id": "a"}, {"oops": 1}]}
        )

        assert "calls[1].oops" in message

    def test_a_missing_field_names_the_index(self) -> None:
        assert "calls[0].tag_id" in self.decode_error(
            {"calls": [{"note": "no id"}]}
        )

    def test_a_wrong_type_names_the_index(self) -> None:
        message = self.decode_error(
            {"calls": [{"tag_id": "a"}, 7]}
        )

        assert "calls[1]" in message
        assert "Line" in message

    def test_a_dict_value_names_its_key(self) -> None:
        with pytest.raises(ArgumentDecoderError) as caught:
            ArgumentDecoder().decode(
                {"root": {"name": "r"}, "others": {"alpha": {"nope": 1}}},
                WithTree,
            )

        assert "others.alpha.nope" in str(caught.value)

    def test_a_top_level_error_names_the_field_barely(self) -> None:
        """The path must not leak into messages for flat inputs.

        A flat tool that started reporting "calls.calls[0].tag_id"
        would be a regression for every existing tool.
        """

        @dataclass
        class Flat:
            tag_id: str

        with pytest.raises(ArgumentDecoderError) as caught:
            ArgumentDecoder().decode({"nope": 1}, Flat)

        assert caught.value.field == "nope"
        assert str(caught.value).startswith("Unknown field 'nope'")


class TestArgumentEnvelopes:
    """A field may be *named* like an envelope, and must still work.

    "arguments", "properties" and "parameters" are the wrapper names
    some providers send. They are also ordinary field names. Unwrapping
    on the name alone meant a type whose single field was called
    "arguments" had that field's value thrown away, and every key
    inside it then read as unknown -- which is how a batch tool whose
    entries each carried their own arguments could never be decoded.
    """


    def test_a_field_named_arguments_keeps_its_value(self) -> None:
        decoded = ArgumentDecoder().decode(
            {"arguments": {"tag_id": "a"}},
            CarriesArguments,
        )

        assert decoded.arguments == {"tag_id": "a"}

    def test_a_real_envelope_is_still_unwrapped(self) -> None:
        for envelope in ("arguments", "properties", "parameters"):
            decoded = ArgumentDecoder().decode(
                {envelope: {"query": "x"}},
                Plain,
            )

            assert decoded.query == "x", envelope

    def test_a_nested_object_named_arguments_survives(self) -> None:
        decoded = ArgumentDecoder().decode(
            {"items": [{"arguments": {"a": 1}}, {"arguments": {"b": 2}}]},
            Batch,
        )

        assert [item.arguments for item in decoded.items] == [
            {"a": 1},
            {"b": 2},
        ]

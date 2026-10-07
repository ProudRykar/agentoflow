"""Optional MCP parameters must be omitted, not sent as null.

An MCP tool with ``sort: str = "name"`` rejects an explicit
``"sort": null`` with "None is not of type 'string'", so omitting a
parameter has to mean *not sending it*. Sending null instead made the
tool unusable and sent the agent into a retry loop it could not
escape.
"""

from __future__ import annotations

from agent_workflow.core.infrastructure.mcp.manager import _to_arguments
from agent_workflow.core.infrastructure.mcp.schema import (
    build_input_dataclass,
    remote_field_map,
)


def _arguments(
    schema: dict,
    supplied: dict,
    remote_names: list[str] | None = None,
) -> dict:
    input_type = build_input_dataclass("mcp_stash_get_tags", schema)

    instance = input_type(**supplied)

    return _to_arguments(
        instance,
        remote_field_map(schema),
    )


PAGINATED = {
    "type": "object",
    "properties": {
        "page": {"type": "integer"},
        "per_page": {"type": "integer"},
        "sort": {"type": "string"},
        "direction": {"type": "string"},
    },
    "required": [],
}


# ======================================================================
# The bug
# ======================================================================


def test_omitted_parameters_are_not_sent_as_null() -> None:
    payload = _arguments(
        PAGINATED,
        {"page": 1, "per_page": 100},
    )

    # Regression: sort and direction arrived as explicit None and the
    # server rejected the call.
    assert "sort" not in payload
    assert "direction" not in payload
    assert payload["page"] == 1
    assert payload["per_page"] == 100


def test_supplied_null_is_still_dropped() -> None:
    payload = _arguments(
        PAGINATED,
        {"page": 1, "sort": None},
    )

    assert "sort" not in payload


def test_explicit_values_survive() -> None:
    payload = _arguments(
        PAGINATED,
        {
            "page": 2,
            "per_page": 50,
            "sort": "name",
            "direction": "DESC",
        },
    )

    assert payload == {
        "page": 2,
        "per_page": 50,
        "sort": "name",
        "direction": "DESC",
    }


# ======================================================================
# Required and falsy values
# ======================================================================


def test_required_parameter_is_sent() -> None:
    schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string"},
            "limit": {"type": "integer"},
        },
        "required": ["query"],
    }

    payload = _arguments(schema, {"query": "tag", "limit": 50})

    assert payload == {"query": "tag", "limit": 50}


def test_required_none_is_dropped_too() -> None:
    # A required field the model could not fill is better omitted
    # than sent as null: the server reports a missing argument, which
    # is actionable, instead of a type error.
    schema = {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    }

    payload = _arguments(schema, {"query": None})

    assert "query" not in payload


def test_falsy_values_are_kept() -> None:
    payload = _arguments(
        PAGINATED,
        {"page": 0, "per_page": 0},
    )

    assert payload["page"] == 0
    assert payload["per_page"] == 0


def test_empty_string_is_kept() -> None:
    payload = _arguments(
        PAGINATED,
        {"sort": ""},
    )

    assert payload["sort"] == ""


def test_boolean_false_is_kept() -> None:
    schema = {
        "type": "object",
        "properties": {"favorites_only": {"type": "boolean"}},
    }

    payload = _arguments(schema, {"favorites_only": False})

    assert payload["favorites_only"] is False


# ======================================================================
# Sanitised names
# ======================================================================


def test_sanitised_names_still_omit_nulls() -> None:
    schema = {
        "type": "object",
        "properties": {
            "class-name": {"type": "string"},
            "page": {"type": "integer"},
        },
    }

    input_type = build_input_dataclass("mcp_stash_get_tags", schema)

    payload = _to_arguments(
        input_type(class_name="x", page=1),
        remote_field_map(schema),
    )

    assert payload == {"class-name": "x", "page": 1}


def test_dict_arguments_pass_through_unchanged() -> None:
    # A non-dataclass dict has no "unset" notion, so it is forwarded
    # as the caller built it.
    assert _to_arguments({"a": 1}, ["a"]) == {"a": 1}
    assert _to_arguments(None, ["a"]) == {}

# ======================================================================
# Partial calls must not shift arguments
# ======================================================================


def test_omitting_a_leading_argument_keeps_the_rest_aligned() -> None:
    """Regression: values were mapped onto names by position.

    Supplying only ``per_page`` sent it as ``page``, so the model
    asked for page 5 of the library instead of 5 items per page and
    read the empty result as "the library has nothing here".
    """

    assert _arguments(PAGINATED, {"per_page": 25}) == {
        "per_page": 25
    }


def test_omitting_a_trailing_argument_keeps_the_rest_aligned() -> None:
    assert _arguments(PAGINATED, {"page": 2}) == {"page": 2}


def test_a_middle_argument_alone_keeps_its_own_name() -> None:
    assert _arguments(PAGINATED, {"sort": "name"}) == {
        "sort": "name"
    }


def test_sorting_arguments_survive_together() -> None:
    assert _arguments(
        PAGINATED,
        {"sort": "name", "direction": "DESC"},
    ) == {"sort": "name", "direction": "DESC"}


def test_all_arguments_map_one_to_one() -> None:
    assert _arguments(
        PAGINATED,
        {
            "page": 2,
            "per_page": 10,
            "sort": "name",
            "direction": "ASC",
        },
    ) == {
        "page": 2,
        "per_page": 10,
        "sort": "name",
        "direction": "ASC",
    }


def test_order_of_supply_does_not_matter() -> None:
    assert _arguments(
        PAGINATED,
        {"direction": "DESC", "page": 3},
    ) == {"page": 3, "direction": "DESC"}


def test_sanitised_name_is_restored_alongside_others() -> None:
    schema = {
        "type": "object",
        "properties": {
            "class-name": {"type": "string"},
            "page": {"type": "integer"},
            "per_page": {"type": "integer"},
        },
    }

    assert _arguments(
        schema,
        {"class_name": "x", "per_page": 5},
    ) == {"class-name": "x", "per_page": 5}


def test_remote_field_map_covers_every_property() -> None:
    assert remote_field_map(PAGINATED) == {
        "page": "page",
        "per_page": "per_page",
        "sort": "sort",
        "direction": "direction",
    }


def test_remote_field_map_handles_a_missing_schema() -> None:
    assert remote_field_map(None) == {}
    assert remote_field_map({}) == {}

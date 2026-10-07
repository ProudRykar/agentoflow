from __future__ import annotations

from dataclasses import make_dataclass
from typing import Any


# JSON Schema scalar -> Python type. Anything richer degrades to
# ``Any`` so the existing SchemaGenerator still produces a usable
# schema and ArgumentDecoder still validates the primitives it
# knows about.
_SCALARS: dict[str, Any] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "null": type(None),
}

_CONTAINER_LIST = list[Any]
_CONTAINER_OBJECT = dict[str, Any]


def python_type_for(
    schema: dict[str, Any] | None,
) -> Any:
    """Map one JSON Schema fragment onto a Python type."""

    if not isinstance(schema, dict):
        return Any

    declared = schema.get("type")

    if isinstance(declared, list):
        # Union type: prefer the first recognised member.
        for entry in declared:
            mapped = python_type_for({"type": entry})

            if mapped is not Any:
                return mapped

        return Any

    if declared == "array":
        items = python_type_for(schema.get("items"))

        return list[items]  # type: ignore[valid-type]

    if declared == "object":
        properties = schema.get("properties")

        if isinstance(properties, dict) and properties:
            # Nested objects are passed through structurally; the
            # model sees an opaque object, which is honest about
            # what the harness validates.
            return _CONTAINER_OBJECT

        return _CONTAINER_OBJECT

    if isinstance(declared, str) and declared in _SCALARS:
        return _SCALARS[declared]

    if "enum" in schema and schema.get("enum"):
        return str

    return Any


def sanitize_field_name(
    name: str,
    taken: set[str],
) -> str:
    """Turn an arbitrary schema key into a usable Python name."""

    cleaned = "".join(
        char if char.isalnum() or char == "_" else "_"
        for char in str(name)
    )

    if not cleaned or cleaned[0].isdigit():
        cleaned = f"field_{cleaned}"

    if cleaned in ("None", "True", "False"):
        cleaned = f"{cleaned}_"

    candidate = cleaned
    suffix = 2

    while candidate in taken:
        candidate = f"{cleaned}_{suffix}"
        suffix += 1

    taken.add(candidate)

    return candidate


def build_input_dataclass(
    name: str,
    schema: dict[str, Any] | None,
) -> type[Any]:
    """Build a dataclass mirroring an MCP tool's input schema.

    The MCP tool is then indistinguishable from a builtin: the
    existing SchemaGenerator derives the JSON schema the model
    sees, and ArgumentDecoder validates incoming arguments. No
    special-casing is needed anywhere else in the harness.
    """

    schema = schema if isinstance(schema, dict) else {}

    properties = schema.get("properties")

    if not isinstance(properties, dict) or not properties:
        # No declared parameters: accept and ignore anything.
        return make_dataclass(
            _class_name(name),
            [],
            namespace={"__annotations__": {}},
        )

    required_raw = schema.get("required")

    required: set[str] = (
        set(required_raw)
        if isinstance(required_raw, list)
        else set()
    )

    from dataclasses import field as dataclass_field

    taken: set[str] = set()
    annotations: dict[str, Any] = {}
    fields: list[tuple[str, Any, Any]] = []

    for raw_name, raw_schema in properties.items():
        field_name = sanitize_field_name(raw_name, taken)
        field_type = python_type_for(
            raw_schema if isinstance(raw_schema, dict) else None
        )

        annotations[field_name] = field_type

        if raw_name in required:
            fields.append(
                (field_name, field_type, dataclass_field())
            )
        else:
            fields.append(
                (
                    field_name,
                    field_type | None
                    if field_type is not Any
                    else Any,
                    dataclass_field(default=None),
                )
            )

    return make_dataclass(
        _class_name(name),
        fields,
        namespace={"__annotations__": annotations},
    )


def _class_name(tool_name: str) -> str:
    parts = [
        part
        for part in "".join(
            char if char.isalnum() else " "
            for char in tool_name
        ).split()
        if part
    ]

    if not parts:
        return "McpToolInput"

    return "".join(
        part[:1].upper() + part[1:]
        for part in parts
    ) + "Input"


def remote_field_map(
    schema: dict[str, Any] | None,
) -> dict[str, str]:
    """Map sanitised dataclass field names back to schema names.

    Keyed by name rather than position. Field order cannot be used
    for this because only the fields the model actually supplied
    survive decoding, so a positional mapping silently shifts every
    value into the wrong parameter as soon as one optional argument
    is omitted.
    """

    if not isinstance(schema, dict):
        return {}

    properties = schema.get("properties")

    if not isinstance(properties, dict):
        return {}

    mapping: dict[str, str] = {}
    taken: set[str] = set()

    for raw_name in properties:
        mapping[
            sanitize_field_name(raw_name, taken)
        ] = str(raw_name)

    return mapping


def property_names(
    schema: dict[str, Any] | None,
) -> list[str]:
    """Original parameter names, for decoding back to MCP."""

    if not isinstance(schema, dict):
        return []

    properties = schema.get("properties")

    if not isinstance(properties, dict):
        return []

    return [str(name) for name in properties]

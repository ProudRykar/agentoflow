from dataclasses import MISSING, fields, is_dataclass
from enum import Enum
from types import UnionType
from typing import Any, Union, get_args, get_origin, get_type_hints

from agent_workflow.core.entities.models.tool import InputT


class SchemaGenerator:
    """Build a JSON schema for a tool input type.

    Structured inputs are dataclasses. Plugins are also allowed
    to declare opaque inputs (``dict``, primitives), which are
    described by a permissive schema rather than rejected.

    Nested dataclasses are described in place. A tool that takes a list
    of objects needs the shape of those objects spelled out, and the
    alternative -- flattening them away -- leaves a model guessing what
    belongs in each entry.
    """

    def generate(
        self,
        input_type: type[InputT],
    ) -> dict[str, object]:

        if not is_dataclass(input_type):
            schema = self._opaque_schema(input_type)

            if schema is None:
                # getattr, not __name__: an unparameterised builtin like
                # ``object`` has no __name__, so the error path raised
                # AttributeError while trying to describe a bad input
                # type, which says nothing about what was wrong.
                raise TypeError(
                    f"{getattr(input_type, '__name__', input_type)} "
                    "must be a dataclass, dict, or a primitive type"
                )

            return schema

        return self._object_schema(input_type, set())

    def _object_schema(
        self,
        input_type: Any,
        seen: set[Any],
    ) -> dict[str, object]:
        """Describe a dataclass as a JSON object.

        ``seen`` is the chain of dataclasses currently being described
        rather than everything described so far. A type that appears
        twice on one path is self-referential, and recursing into it
        would never terminate; one that merely repeats across sibling
        fields is ordinary and is described in full each time.
        """

        if input_type in seen:
            # A cycle cannot be spelled out in JSON Schema without
            # inventing a $ref convention nothing else here understands.
            # A free-form object is the honest description, and it
            # keeps a recursive input type from taking down every tool
            # page it is registered on.
            return {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": True,
            }

        hints = get_type_hints(input_type)

        schema: dict[str, object] = {
            "type": "object",
            "properties": {},
            "required": [],
        }

        properties = schema["properties"]
        required = schema["required"]

        nested = {*seen, input_type}

        for field in fields(input_type):
            field_type = hints[field.name]

            properties[field.name] = self._type_to_schema(
                field_type,
                nested,
            )

            if (
                field.default is MISSING
                and field.default_factory is MISSING
            ):
                required.append(field.name)

        return schema

    def _opaque_schema(
        self,
        input_type: Any,
    ) -> dict[str, object] | None:
        """Schema for a non-dataclass input type.

        ``dict`` (and ``Mapping``) become a free-form object.
        Primitives map to their JSON counterpart. Everything
        else is rejected by the caller.
        """

        if input_type is Any or input_type is object:
            return {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": True,
            }

        origin = get_origin(input_type)

        if origin in (dict,) or input_type is dict:
            return {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": True,
            }

        if input_type is str:
            return {"type": "string"}

        if input_type is bool:
            return {"type": "boolean"}

        if input_type is int:
            return {"type": "integer"}

        if input_type is float:
            return {"type": "number"}

        if origin is list:
            args = get_args(input_type)

            return {
                "type": "array",
                "items": (
                    # No nesting chain here: this is a top-level opaque
                    # type, so nothing is being described above it.
                    self._type_to_schema(args[0], set())
                    if args
                    else {"type": "object"}
                ),
            }

        if origin in (Union, UnionType):
            args = tuple(
                arg for arg in get_args(input_type)
                if arg is not type(None)
            )

            if len(args) == 1:
                return self._opaque_schema(args[0])

        if (
            isinstance(input_type, type)
            and issubclass(input_type, Enum)
        ):
            return {
                "type": "string",
                "enum": [member.value for member in input_type],
            }

        return None

    def _type_to_schema(
        self,
        field_type: type,
        seen: set[Any] | None = None,
    ) -> dict[str, object]:
        # An unconstrained value. MCP tools reach this whenever the
        # remote schema declares no recognisable type, and Any is
        # also how callers spell "pass it through". An empty schema
        # is the honest JSON Schema for "anything goes"; rejecting it
        # would let one remote tool break the tools API.
        if field_type is Any or field_type is object:
            return {}

        if field_type is type(None):
            return {"type": "null"}

        origin = get_origin(field_type)

        if origin in (Union, UnionType):
            args = get_args(field_type)

            non_none_types = tuple(
                arg for arg in args if arg is not type(None)
            )

            if len(non_none_types) == 1 and len(args) == 2:
                inner = non_none_types[0]

                # ``Any | None`` is still "anything", so an empty
                # schema describes it better than a nullable type
                # that does not exist.
                if inner is Any or inner is object:
                    return {}

                schema = self._type_to_schema(inner, seen)
                schema["nullable"] = True
                return schema

        if origin is tuple:
            args = get_args(field_type)

            if len(args) == 2 and args[1] is Ellipsis:
                return {
                    "type": "array",
                    "items": self._type_to_schema(args[0], seen),
                }

        if origin is list:
            args = get_args(field_type)

            return {
                "type": "array",
                "items": (
                    self._type_to_schema(args[0], seen)
                    if args
                    else {"type": "object"}
                ),
            }

        if origin is set:
            args = get_args(field_type)

            return {
                "type": "array",
                "items": (
                    self._type_to_schema(args[0], seen)
                    if args
                    else {"type": "object"}
                ),
                "uniqueItems": True,
            }

        if origin is dict:
            args = get_args(field_type)

            if len(args) == 2:
                return {
                    "type": "object",
                    "additionalProperties": self._type_to_schema(
                        args[1], seen
                    ),
                }

            return {
                "type": "object",
                "additionalProperties": True,
            }

        # A dataclass in a field position: describe it in place rather
        # than rejecting it. Raising here is what made a tool with a
        # list of objects unlistable, and the failure surfaced as a 500
        # on the whole tools page rather than on that one entry.
        if is_dataclass(field_type):
            return self._object_schema(
                field_type,
                seen or set(),
            )

        if field_type is str:
            return {"type": "string"}

        if field_type is int:
            return {"type": "integer"}

        if field_type is float:
            return {"type": "number"}

        if field_type is bool:
            return {"type": "boolean"}

        if isinstance(field_type, type) and issubclass(field_type, Enum):
            return {
                "type": "string",
                "enum": [member.value for member in field_type],
            }

        raise TypeError(
            f"Unsupported field type: {field_type}"
        )
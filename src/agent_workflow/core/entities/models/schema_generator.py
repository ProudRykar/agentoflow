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
    """

    def generate(
        self,
        input_type: type[InputT],
    ) -> dict[str, object]:

        if not is_dataclass(input_type):
            schema = self._opaque_schema(input_type)

            if schema is None:
                raise TypeError(
                    f"{input_type.__name__} must be a dataclass, "
                    "dict, or a primitive type"
                )

            return schema

        hints = get_type_hints(input_type)

        schema: dict[str, object] = {
            "type": "object",
            "properties": {},
            "required": [],
        }

        properties = schema["properties"]
        required = schema["required"]

        for field in fields(input_type):
            field_type = hints[field.name]

            properties[field.name] = self._type_to_schema(
                field_type
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
                    self._type_to_schema(args[0])
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
    ) -> dict[str, object]:
        origin = get_origin(field_type)

        if origin in (Union, UnionType):
            args = get_args(field_type)

            non_none_types = tuple(
                arg for arg in args if arg is not type(None)
            )

            if len(non_none_types) == 1 and len(args) == 2:
                schema = self._type_to_schema(non_none_types[0])
                schema["nullable"] = True
                return schema

        if origin is tuple:
            args = get_args(field_type)

            if len(args) == 2 and args[1] is Ellipsis:
                return {
                    "type": "array",
                    "items": self._type_to_schema(args[0]),
                }

        if origin is list:
            args = get_args(field_type)

            return {
                "type": "array",
                "items": (
                    self._type_to_schema(args[0])
                    if args
                    else {"type": "object"}
                ),
            }

        if origin is set:
            args = get_args(field_type)

            return {
                "type": "array",
                "items": (
                    self._type_to_schema(args[0])
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
                        args[1]
                    ),
                }

            return {
                "type": "object",
                "additionalProperties": True,
            }

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
from dataclasses import MISSING, fields, is_dataclass
from enum import Enum
from types import UnionType
from typing import Union, get_args, get_origin, get_type_hints

from core.entities.models.tool import InputT


class SchemaGenerator:
    def generate(
        self,
        input_type: type[InputT],
    ) -> dict[str, object]:

        if not is_dataclass(input_type):
            raise TypeError(
                f"{input_type.__name__} must be a dataclass"
            )

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
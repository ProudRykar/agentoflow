from dataclasses import MISSING, fields, is_dataclass
from typing import get_type_hints

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
    ) -> dict[str, str]:
        if field_type is str:
            return {"type": "string"}

        if field_type is int:
            return {"type": "integer"}

        if field_type is float:
            return {"type": "number"}

        if field_type is bool:
            return {"type": "boolean"}

        raise TypeError(
            f"Unsupported field type: {field_type}"
        )
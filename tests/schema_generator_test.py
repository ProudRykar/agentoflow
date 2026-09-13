from dataclasses import dataclass

import pytest

from core.entities.models.builtin.read_file import ReadFileInput
from core.entities.models.schema_generator import SchemaGenerator


@pytest.fixture
def generator() -> SchemaGenerator:
    return SchemaGenerator()


def test_generate_read_file_schema(
    generator: SchemaGenerator,
) -> None:
    schema = generator.generate(ReadFileInput)

    assert schema == {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
            },
        },
        "required": ["path"],
    }


@dataclass
class ExampleInput:
    path: str
    encoding: str = "utf-8"


def test_generate_schema_with_default(
    generator: SchemaGenerator,
) -> None:
    schema = generator.generate(ExampleInput)

    assert schema == {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
            },
            "encoding": {
                "type": "string",
            },
        },
        "required": ["path"],
    }
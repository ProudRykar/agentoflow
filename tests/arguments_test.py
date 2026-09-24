

import pytest

from core.entities.models.arguments import (
    ArgumentDecoder,
    ArgumentDecoderError,
)
from core.entities.models.builtin.read_file import ReadFileInput
from dataclasses import dataclass

@pytest.fixture
def decoder() -> ArgumentDecoder:
    return ArgumentDecoder()

@dataclass
class ExampleInput:
    path: str
    encoding: str = "utf-8"

def test_decode_valid_arguments(
    decoder: ArgumentDecoder,
) -> None:
    result = decoder.decode(
        {"path": "src/main.py"},
        ReadFileInput,
    )

    assert result == ReadFileInput(path="src/main.py")


def test_decode_invalid_type(
    decoder: ArgumentDecoder,
) -> None:
    with pytest.raises(ArgumentDecoderError) as exc_info:
        decoder.decode(
            {"path": 123},
            ReadFileInput,
        )

    error = exc_info.value

    assert error.code == "invalid_type"
    assert error.field == "path"
    assert error.expected == "str"
    assert error.actual == "int"


def test_decode_unknown_field(
    decoder: ArgumentDecoder,
) -> None:
    with pytest.raises(ArgumentDecoderError) as exc_info:
        decoder.decode(
            {
                "path": "src/main.py",
                "unknown": 123,
            },
            ReadFileInput,
        )

    error = exc_info.value

    assert error.code == "unknown_field"
    assert error.field == "unknown"
    assert "path" in str(error)


def test_decode_envelope_is_unwrapped(
    decoder: ArgumentDecoder,
) -> None:
    result = decoder.decode(
        {"properties": {"path": "src/main.py"}},
        ReadFileInput,
    )

    assert result == ReadFileInput(path="src/main.py")


def test_decode_nested_envelope_stays_error(
    decoder: ArgumentDecoder,
) -> None:
    with pytest.raises(ArgumentDecoderError) as exc_info:
        decoder.decode(
            {"properties": {"properties": {"path": "x"}}},
            ReadFileInput,
        )

    assert exc_info.value.code == "unknown_field"


def test_decode_default_field(
    decoder: ArgumentDecoder,
) -> None:
    result = decoder.decode(
        {"path": "src/main.py"},
        ExampleInput,
    )

    assert result == ExampleInput(
        path="src/main.py",
        encoding="utf-8",
    )


def test_decode_missing_field(
    decoder: ArgumentDecoder,
) -> None:
    with pytest.raises(ArgumentDecoderError) as exc_info:
        decoder.decode(
            {},
            ReadFileInput,
        )

    error = exc_info.value

    assert error.code == "missing_field"
    assert error.field == "path"
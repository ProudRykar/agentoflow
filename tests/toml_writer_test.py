from __future__ import annotations

import datetime as dt
import tomllib

import pytest

from agent_workflow.core.infrastructure.toml_writer import (
    TOMLWriteError,
    dumps,
)


def round_trip(data: dict) -> dict:
    text = dumps(data)

    return tomllib.loads(text)


def test_scalars() -> None:
    data = {
        "agent": {"max_iterations": 10},
        "llm": {
            "provider": "ollama",
            "model": "gemma4:12b",
            "timeout": 600.0,
            "thinking": True,
            "stream": False,
        },
    }

    assert round_trip(data) == data


def test_booleans_are_not_integers() -> None:
    parsed = round_trip({"flag": True, "count": 1})

    assert parsed["flag"] is True
    assert parsed["count"] == 1


def test_arrays_of_scalars() -> None:
    data = {
        "mcp": {
            "default_permissions": ["mcp.execute", "mcp.net"],
            "ports": [1, 2, 3],
        }
    }

    assert round_trip(data) == data


def test_empty_array() -> None:
    assert round_trip({"items": []}) == {"items": []}


def test_empty_table_is_preserved() -> None:
    """An empty table still emits its header, so it round-trips."""

    data = {"a": {}, "b": {"x": 1}}

    text = dumps(data)

    assert "[a]" in text
    assert tomllib.loads(text) == data


def test_nested_tables() -> None:
    data = {"a": {"b": {"c": {"d": 1}}}}

    assert round_trip(data) == data


def test_array_of_tables() -> None:
    data = {
        "models": [
            {
                "name": "qwen",
                "description": "fast",
                "capabilities": {"reasoning": 3},
                "requirements": {"context_size": 32768},
            },
            {"name": "llama", "description": "slow"},
        ]
    }

    assert round_trip(data) == data


def test_array_of_tables_with_siblings() -> None:
    data = {
        "mcp": {"enabled": True},
        "servers": [
            {"name": "a", "command": "x"},
            {"name": "b", "command": "y"},
        ],
    }

    assert round_trip(data) == data


def test_scalars_precede_subtables() -> None:
    """TOML requires this order or the document is invalid."""

    data = {
        "mcp": {
            "enabled": True,
            "servers": [{"name": "a", "command": "x"}],
        }
    }

    text = dumps(data)

    assert text.index("enabled") < text.index("[[mcp.servers]]")
    assert round_trip(data) == data


def test_inline_table() -> None:
    data = {"server": {"env": {"TOKEN": "x", "PORT": 1}}}

    parsed = round_trip(data)

    assert parsed["server"]["env"] == {"TOKEN": "x", "PORT": 1}


def test_quotes_and_escapes() -> None:
    data = {
        "text": 'he said "hi"\nnewline\ttab\\backslash',
        "unicode": "привет",
    }

    assert round_trip(data) == data


def test_control_characters_are_escaped() -> None:
    data = {"text": "bell\x07end"}

    assert round_trip(data) == data


def test_keys_needing_quotes() -> None:
    data = {"key with space": 1, "dotted.key": 2, "": 3}

    del data[""]

    parsed = round_trip(data)

    assert parsed == data


def test_empty_key_rejected() -> None:
    with pytest.raises(TOMLWriteError, match="Empty key"):
        dumps({"": 1})


def test_non_string_keys_are_stringified() -> None:
    assert round_trip({1: "a"}) == {"1": "a"}


def test_nan_rejected() -> None:
    with pytest.raises(TOMLWriteError, match="NaN"):
        dumps({"x": float("nan")})


def test_infinity_rejected() -> None:
    with pytest.raises(TOMLWriteError, match="NaN"):
        dumps({"x": float("inf")})


def test_unsupported_type_rejected() -> None:
    class Weird:
        pass

    with pytest.raises(TOMLWriteError, match="Unsupported"):
        dumps({"x": Weird()})


def test_none_rejected() -> None:
    with pytest.raises(TOMLWriteError, match="Unsupported"):
        dumps({"x": None})


def test_top_level_must_be_table() -> None:
    with pytest.raises(TOMLWriteError, match="Top level"):
        dumps([1, 2])  # type: ignore[arg-type]


def test_datetimes() -> None:
    data = {
        "when": dt.datetime(2026, 1, 2, 3, 4, 5),
        "day": dt.date(2026, 1, 2),
        "clock": dt.time(3, 4, 5),
    }

    assert round_trip(data) == data


def test_multiline_string_round_trips() -> None:
    instructions = (
        "# Title\n\n"
        "1. first step\n"
        "2. second step\n"
        "\n"
        "Some prose with 'quotes' and \"doubles\"."
    )

    parsed = round_trip({"skill": {"body": instructions}})

    assert parsed["skill"]["body"] == instructions


def test_empty_document() -> None:
    assert dumps({}) == ""


def test_output_is_deterministic() -> None:
    data = {"b": {"y": 2, "x": 1}, "a": 1}

    assert dumps(data) == dumps(data)


def test_realistic_config() -> None:
    data = {
        "agent": {"max_iterations": 10},
        "llm": {
            "provider": "ollama",
            "model": "gemma4:12b",
            "timeout": 600.0,
        },
        "context": {
            "max_messages": 24,
            "token_estimation_divisor": 4,
        },
        "memory": {"database": "memory.db"},
        "models": {
            "catalog": "models.toml",
            "vram_budget_gb": 24.0,
            "single_model_mode": False,
        },
        "mcp": {
            "enabled": True,
            "require_approval": True,
            "default_permissions": ["mcp.execute"],
            "servers": [
                {
                    "name": "fs",
                    "command": "npx",
                    "args": ["-y", "server-filesystem", "/tmp"],
                    "env": {"TOKEN": "x"},
                }
            ],
        },
    }

    assert round_trip(data) == data


def test_realistic_model_catalog() -> None:
    data = {
        "models": [
            {
                "name": "qwen3:8b",
                "description": "balanced",
                "capabilities": {
                    "reasoning": 3,
                    "coding": 4,
                    "vision": 0,
                },
                "attributes": {
                    "speed": "medium",
                    "quality": 3,
                    "resource_usage": 3,
                },
                "requirements": {
                    "context_size": 32768,
                    "vram_gb": 6.0,
                    "ram_gb": 12.0,
                    "thinking": True,
                },
            }
        ]
    }

    assert round_trip(data) == data

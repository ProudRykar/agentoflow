from __future__ import annotations

import datetime as dt
import math
from typing import Any


class TOMLWriteError(ValueError):
    """A value cannot be represented in TOML."""


_BARE_KEY_CHARS = set(
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789_-"
)

_ESCAPES = {
    "\\": "\\\\",
    '"': '\\"',
    "\b": "\\b",
    "\f": "\\f",
    "\n": "\\n",
    "\r": "\\r",
    "\t": "\\t",
}


def dumps(data: dict[str, Any]) -> str:
    """Serialise a mapping to TOML.

    Only the subset the harness actually stores is supported:
    tables, arrays of tables, strings, integers, floats, booleans,
    homogeneous arrays of scalars, and datetimes. Anything else
    raises rather than being written as a lossy approximation.

    Scalars inside a table are emitted before its sub-tables, as
    TOML requires.
    """

    if not isinstance(data, dict):
        raise TOMLWriteError("Top level must be a table")

    lines: list[str] = []

    _emit_table(data, [], lines)

    text = "\n".join(lines).strip("\n")

    return text + "\n" if text else ""


def _emit_table(
    table: dict[str, Any],
    path: list[str],
    lines: list[str],
) -> None:
    scalars: list[tuple[str, Any]] = []
    tables: list[tuple[str, dict[str, Any]]] = []
    table_arrays: list[tuple[str, list[dict[str, Any]]]] = []

    for key, value in table.items():
        name = _key(key)

        if isinstance(value, dict):
            tables.append((name, value))
        elif _is_table_array(value):
            table_arrays.append((name, value))
        else:
            scalars.append((name, value))

    for name, value in scalars:
        lines.append(f"{name} = {_format(value)}")

    for name, value in tables:
        child = [*path, name]

        if lines and lines[-1] != "":
            lines.append("")

        lines.append(f"[{'.'.join(child)}]")

        _emit_table(value, child, lines)

    for name, entries in table_arrays:
        for entry in entries:
            child = [*path, name]

            if lines and lines[-1] != "":
                lines.append("")

            lines.append(f"[[{'.'.join(child)}]]")

            _emit_table(entry, child, lines)


def _is_table_array(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) > 0
        and all(isinstance(item, dict) for item in value)
    )


def _key(name: Any) -> str:
    text = str(name)

    if not text:
        raise TOMLWriteError("Empty key")

    if all(char in _BARE_KEY_CHARS for char in text):
        return text

    return _quote(text)


def _quote(text: str) -> str:
    out = ['"']

    for char in text:
        escape = _ESCAPES.get(char)

        if escape is not None:
            out.append(escape)
        elif ord(char) < 0x20 or ord(char) == 0x7F:
            out.append(f"\\u{ord(char):04x}")
        else:
            out.append(char)

    out.append('"')

    return "".join(out)


def _format(value: Any) -> str:
    if isinstance(value, bool):
        # Must precede int: bool is a subclass of int.
        return "true" if value else "false"

    if isinstance(value, int):
        return str(value)

    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            raise TOMLWriteError(
                "NaN and Infinity have no TOML representation"
            )

        return repr(value)

    if isinstance(value, str):
        return _quote(value)

    if isinstance(value, dt.datetime):
        return value.isoformat()

    if isinstance(value, dt.date):
        return value.isoformat()

    if isinstance(value, dt.time):
        return value.isoformat()

    if isinstance(value, list):
        if not value:
            return "[]"

        rendered = [_format(item) for item in value]

        return "[" + ", ".join(rendered) + "]"

    if isinstance(value, dict):
        # Inline table, used for leaf mappings such as
        # `env = { TOKEN = "x" }`.
        pairs = [
            f"{_key(key)} = {_format(item)}"
            for key, item in value.items()
        ]

        return "{" + ", ".join(pairs) + "}"

    raise TOMLWriteError(
        f"Unsupported value type: {type(value).__name__}"
    )

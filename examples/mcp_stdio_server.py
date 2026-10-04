#!/usr/bin/env python3
"""A tiny stdio MCP server used by the test suite.

Speaks newline-delimited JSON-RPC 2.0 and implements just enough of
the protocol to exercise the client: the initialize handshake,
tools/list and tools/call.

Run directly for manual experiments::

    uv run python examples/mcp_stdio_server.py
"""

from __future__ import annotations

import json
import sys
from typing import Any


SERVER_INFO = {
    "name": "agentoflow-test-server",
    "version": "1.2.3",
}

PROTOCOL_VERSION = "2025-06-18"

TOOLS = [
    {
        "name": "echo",
        "description": "Return the text unchanged.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
            },
            "required": ["text"],
        },
    },
    {
        "name": "add",
        "description": "Add two integers.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "a": {"type": "integer"},
                "b": {"type": "integer"},
            },
            "required": ["a", "b"],
        },
    },
    {
        "name": "explode",
        "description": "Always fail, to exercise error handling.",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
]


def reply(request_id: Any, result: Any) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "result": result,
    }


def error(
    request_id: Any,
    code: int,
    message: str,
) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {"code": code, "message": message},
    }


def text_block(text: str) -> dict[str, Any]:
    return {"type": "text", "text": text}


def handle(
    message: dict[str, Any],
) -> dict[str, Any] | None:
    method = message.get("method")
    request_id = message.get("id")

    if request_id is None:
        # Notification: nothing to answer.
        return None

    if method == "initialize":
        return reply(
            request_id,
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "serverInfo": SERVER_INFO,
                "instructions": "Test server for agentoflow.",
            },
        )

    if method == "tools/list":
        return reply(request_id, {"tools": TOOLS})

    if method == "tools/call":
        params = message.get("params", {})
        name = params.get("name")
        arguments = params.get("arguments", {})

        if name == "echo":
            return reply(
                request_id,
                {
                    "content": [
                        text_block(str(arguments.get("text", ""))),
                    ],
                },
            )

        if name == "add":
            try:
                total = int(arguments.get("a", 0)) + int(
                    arguments.get("b", 0)
                )
            except (TypeError, ValueError):
                return reply(
                    request_id,
                    {
                        "isError": True,
                        "content": [
                            text_block("a and b must be integers"),
                        ],
                    },
                )

            return reply(
                request_id,
                {"content": [text_block(str(total))]},
            )

        if name == "explode":
            return reply(
                request_id,
                {
                    "isError": True,
                    "content": [text_block("deliberate failure")],
                },
            )

        return error(
            request_id,
            -32602,
            f"Unknown tool: {name}",
        )

    return error(
        request_id,
        -32601,
        f"Method not found: {method}",
    )


def main() -> None:
    for line in sys.stdin:
        text = line.strip()

        if not text:
            continue

        try:
            message = json.loads(text)
        except json.JSONDecodeError:
            continue

        if not isinstance(message, dict):
            continue

        response = handle(message)

        if response is None:
            continue

        sys.stdout.write(
            json.dumps(response, separators=(",", ":")) + "\n"
        )
        sys.stdout.flush()


if __name__ == "__main__":
    main()

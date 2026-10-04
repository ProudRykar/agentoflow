#!/usr/bin/env python3
"""A Streamable HTTP MCP server for the test suite.

Speaks the request/response half of the MCP HTTP transport: JSON-RPC
POSTed to ``/mcp``, answered either with JSON or with an SSE stream
when the client asks for one.

Run directly for manual experiments::

    uv run python examples/mcp_http_server.py
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


PROTOCOL_VERSION = "2025-06-18"

SESSION_ID = "test-session-1"


TOOLS = [
    {
        "name": "echo",
        "description": "Return the text unchanged.",
        "inputSchema": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
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
]


def text_block(text: str) -> dict[str, Any]:
    return {"type": "text", "text": text}


def result(request_id: Any, payload: Any) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "result": payload,
    }


def handle(message: dict[str, Any]) -> dict[str, Any] | None:
    method = message.get("method")
    request_id = message.get("id")

    if request_id is None:
        return None

    if method == "initialize":
        return result(
            request_id,
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "serverInfo": {
                    "name": "agentoflow-http-test",
                    "version": "2.0.0",
                },
                "instructions": "HTTP test server.",
            },
        )

    if method == "tools/list":
        return result(request_id, {"tools": TOOLS})

    if method == "tools/call":
        params = message.get("params", {})
        name = params.get("name")
        arguments = params.get("arguments", {})

        if name == "echo":
            return result(
                request_id,
                {
                    "content": [
                        text_block(str(arguments.get("text", ""))),
                    ],
                },
            )

        if name == "add":
            return result(
                request_id,
                {
                    "content": [
                        text_block(
                            str(
                                int(arguments.get("a", 0))
                                + int(arguments.get("b", 0))
                            )
                        ),
                    ],
                },
            )

        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {
                "code": -32602,
                "message": f"Unknown tool: {name}",
            },
        }

    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {
            "code": -32601,
            "message": f"Method not found: {method}",
        },
    }


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_: object) -> None:
        return

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))

        raw = self.rfile.read(length) if length else b"{}"

        try:
            message = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError:
            self._json(
                {
                    "jsonrpc": "2.0",
                    "error": {"code": -32700,
                              "message": "Parse error"},
                },
                status=400,
            )
            return

        if not isinstance(message, dict):
            self._json(
                {
                    "jsonrpc": "2.0",
                    "error": {
                        "code": -32600,
                        "message": "Invalid request",
                    },
                },
                status=400,
            )
            return

        response = handle(message)

        wants_sse = "text/event-stream" in self.headers.get(
            "Accept",
            "",
        )

        if response is None:
            # Notification: acknowledge with 202 and no body.
            self.send_response(202)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return

        if wants_sse:
            body = (
                ": ping\n\n"
                "event: message\n"
                f"data: {json.dumps(response)}\n\n"
            ).encode("utf-8")

            self.send_response(200)
            self.send_header(
                "Content-Type",
                "text/event-stream; charset=utf-8",
            )
        else:
            body = json.dumps(response).encode("utf-8")

            self.send_response(200)
            self.send_header(
                "Content-Type",
                "application/json",
            )

        self.send_header("Mcp-Session-Id", SESSION_ID)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        self._json(
            {
                "name": "agentoflow-http-test",
                "version": "2.0.0",
                "protocolVersion": PROTOCOL_VERSION,
                "tools": [tool["name"] for tool in TOOLS],
            }
        )

    def _json(
        self,
        payload: Any,
        status: int = 200,
    ) -> None:
        body = json.dumps(payload).encode("utf-8")

        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    import sys

    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8931

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)

    print(f"MCP HTTP test server on http://127.0.0.1:{port}/mcp")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

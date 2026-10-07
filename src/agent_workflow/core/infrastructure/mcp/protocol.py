from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from agent_workflow.core.entities.models.approval import (
    ApprovalRequested,
    ApprovalResolved,
)
from agent_workflow.core.entities.models.agent_trace import AgentEvent
from agent_workflow.core.entities.models.tool import (
    ToolError,
    ToolResult,
)
from agent_workflow.core.application.events import DomainEvent


JSONRPC_VERSION = "2.0"

PROTOCOL_VERSION = "2025-06-18"

CLIENT_NAME = "agentoflow"

CLIENT_VERSION = "0.1.0"


class MCPError(Exception):
    """Base error for the MCP integration."""


class MCPTransportError(MCPError):
    """The server process failed or spoke an unexpected dialect."""


class MCPProtocolError(MCPError):
    """The server answered with a malformed or error payload."""


class MCPToolError(MCPError):
    """The server reported that a tool call failed."""


@dataclass(slots=True, frozen=True)
class Request:
    method: str
    params: dict[str, Any] = field(default_factory=dict)
    id: int | None = None

    def to_message(self) -> dict[str, Any]:
        message: dict[str, Any] = {
            "jsonrpc": JSONRPC_VERSION,
            "method": self.method,
        }

        if self.params:
            message["params"] = self.params

        # A notification has no id and expects no response.
        if self.id is not None:
            message["id"] = self.id

        return message

    def encode(self) -> bytes:
        # MCP stdio framing is newline-delimited JSON.
        return (
            json.dumps(
                self.to_message(),
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")


@dataclass(slots=True, frozen=True)
class Notification:
    method: str
    params: dict[str, Any] = field(default_factory=dict)

    def encode(self) -> bytes:
        return (
            json.dumps(
                {
                    "jsonrpc": JSONRPC_VERSION,
                    "method": self.method,
                    "params": self.params,
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")


@dataclass(slots=True, frozen=True)
class ServerInfo:
    name: str
    version: str = ""
    protocol_version: str = ""
    instructions: str = ""


@dataclass(slots=True, frozen=True)
class ToolDescriptor:
    """A tool as advertised by an MCP server."""

    name: str
    description: str
    input_schema: dict[str, Any] = field(default_factory=dict)


def parse_response(
    line: bytes | str,
) -> dict[str, Any]:
    """Decode one framed server message."""

    text = line.decode("utf-8") if isinstance(line, bytes) else line

    text = text.strip()

    if not text:
        raise MCPTransportError("Empty response line")

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise MCPTransportError(
            f"Server sent invalid JSON: {text[:200]!r}"
        ) from exc

    if not isinstance(payload, dict):
        raise MCPTransportError(
            "Server response must be a JSON object",
        )

    return payload


def raise_for_error(payload: dict[str, Any]) -> None:
    error = payload.get("error")

    if error is None:
        return

    if not isinstance(error, dict):
        raise MCPProtocolError(f"Malformed error: {error!r}")

    code = error.get("code", -1)
    message = str(error.get("message", "unknown error"))

    raise MCPProtocolError(f"[{code}] {message}")


def tool_result_text(
    result: dict[str, Any],
) -> str:
    """Flatten an MCP ``tools/call`` result into text.

    Takes the ``result`` object of a JSON-RPC response, which is
    what ``MCPClient.call_tool`` returns. MCP answers with a list of
    content blocks; only text is meaningful to the model, so other
    block kinds are described rather than silently dropped.
    """

    if not isinstance(result, dict):
        return ""

    if result.get("isError"):
        parts = [
            str(block.get("text", ""))
            for block in result.get("content", [])
            if isinstance(block, dict)
        ]

        raise MCPToolError("\n".join(part for part in parts if part))

    blocks = result.get("content", [])

    if not isinstance(blocks, list):
        return ""

    parts: list[str] = []

    for block in blocks:
        if not isinstance(block, dict):
            continue

        kind = block.get("type")

        if kind == "text":
            parts.append(str(block.get("text", "")))
        elif kind in ("image", "audio"):
            parts.append(f"[{kind} omitted]")
        elif kind == "resource":
            parts.append("[resource omitted]")
        else:
            parts.append(f"[{kind or 'unknown'} block omitted]")

    return "\n".join(part for part in parts if part)


def tool_error(
    message: str,
    code: str = "mcp_error",
    retryable: bool = True,
) -> ToolResult:
    return ToolResult(
        error=ToolError(
            message=message,
            code=code,
            retryable=retryable,
        ),
    )


MCPEvent = (
    AgentEvent
    | ApprovalRequested
    | ApprovalResolved
)


__all__ = [
    "CLIENT_NAME",
    "CLIENT_VERSION",
    "DomainEvent",
    "JSONRPC_VERSION",
    "MCPError",
    "MCPEvent",
    "MCPProtocolError",
    "MCPTransportError",
    "MCPToolError",
    "Notification",
    "PROTOCOL_VERSION",
    "Request",
    "ServerInfo",
    "ToolDescriptor",
    "parse_response",
    "raise_for_error",
    "tool_error",
    "tool_result_text",
]

from __future__ import annotations

import base64
import datetime as dt
import enum
import json
import uuid
from dataclasses import fields, is_dataclass
from pathlib import Path, PurePath
from typing import Any

from agent_workflow.core.application.events import DomainEvent
from agent_workflow.core.application.session import RunFailed
from agent_workflow.core.entities.models.agent_trace import (
    AgentFinished,
    AgentPhaseChanged,
    AgentStarted,
    PlanUpdated,
    LLMContentChunk,
    LLMRequested,
    LLMResponded,
    LLMThinkingChunk,
    ToolFinished,
    ToolStarted,
)
from agent_workflow.core.entities.models.approval import (
    ApprovalRequested,
    ApprovalResolved,
)


# Stable wire type names. The frontend switches on these, so they
# are part of the contract and must not be derived from class
# names at runtime.
EVENT_TYPES: dict[type, str] = {
    AgentStarted: "agent.started",
    AgentPhaseChanged: "agent.phase_changed",
    PlanUpdated: "plan.updated",
    LLMRequested: "llm.requested",
    LLMThinkingChunk: "llm.thinking_chunk",
    LLMContentChunk: "llm.content_chunk",
    LLMResponded: "llm.responded",
    ToolStarted: "tool.started",
    ToolFinished: "tool.finished",
    AgentFinished: "agent.finished",
    ApprovalRequested: "approval.requested",
    ApprovalResolved: "approval.resolved",
    RunFailed: "run.failed",
}

UNKNOWN_EVENT_TYPE = "unknown"


def event_type(event: DomainEvent) -> str:
    return EVENT_TYPES.get(type(event), UNKNOWN_EVENT_TYPE)


def to_jsonable(value: Any) -> Any:
    """Convert an arbitrary core value into JSON-safe data.

    Handles dataclasses (including nested), enums, paths,
    UUIDs, datetimes, sets/frozensets, bytes and exceptions.
    Anything unrecognised degrades to its ``repr`` rather than
    failing the stream, because a tool returning an exotic
    object must not break event delivery.
    """

    if value is None or isinstance(value, str | bool | int):
        return value

    if isinstance(value, float):
        # NaN/Infinity are not valid JSON.
        if value != value or value in (
            float("inf"),
            float("-inf"),
        ):
            return str(value)

        return value

    if isinstance(value, enum.Enum):
        return to_jsonable(value.value)

    if isinstance(value, dict):
        return {
            str(key): to_jsonable(item)
            for key, item in value.items()
        }

    if isinstance(value, list | tuple):
        return [to_jsonable(item) for item in value]

    if isinstance(value, set | frozenset):
        return [to_jsonable(item) for item in sorted(value, key=repr)]

    if isinstance(value, bytes | bytearray):
        return base64.b64encode(bytes(value)).decode("ascii")

    if isinstance(value, PurePath):
        return str(value)

    if isinstance(value, uuid.UUID):
        return str(value)

    if isinstance(value, dt.datetime | dt.date | dt.time):
        return value.isoformat()

    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: to_jsonable(getattr(value, field.name))
            for field in fields(value)
        }

    if isinstance(value, BaseException):
        return {
            "error": type(value).__name__,
            "message": str(value),
        }

    if callable(value):
        return getattr(value, "__name__", "callable")

    return repr(value)


def event_payload(event: DomainEvent) -> dict[str, Any]:
    """Event-specific payload for the ``data`` field."""

    data = to_jsonable(event)

    if not isinstance(data, dict):
        return {"value": data}

    return data


def to_wire(
    event: DomainEvent,
    session_id: str,
    seq: int,
    timestamp: str | None = None,
) -> dict[str, Any]:
    """Build the full wire envelope for one event.

    Shape::

        {
            "type": "tool.started",
            "session_id": "...",
            "seq": 12,
            "timestamp": "2026-01-01T00:00:00+00:00",
            "data": {...}
        }
    """

    kind = event_type(event)

    envelope: dict[str, Any] = {
        "type": kind,
        "session_id": session_id,
        "seq": seq,
        "timestamp": (
            timestamp
            or dt.datetime.now(dt.UTC).isoformat()
        ),
        "data": event_payload(event),
    }

    run_id = getattr(event, "run_id", None)

    if isinstance(run_id, str) and run_id:
        envelope["run_id"] = run_id

    parent_run_id = getattr(event, "parent_run_id", None)

    if isinstance(parent_run_id, str) and parent_run_id:
        envelope["parent_run_id"] = parent_run_id

    return envelope


def dumps(
    event: DomainEvent,
    session_id: str,
    seq: int,
    timestamp: str | None = None,
) -> str:
    return json.dumps(
        to_wire(
            event,
            session_id=session_id,
            seq=seq,
            timestamp=timestamp,
        ),
        ensure_ascii=False,
        separators=(",", ":"),
    )

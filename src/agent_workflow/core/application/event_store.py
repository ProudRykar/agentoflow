"""Durable event log for session transcripts.

The frontend renders a chat purely from the event stream, and the
WebSocket replays that stream on reconnect. Persisting the events
therefore restores the transcript with no protocol change and no
second read path: reopening a session re-seeds its bus with what was
recorded, and the client replays from its cursor exactly as it does
after a network drop.

Only the events are stored. Runtimes, tool registries, MCP clients
and plugin state are rebuilt on demand instead.
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import fields, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from agent_workflow.core.application.events import DomainEvent, StoredEvent
from agent_workflow.core.entities.models.agent_phase import AgentPhase
from agent_workflow.core.entities.models.plan_step import PlanStepStatus
from agent_workflow.core.entities.models.agent_trace import (
    AgentEvent,
    AgentFinished,
    AgentPhaseChanged,
    PlanUpdated,
    AgentStarted,
    LLMContentChunk,
    LLMRequested,
    LLMResponded,
    LLMThinkingChunk,
    RunFailed,
    ToolFinished,
    ToolStarted,
)
from agent_workflow.core.entities.models.approval import (
    ApprovalRequested,
    ApprovalResolved,
)


EVENTS_DATABASE = "session-events.db"

# Streaming a response produces a lot of chunks, so a busy session
# reaches tens of thousands of rows. The log is append-only and SQLite
# reads it fine, so nothing is trimmed by default: dropping the head
# silently destroyed the opening of any long conversation, including
# the `agent.started` events the UI turns back into user messages.
# Trimming remains available for a caller that genuinely wants a bound.
DEFAULT_KEEP_EVENTS: int | None = None

# A restored tool result re-enters the context on every later turn,
# so the excerpt stays small even though the transcript keeps the
# full output.
MAX_RESTORED_TOOL_CHARS = 2_000


# The registry is explicit rather than derived from __subclasses__:
# an event whose payload cannot be rebuilt is better skipped on
# replay than crashing the session it belongs to.
_REGISTRY: dict[str, type[Any]] = {
    "AgentStarted": AgentStarted,
    "AgentPhaseChanged": AgentPhaseChanged,
    "PlanUpdated": PlanUpdated,
    "LLMRequested": LLMRequested,
    "LLMThinkingChunk": LLMThinkingChunk,
    "LLMContentChunk": LLMContentChunk,
    "LLMResponded": LLMResponded,
    "ToolStarted": ToolStarted,
    "ToolFinished": ToolFinished,
    "AgentFinished": AgentFinished,
    "ApprovalRequested": ApprovalRequested,
    "ApprovalResolved": ApprovalResolved,
    "RunFailed": RunFailed,
}

_ENUMS: dict[str, type[Enum]] = {
    "AgentPhase": AgentPhase,
    # PlanUpdated carries step status and phase as plain strings, so a
    # plan event survives the round trip through storage without the
    # encoder needing to know these names.
    "PlanStepStatus": PlanStepStatus,
}


SCHEMA = """
CREATE TABLE IF NOT EXISTS session_events (
    session_id TEXT NOT NULL,
    seq        INTEGER NOT NULL,
    type       TEXT NOT NULL,
    payload    TEXT NOT NULL,
    created_at REAL NOT NULL,
    PRIMARY KEY (session_id, seq)
);

CREATE INDEX IF NOT EXISTS session_events_created
    ON session_events (session_id, created_at);
"""


def _encode(value: Any) -> Any:
    """Convert a field value into JSON-safe primitives."""

    if isinstance(value, Enum):
        return {
            "__enum__": value.__class__.__name__,
            "value": value.value,
        }

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    if is_dataclass(value) and not isinstance(value, type):
        return {
            "__dataclass__": value.__class__.__name__,
            "fields": {
                item.name: _encode(getattr(value, item.name))
                for item in fields(value)
            },
        }

    if isinstance(value, dict):
        return {str(key): _encode(item) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        return [_encode(item) for item in value]

    # Anything else is described, not rejected: a chat transcript is
    # more valuable than strict typing of a debug field.
    return str(value)


def _decode(value: Any) -> Any:
    if isinstance(value, dict):
        if "__enum__" in value:
            enum_type = _ENUMS.get(value["__enum__"])

            if enum_type is None:
                return value["value"]

            return enum_type(value["value"])

        if "__dataclass__" in value:
            nested = _REGISTRY.get(value["__dataclass__"])

            if nested is not None:
                try:
                    return nested(
                        **{
                            name: _decode(item)
                            for name, item in value["fields"].items()
                        }
                    )
                except Exception:
                    return value["fields"]

            return {
                name: _decode(item)
                for name, item in value["fields"].items()
            }

        return {
            str(key): _decode(item)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [_decode(item) for item in value]

    return value


def encode_event(event: DomainEvent) -> str:
    """Serialise one event.

    The event is itself the dataclass, so its fields are encoded
    directly; the type name is what makes the payload rebuildable
    after a restart.
    """

    return json.dumps(
        {
            "type": type(event).__name__,
            "fields": {
                item.name: _encode(getattr(event, item.name))
                for item in fields(event)
            },
        },
        separators=(",", ":"),
    )


def decode_event(raw: str) -> DomainEvent | None:
    """Rebuild an event, or ``None`` when it cannot be trusted."""

    try:
        payload = json.loads(raw)
        event_type = _REGISTRY.get(payload["type"])
    except (json.JSONDecodeError, KeyError, TypeError):
        return None

    if event_type is None:
        return None

    known = {item.name for item in fields(event_type)}

    try:
        return event_type(
            **{
                name: _decode(value)
                for name, value in payload["fields"].items()
                if name in known
            }
        )
    except Exception:
        # A payload that no longer matches its dataclass is skipped
        # rather than allowed to break the replay of everything
        # after it.
        return None


class EventStore:
    """SQLite-backed event log, one row per published event."""

    def __init__(
        self,
        path: Path,
        *,
        keep_events: int | None = DEFAULT_KEEP_EVENTS,
    ) -> None:
        self._path = path
        self._keep = max(keep_events, 1) if keep_events else None

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._path)

        connection.execute("PRAGMA journal_mode=WAL")
        connection.executescript(SCHEMA)

        return connection

    def append(
        self,
        session_id: str,
        events: list[StoredEvent],
    ) -> None:
        """Store a batch of events and trim the session's tail."""

        if not events:
            return

        rows = [
            (
                session_id,
                stored.seq,
                type(stored.event).__name__,
                encode_event(stored.event),
                time.time(),
            )
            for stored in events
        ]

        with self._connect() as connection:
            connection.executemany(
                """
                INSERT OR REPLACE INTO session_events
                    (session_id, seq, type, payload, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                rows,
            )

            if self._keep is not None:
                connection.execute(
                    """
                    DELETE FROM session_events
                    WHERE session_id = ?
                      AND seq < (
                          SELECT seq FROM session_events
                          WHERE session_id = ?
                          ORDER BY seq DESC
                          LIMIT 1 OFFSET ?
                      )
                    """,
                    (session_id, session_id, self._keep - 1),
                )

    def load(
        self,
        session_id: str,
        *,
        after_seq: int = 0,
    ) -> list[StoredEvent]:
        rows = self._connect().execute(
            """
            SELECT seq, payload FROM session_events
            WHERE session_id = ? AND seq > ?
            ORDER BY seq ASC
            """,
            (session_id, after_seq),
        ).fetchall()

        stored: list[StoredEvent] = []

        for seq, payload in rows:
            event = decode_event(payload)

            if event is None:
                continue

            stored.append(StoredEvent(seq=seq, event=event))

        return stored

    def oldest_seq(self, session_id: str) -> int:
        """The first sequence still on disk, 0 when the session is empty.

        Reported to the client instead of a hardcoded 1, so a trimmed
        log is visible as such rather than passing for a whole one.
        """

        row = self._connect().execute(
            "SELECT COALESCE(MIN(seq), 0) FROM session_events"
            " WHERE session_id = ?",
            (session_id,),
        ).fetchone()

        return int(row[0]) if row else 0

    def latest_seq(self, session_id: str) -> int:
        row = self._connect().execute(
            "SELECT COALESCE(MAX(seq), 0) FROM session_events"
            " WHERE session_id = ?",
            (session_id,),
        ).fetchone()

        return int(row[0]) if row else 0

    def count(self, session_id: str) -> int:
        row = self._connect().execute(
            "SELECT COUNT(*) FROM session_events WHERE session_id = ?",
            (session_id,),
        ).fetchone()

        return int(row[0]) if row else 0

    def forget(self, session_id: str) -> None:
        """Drop a session's transcript, used when it is deleted."""

        with self._connect() as connection:
            connection.execute(
                "DELETE FROM session_events WHERE session_id = ?",
                (session_id,),
            )

    def session_ids(self) -> list[str]:
        rows = self._connect().execute(
            "SELECT DISTINCT session_id FROM session_events"
            " ORDER BY session_id"
        ).fetchall()

        return [row[0] for row in rows]


def dialogue_from_events(
    events: list[StoredEvent],
) -> list[dict[str, Any]]:
    """Rebuild the conversation window from a stored transcript.

    Assistant turns keep their ``tool_calls`` and the matching ``tool``
    results, so a restored window is still a valid request. Dropping
    them flattened the conversation into a bare user/assistant
    alternation and lost everything the model had read.

    Tool results are truncated to a bounded excerpt: a restored window
    is context the model will re-read on every turn, and a single
    100 KB file dump would dominate it.

    An assistant whose tool results were partly evicted keeps only the
    calls it can still answer, so the pairing invariant holds.
    """

    dialogue: list[dict[str, Any]] = []

    index = 0

    while index < len(events):
        event = events[index].event

        if isinstance(event, AgentStarted):
            dialogue.append(
                {
                    "role": "user",
                    "content": event.prompt,
                }
            )

            index += 1
            continue

        if isinstance(event, LLMResponded) and event.tool_call_count:
            turn = _tool_turn(events, index)

            dialogue.append(turn["assistant"])
            dialogue.extend(turn["results"])

            index = _skip_turn(events, index)

            continue

        if isinstance(event, AgentFinished):
            dialogue.append(
                {
                    "role": "assistant",
                    "content": event.result,
                }
            )

        index += 1

    return _drop_orphan_tools(dialogue)


def _tool_turn(
    events: list[StoredEvent],
    index: int,
) -> dict[str, Any]:
    """The assistant message of a tool turn and its tool results."""

    event = events[index].event

    content = event.content or ""

    calls: list[dict[str, Any]] = []
    results: dict[str, dict[str, Any]] = {}

    offset = index + 1

    while offset < len(events):
        candidate = events[offset].event

        if isinstance(candidate, ToolStarted):
            calls.append(
                {
                    "id": candidate.tool_call_id,
                    "type": "function",
                    "function": {
                        "name": candidate.tool_name,
                        "arguments": json.dumps(
                            candidate.arguments or {},
                        ),
                    },
                }
            )

        elif isinstance(candidate, ToolFinished):
            results[candidate.tool_call_id] = _tool_message(candidate)

        elif isinstance(
            candidate,
            (AgentStarted, AgentFinished, LLMResponded),
        ):
            break

        offset += 1

    # Keep only calls whose result survived, so the request stays
    # valid for the provider.
    if calls:
        calls = [
            call
            for call in calls
            if call["id"] in results
        ]

    message: dict[str, Any] = {
        "role": "assistant",
        "content": content,
    }

    if calls:
        message["tool_calls"] = calls

    # In call order, so the sequence reads the way it happened.
    ordered = [
        results[call["id"]]
        for call in calls
        if call["id"] in results
    ]

    return {
        "assistant": message,
        "results": ordered,
    }


def _tool_message(event: ToolFinished) -> dict[str, Any]:
    body = event.output or ""

    if event.error_code or event.error_message:
        body = (
            f"Error {event.error_code or 'tool'}: "
            f"{event.error_message or 'failed'}"
        )

    if len(body) > MAX_RESTORED_TOOL_CHARS:
        body = (
            body[:MAX_RESTORED_TOOL_CHARS]
            + "\n[tool output truncated on restore]"
        )

    return {
        "role": "tool",
        "tool_call_id": event.tool_call_id,
        "content": body,
    }


def _skip_turn(
    events: list[StoredEvent],
    index: int,
) -> int:
    """Advance past the events that belong to one assistant turn."""

    offset = index + 1

    while offset < len(events):
        candidate = events[offset].event

        if isinstance(
            candidate,
            (AgentStarted, AgentFinished, LLMResponded),
        ):
            break

        offset += 1

    return offset


def _drop_orphan_tools(
    dialogue: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Enforce the assistant/tool pairing invariant."""

    answered: set[str] = set()

    for message in dialogue:
        if message.get("role") == "assistant":
            for call in message.get("tool_calls") or []:
                call_id = call.get("id")

                if isinstance(call_id, str):
                    answered.add(call_id)

    return [
        message
        for message in dialogue
        if message.get("role") != "tool"
        or message.get("tool_call_id") in answered
    ]

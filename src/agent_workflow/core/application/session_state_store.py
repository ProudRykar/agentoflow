"""Durable working state: anchor, checkpoints and evidence.

A restored session used to come back with its transcript but no
working state: ``[TASK ANCHOR]``, ``[CHECKPOINT]``, ``[EVIDENCE]``
and ``[HISTORY]`` were all rebuilt empty, and ``continue_run`` minted
a brand-new task id, orphaning the restored dialogue from the task it
belonged to.

All three structures already had ``to_dict`` / ``from_dict`` with no
production caller. This store is that missing caller.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any

from agent_workflow.core.context.checkpoint import TaskCheckpoint
from agent_workflow.core.context.evidence import EvidenceStore
from agent_workflow.core.context.task_anchor import TaskAnchor
from agent_workflow.core.context.task_state import TaskState


STATE_DATABASE = "session-state.db"

# Evidence pages keep their full body in the transcript already, so
# the working copy is capped. Pinned items survive the cap.
MAX_RESTORED_EVIDENCE = 200


SCHEMA = """
CREATE TABLE IF NOT EXISTS session_state (
    session_id TEXT PRIMARY KEY,
    payload    TEXT NOT NULL,
    updated_at REAL NOT NULL
);
"""


def _dump(value: Any) -> str:
    return json.dumps(value, separators=(",", ":"), default=str)


def _load(raw: str) -> dict[str, Any]:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return {}

    return payload if isinstance(payload, dict) else {}


class SessionStateStore:
    """Per-session working state, one JSON blob per session."""

    def __init__(self, path: Path) -> None:
        self._path = path

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._path)

        connection.execute("PRAGMA journal_mode=WAL")
        connection.executescript(SCHEMA)

        return connection

    def save(
        self,
        session_id: str,
        *,
        anchor: TaskAnchor | None = None,
        checkpoint: TaskCheckpoint | None = None,
        task_state: TaskState | None = None,
        evidence: EvidenceStore | None = None,
    ) -> None:
        """Write the working state, merging with what is stored.

        ``None`` means "leave as-is" so a caller holding only an
        anchor does not erase a checkpoint it knows nothing about.
        """

        payload = self.load(session_id)

        if anchor is not None:
            payload["anchor"] = anchor.to_dict()

        if task_state is not None:
            payload["task_state"] = _task_state_to_dict(task_state)

        if checkpoint is not None:
            payload["checkpoint"] = checkpoint.to_dict()

        if evidence is not None:
            payload["evidence"] = evidence.to_dict()

        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO session_state
                    (session_id, payload, updated_at)
                VALUES (?, ?, ?)
                """,
                (session_id, _dump(payload), time.time()),
            )

    def load(
        self,
        session_id: str,
    ) -> dict[str, Any]:
        row = self._connect().execute(
            "SELECT payload FROM session_state WHERE session_id = ?",
            (session_id,),
        ).fetchone()

        if row is None:
            return {}

        return _load(row[0])

    def anchor(
        self,
        session_id: str,
    ) -> TaskAnchor | None:
        raw = self.load(session_id).get("anchor")

        if not isinstance(raw, dict):
            return None

        try:
            return TaskAnchor.from_dict(raw)
        except Exception:
            return None

    def checkpoint(
        self,
        session_id: str,
    ) -> TaskCheckpoint | None:
        raw = self.load(session_id).get("checkpoint")

        if not isinstance(raw, dict):
            return None

        try:
            return TaskCheckpoint.from_dict(raw)
        except Exception:
            return None

    def evidence(
        self,
        session_id: str,
    ) -> EvidenceStore | None:
        raw = self.load(session_id).get("evidence")

        if not isinstance(raw, dict):
            return None

        try:
            store = EvidenceStore.from_dict(raw)
        except Exception:
            return None

        store.trim_to(MAX_RESTORED_EVIDENCE)

        return store

    def forget(self, session_id: str) -> None:
        """Drop a deleted session's working state."""

        with self._connect() as connection:
            connection.execute(
                "DELETE FROM session_state WHERE session_id = ?",
                (session_id,),
            )

    def session_ids(self) -> list[str]:
        rows = self._connect().execute(
            "SELECT session_id FROM session_state ORDER BY session_id"
        ).fetchall()

        return [row[0] for row in rows]


def _task_state_to_dict(state: TaskState) -> dict[str, Any]:
    return {
        "task_id": state.anchor.task_id,
        "coverage": {
            "fetched_urls": sorted(state.coverage.fetched_urls),
            "failed_urls": sorted(state.coverage.failed_urls),
            "discovered_urls": sorted(
                state.coverage.discovered_urls
            ),
            "max_depth_reached": state.coverage.max_depth_reached,
            "total_bytes": state.coverage.total_bytes,
        },
    }
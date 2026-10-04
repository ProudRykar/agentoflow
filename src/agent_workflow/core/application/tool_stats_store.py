from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path


SCHEMA = """
CREATE TABLE IF NOT EXISTS tool_usage (
    session_id TEXT NOT NULL,
    tool_name  TEXT NOT NULL,
    calls      INTEGER NOT NULL DEFAULT 0,
    successes  INTEGER NOT NULL DEFAULT 0,
    errors     INTEGER NOT NULL DEFAULT 0,
    total_duration REAL NOT NULL DEFAULT 0,
    last_error_code TEXT,
    error_codes TEXT NOT NULL DEFAULT '{}',
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (session_id, tool_name)
);

CREATE TABLE IF NOT EXISTS approval_usage (
    session_id TEXT PRIMARY KEY,
    requested  INTEGER NOT NULL DEFAULT 0,
    allowed    INTEGER NOT NULL DEFAULT 0,
    denied     INTEGER NOT NULL DEFAULT 0,
    by_permission TEXT NOT NULL DEFAULT '{}',
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


@dataclass(slots=True, frozen=True)
class UsageRecord:
    session_id: str
    tool_name: str
    calls: int
    successes: int
    errors: int
    total_duration: float
    last_error_code: str | None
    error_codes: dict[str, int]

    @property
    def average_duration(self) -> float | None:
        finished = self.successes + self.errors

        if finished == 0:
            return None

        return self.total_duration / finished

    @property
    def error_rate(self) -> float:
        finished = self.successes + self.errors

        if finished == 0:
            return 0.0

        return self.errors / finished


class ToolStatsStore:
    """Durable counters for tool usage, keyed by session.

    Implements the ``StatsSink`` protocol so ``ToolStats.flush()``
    can hand counters over without importing persistence details.

    Statistics are a convenience, not a source of truth, so the
    store is deliberately forgiving: the schema is created on
    demand, a malformed JSON blob is treated as empty, and every
    failure is contained rather than propagated into the agent
    loop.
    """

    def __init__(self, database: Path) -> None:
        self._database = database

    def _connect(self) -> sqlite3.Connection:
        self._database.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        connection = sqlite3.connect(
            self._database,
            timeout=5.0,
        )

        # WAL lets the web UI read while an agent writes.
        connection.execute("PRAGMA journal_mode=WAL")
        connection.executescript(SCHEMA)

        return connection

    # ==================================================================
    # StatsSink protocol
    # ==================================================================

    def record_tool(
        self,
        session_id: str,
        usage: object,
    ) -> None:
        self.record_tool_delta(
            session_id,
            tool_name=str(getattr(usage, "name", "")),
            calls=int(getattr(usage, "calls", 0)),
            successes=int(getattr(usage, "successes", 0)),
            errors=int(getattr(usage, "errors", 0)),
            duration=float(getattr(usage, "total_duration", 0.0)),
            error_code=getattr(usage, "last_error_code", None),
            error_codes=getattr(usage, "error_codes", None),
        )

    def record_approval(
        self,
        session_id: str,
        usage: object,
    ) -> None:
        self.record_approval_delta(
            session_id,
            requested=int(getattr(usage, "requested", 0)),
            allowed=int(getattr(usage, "allowed", 0)),
            denied=int(getattr(usage, "denied", 0)),
            by_permission=getattr(usage, "by_permission", None),
        )

    # ==================================================================
    # Writes
    # ==================================================================

    def record_tool_delta(
        self,
        session_id: str,
        *,
        tool_name: str,
        calls: int = 0,
        successes: int = 0,
        errors: int = 0,
        duration: float = 0.0,
        error_code: str | None = None,
        error_codes: dict[str, int] | None = None,
    ) -> None:
        """Add the given deltas to a session's row for one tool."""

        if not tool_name:
            return

        try:
            with self._connect() as connection:
                connection.execute(
                    """
                    INSERT INTO tool_usage (
                        session_id, tool_name, calls, successes,
                        errors, total_duration, last_error_code,
                        error_codes
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(session_id, tool_name) DO UPDATE SET
                        calls = calls + excluded.calls,
                        successes = successes + excluded.successes,
                        errors = errors + excluded.errors,
                        total_duration =
                            total_duration + excluded.total_duration,
                        last_error_code = COALESCE(
                            excluded.last_error_code,
                            last_error_code
                        ),
                        error_codes = excluded.error_codes,
                        updated_at = datetime('now')
                    """,
                    (
                        session_id,
                        tool_name,
                        calls,
                        successes,
                        errors,
                        duration,
                        error_code,
                        json.dumps(error_codes or {}),
                    ),
                )
        except sqlite3.Error:
            # Statistics must never break a run.
            return

    def record_approval_delta(
        self,
        session_id: str,
        *,
        requested: int = 0,
        allowed: int = 0,
        denied: int = 0,
        by_permission: dict[str, int] | None = None,
    ) -> None:
        try:
            with self._connect() as connection:
                connection.execute(
                    """
                    INSERT INTO approval_usage (
                        session_id, requested, allowed, denied,
                        by_permission
                    )
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(session_id) DO UPDATE SET
                        requested = requested + excluded.requested,
                        allowed = allowed + excluded.allowed,
                        denied = denied + excluded.denied,
                        by_permission = excluded.by_permission,
                        updated_at = datetime('now')
                    """,
                    (
                        session_id,
                        requested,
                        allowed,
                        denied,
                        json.dumps(by_permission or {}),
                    ),
                )
        except sqlite3.Error:
            return

    # ==================================================================
    # Reads
    # ==================================================================

    def tool_rows(
        self,
        session_id: str,
    ) -> list[UsageRecord]:
        try:
            with self._connect() as connection:
                rows = connection.execute(
                    """
                    SELECT session_id, tool_name, calls, successes,
                           errors, total_duration, last_error_code,
                           error_codes
                    FROM tool_usage
                    WHERE session_id = ?
                    ORDER BY calls DESC, tool_name
                    """,
                    (session_id,),
                ).fetchall()
        except sqlite3.Error:
            return []

        return [
            UsageRecord(
                session_id=str(row[0]),
                tool_name=str(row[1]),
                calls=int(row[2]),
                successes=int(row[3]),
                errors=int(row[4]),
                total_duration=float(row[5] or 0.0),
                last_error_code=row[6],
                error_codes=_decode(row[7]),
            )
            for row in rows
        ]

    def approval_row(
        self,
        session_id: str,
    ) -> dict[str, object]:
        try:
            with self._connect() as connection:
                row = connection.execute(
                    """
                    SELECT requested, allowed, denied, by_permission
                    FROM approval_usage
                    WHERE session_id = ?
                    """,
                    (session_id,),
                ).fetchone()
        except sqlite3.Error:
            row = None

        if row is None:
            return {
                "requested": 0,
                "allowed": 0,
                "denied": 0,
                "by_permission": {},
            }

        return {
            "requested": int(row[0]),
            "allowed": int(row[1]),
            "denied": int(row[2]),
            "by_permission": _decode(row[3]),
        }

    def total_tool_calls(
        self,
        session_id: str | None = None,
    ) -> int:
        """Aggregate for one session, or across all of them."""

        try:
            with self._connect() as connection:
                if session_id is None:
                    row = connection.execute(
                        """
                        SELECT COALESCE(SUM(calls), 0)
                        FROM tool_usage
                        """
                    ).fetchone()
                else:
                    row = connection.execute(
                        """
                        SELECT COALESCE(SUM(calls), 0)
                        FROM tool_usage
                        WHERE session_id = ?
                        """,
                        (session_id,),
                    ).fetchone()
        except sqlite3.Error:
            return 0

        return int(row[0] or 0)

    def tool_totals(self) -> list[UsageRecord]:
        """Per-tool totals aggregated across every session."""

        try:
            with self._connect() as connection:
                rows = connection.execute(
                    """
                    SELECT tool_name,
                           SUM(calls) AS calls,
                           SUM(successes) AS successes,
                           SUM(errors) AS errors,
                           SUM(total_duration) AS total_duration
                    FROM tool_usage
                    GROUP BY tool_name
                    ORDER BY calls DESC, tool_name
                    """
                ).fetchall()
        except sqlite3.Error:
            return []

        totals: list[UsageRecord] = []

        for row in rows:
            total_duration = float(row[4] or 0.0)
            finished = int(row[2]) + int(row[3])

            totals.append(
                UsageRecord(
                    session_id="",
                    tool_name=str(row[0]),
                    calls=int(row[1]),
                    successes=int(row[2]),
                    errors=int(row[3]),
                    total_duration=total_duration,
                    last_error_code=None,
                    error_codes={},
                )
            )

            del finished

        return totals

    def top_tools(
        self,
        limit: int = 10,
    ) -> list[tuple[str, int]]:
        """Most-called tools across all sessions."""

        try:
            with self._connect() as connection:
                rows = connection.execute(
                    """
                    SELECT tool_name, SUM(calls) AS total
                    FROM tool_usage
                    GROUP BY tool_name
                    ORDER BY total DESC, tool_name
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
        except sqlite3.Error:
            return []

        return [(str(row[0]), int(row[1])) for row in rows]

    def forget(
        self,
        session_id: str,
    ) -> None:
        """Drop a session's rows when the session is closed."""

        try:
            with self._connect() as connection:
                connection.execute(
                    "DELETE FROM tool_usage WHERE session_id = ?",
                    (session_id,),
                )
                connection.execute(
                    "DELETE FROM approval_usage WHERE session_id = ?",
                    (session_id,),
                )
        except sqlite3.Error:
            return


def _decode(raw: object) -> dict[str, int]:
    if isinstance(raw, dict):
        return {
            str(key): int(value)
            for key, value in raw.items()
            if isinstance(value, (int, float))
        }

    if not isinstance(raw, str):
        return {}

    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError:
        return {}

    if not isinstance(decoded, dict):
        return {}

    return {
        str(key): int(value)
        for key, value in decoded.items()
        if isinstance(value, (int, float))
    }

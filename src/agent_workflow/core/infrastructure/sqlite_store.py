import sqlite3
from datetime import datetime
from pathlib import Path

from agent_workflow.core.entities.models.memory import MemoryEntry, MemoryStore


class SQLiteStore(MemoryStore):
    def __init__(self, database: Path) -> None:
        self._database = database

        self._database.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self._connection = sqlite3.connect(
            self._database,
        )

        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS memory (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )

        self._connection.commit()

    async def get(
        self,
        key: str,
    ) -> MemoryEntry | None:
        cursor = self._connection.execute(
            """
            SELECT key, value, created_at, updated_at
            FROM memory
            WHERE key = ?
            """,
            (key,),
        )

        row = cursor.fetchone()

        if row is None:
            return None

        return self._entry_from_row(row)

    async def set(
        self,
        entry: MemoryEntry,
    ) -> None:
        self._connection.execute(
            """
            INSERT INTO memory (
                key,
                value,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET
                value = excluded.value,
                updated_at = excluded.updated_at
            """,
            (
                entry.key,
                entry.value,
                entry.created_at.isoformat(),
                entry.updated_at.isoformat(),
            ),
        )

        self._connection.commit()

    async def delete(
        self,
        key: str,
    ) -> None:
        self._connection.execute(
            """
            DELETE FROM memory
            WHERE key = ?
            """,
            (key,),
        )

        self._connection.commit()

    async def all(
        self,
    ) -> tuple[MemoryEntry, ...]:
        cursor = self._connection.execute(
            """
            SELECT key, value, created_at, updated_at
            FROM memory
            ORDER BY key
            """
        )

        return tuple(
            self._entry_from_row(row)
            for row in cursor.fetchall()
        )

    @staticmethod
    def _entry_from_row(
        row: tuple[str, str, str, str],
    ) -> MemoryEntry:
        return MemoryEntry(
            key=row[0],
            value=row[1],
            created_at=datetime.fromisoformat(row[2]),
            updated_at=datetime.fromisoformat(row[3]),
        )

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> "SQLiteStore":
        return self

    def __exit__(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
    ) -> None:
        self.close()
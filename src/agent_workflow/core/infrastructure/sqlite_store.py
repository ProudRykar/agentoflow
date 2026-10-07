import sqlite3
from datetime import datetime
from pathlib import Path

from agent_workflow.core.entities.models.memory import (
    GLOBAL_SCOPE,
    MemoryEntry,
    MemoryStore,
)


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

        self._migrate()

    def _migrate(self) -> None:
        """Ensure the scoped schema, moving an older one across.

        The original table keyed notes by ``key`` alone, which cannot
        hold the same key in two task scopes. Rather than fail on an
        existing database, the rows are copied into the new shape:
        they were all written before scopes existed, so every one of
        them is global by definition and the migration is lossless.
        """

        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS memory (
                scope TEXT NOT NULL DEFAULT '',
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                tags TEXT NOT NULL DEFAULT '',
                PRIMARY KEY (scope, key)
            )
            """
        )

        columns = {
            row[1]
            for row in self._connection.execute(
                "PRAGMA table_info(memory)"
            )
        }

        if "scope" in columns:
            self._connection.commit()
            return

        # A table without "scope" is the old shape under a name the
        # CREATE above will not have replaced, because it already
        # existed.
        self._connection.execute(
            """
            CREATE TABLE memory_scoped (
                scope TEXT NOT NULL DEFAULT '',
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                tags TEXT NOT NULL DEFAULT '',
                PRIMARY KEY (scope, key)
            )
            """
        )

        self._connection.execute(
            """
            INSERT OR REPLACE INTO memory_scoped (
                scope, key, value, created_at, updated_at, tags
            )
            SELECT '', key, value, created_at, updated_at, ''
            FROM memory
            """
        )

        self._connection.execute("DROP TABLE memory")
        self._connection.execute(
            "ALTER TABLE memory_scoped RENAME TO memory"
        )

        self._connection.commit()

    async def get(
        self,
        key: str,
        *,
        scope: str = GLOBAL_SCOPE,
    ) -> MemoryEntry | None:
        cursor = self._connection.execute(
            """
            SELECT scope, key, value, created_at, updated_at, tags
            FROM memory
            WHERE scope = ? AND key = ?
            """,
            (scope, key),
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
                scope, key, value, created_at, updated_at, tags
            )
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(scope, key) DO UPDATE SET
                value = excluded.value,
                updated_at = excluded.updated_at,
                tags = excluded.tags
            """,
            (
                entry.scope,
                entry.key,
                entry.value,
                entry.created_at.isoformat(),
                entry.updated_at.isoformat(),
                ",".join(entry.tags),
            ),
        )

        self._connection.commit()

    async def delete(
        self,
        key: str,
        *,
        scope: str = GLOBAL_SCOPE,
    ) -> None:
        self._connection.execute(
            """
            DELETE FROM memory
            WHERE scope = ? AND key = ?
            """,
            (scope, key),
        )

        self._connection.commit()

    async def all(
        self,
        *,
        scope: str | None = None,
    ) -> tuple[MemoryEntry, ...]:
        if scope is None:
            cursor = self._connection.execute(
                """
                SELECT scope, key, value, created_at, updated_at, tags
                FROM memory
                ORDER BY scope, key
                """
            )
        else:
            cursor = self._connection.execute(
                """
                SELECT scope, key, value, created_at, updated_at, tags
                FROM memory
                WHERE scope IN (?, '')
                ORDER BY scope, key
                """,
                (scope,),
            )

        return tuple(
            self._entry_from_row(row)
            for row in cursor.fetchall()
        )

    def _entry_from_row(
        self,
        row: tuple,
    ) -> MemoryEntry:
        (
            scope,
            key,
            value,
            created_at,
            updated_at,
            tags,
        ) = row

        return MemoryEntry(
            key=key,
            value=value,
            created_at=datetime.fromisoformat(created_at),
            updated_at=datetime.fromisoformat(updated_at),
            scope=scope or GLOBAL_SCOPE,
            tags=tuple(
                item for item in tags.split(",") if item
            ),
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

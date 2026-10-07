from __future__ import annotations

from datetime import UTC, datetime

from agent_workflow.core.entities.models.memory import (
    GLOBAL_SCOPE,
    MemoryEntry,
    MemoryStore,
)


class MemoryManager:
    """Writes notes, scoped to a task unless told otherwise.

    Every method takes a scope because "where does this belong" is a
    question every caller has to answer anyway. Making it a required
    argument at the boundary is why notes do not silently become
    global: forgetting to scope is now visible in a signature.
    """

    def __init__(self, store: MemoryStore) -> None:
        self._store = store

    async def get(
        self,
        key: str,
        *,
        scope: str = GLOBAL_SCOPE,
    ) -> MemoryEntry | None:
        return await self._store.get(key, scope=scope)

    async def remember(
        self,
        key: str,
        value: str,
        *,
        scope: str = GLOBAL_SCOPE,
        tags: tuple[str, ...] = (),
    ) -> MemoryEntry:
        now = datetime.now(UTC)

        existing = await self._store.get(key, scope=scope)

        if existing is None:
            entry = MemoryEntry(
                key=key,
                value=value,
                created_at=now,
                updated_at=now,
                scope=scope,
                tags=tags,
            )
        else:
            entry = MemoryEntry(
                key=key,
                value=value,
                created_at=existing.created_at,
                updated_at=now,
                scope=scope,
                # Tags accumulate: a note rewritten without repeating
                # its labels should not lose them.
                tags=tuple(
                    dict.fromkeys((*existing.tags, *tags))
                ),
            )

        await self._store.set(entry)

        return entry

    async def promote(
        self,
        key: str,
        *,
        from_scope: str,
        tags: tuple[str, ...] = (),
    ) -> MemoryEntry:
        """Copy a task-scoped note into the global scope.

        The copy rather than a move: a task may still want the note
        after it stops being new, and deleting the original would make
        that a surprise mid-run.
        """

        if from_scope == GLOBAL_SCOPE:
            raise ValueError(
                "a global note is already global"
            )

        entry = await self._store.get(key, scope=from_scope)

        if entry is None:
            raise KeyError(
                f"no note '{key}' in scope '{from_scope}'"
            )

        return await self.remember(
            key,
            entry.value,
            scope=GLOBAL_SCOPE,
            tags=tags or entry.tags,
        )

    async def forget(
        self,
        key: str,
        *,
        scope: str = GLOBAL_SCOPE,
    ) -> None:
        await self._store.delete(key, scope=scope)

    async def all(
        self,
        *,
        scope: str | None = None,
    ) -> tuple[MemoryEntry, ...]:
        return await self._store.all(scope=scope)

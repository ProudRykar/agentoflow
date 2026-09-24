from datetime import UTC, datetime

from agent_workflow.core.entities.models.memory import MemoryEntry, MemoryStore


class MemoryManager:
    def __init__(self, store: MemoryStore) -> None:
        self._store = store

    async def get(self, key: str) -> MemoryEntry | None:
        return await self._store.get(key)

    async def remember(
        self,
        key: str,
        value: str,
    ) -> MemoryEntry:
        now = datetime.now(UTC)

        existing = await self._store.get(key)

        if existing is None:
            entry = MemoryEntry(
                key=key,
                value=value,
                created_at=now,
                updated_at=now,
            )
        else:
            entry = MemoryEntry(
                key=key,
                value=value,
                created_at=existing.created_at,
                updated_at=now,
            )

        await self._store.set(entry)

        return entry

    async def forget(self, key: str) -> None:
        await self._store.delete(key)

    async def all(self) -> tuple[MemoryEntry, ...]:
        return await self._store.all()
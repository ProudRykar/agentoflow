from agent_workflow.core.entities.models.memory import MemoryEntry, MemoryStore


class InMemoryStore(MemoryStore):
    def __init__(self) -> None:
        self._entries: dict[str, MemoryEntry] = {}

    async def get(self, key: str) -> MemoryEntry | None:
        return self._entries.get(key)

    async def set(self, entry: MemoryEntry) -> None:
        self._entries[entry.key] = entry

    async def delete(self, key: str) -> None:
        self._entries.pop(key, None)

    async def all(self) -> tuple[MemoryEntry, ...]:
        return tuple(self._entries.values())
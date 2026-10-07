from agent_workflow.core.entities.models.memory import (
    GLOBAL_SCOPE,
    MemoryEntry,
    MemoryStore,
)


class InMemoryStore(MemoryStore):
    def __init__(self) -> None:
        self._entries: dict[tuple[str, str], MemoryEntry] = {}

    async def get(
        self,
        key: str,
        *,
        scope: str = GLOBAL_SCOPE,
    ) -> MemoryEntry | None:
        return self._entries.get((scope, key))

    async def set(self, entry: MemoryEntry) -> None:
        self._entries[(entry.scope, entry.key)] = entry

    async def delete(
        self,
        key: str,
        *,
        scope: str = GLOBAL_SCOPE,
    ) -> None:
        self._entries.pop((scope, key), None)

    async def all(
        self,
        *,
        scope: str | None = None,
    ) -> tuple[MemoryEntry, ...]:
        if scope is None:
            return tuple(self._entries.values())

        # A task sees its own notes and the global ones, never another
        # task's.
        return tuple(
            entry
            for entry in self._entries.values()
            if entry.scope in (scope, GLOBAL_SCOPE)
        )

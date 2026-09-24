from dataclasses import dataclass
from datetime import datetime


@dataclass(slots=True, frozen=True)
class MemoryEntry:
    key: str
    value: str
    created_at: datetime
    updated_at: datetime


class MemoryStore:
    async def get(self, key: str) -> MemoryEntry | None:
        raise NotImplementedError

    async def set(self, entry: MemoryEntry) -> None:
        raise NotImplementedError

    async def delete(self, key: str) -> None:
        raise NotImplementedError

    async def all(self) -> tuple[MemoryEntry, ...]:
        raise NotImplementedError
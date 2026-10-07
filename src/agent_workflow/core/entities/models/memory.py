from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

# An empty scope means the note outlives the task that wrote it.
# Named rather than a bool so a future "this whole project" scope can
# be added without another column, and so a scope reads as an
# identifier in a database dump.
GLOBAL_SCOPE = ""


@dataclass(slots=True, frozen=True)
class MemoryEntry:
    key: str
    value: str
    created_at: datetime
    updated_at: datetime

    scope: str = GLOBAL_SCOPE
    """Whose knowledge this is.

    Scoped notes belong to one task and are only offered back to it.
    An unscoped note is global and survives the task. The distinction
    is what keeps a note about how one game works from being handed to
    an unrelated request a week later.
    """

    tags: tuple[str, ...] = ()
    """Free-form labels, for grouping that key prefixes cannot express.

    Empty by default and never used for retrieval; keywords do that.
    Kept because "related to this project" is a thing a person wants
    to record even when it is not a fact to retrieve.
    """

    @property
    def global_scope(self) -> bool:
        return self.scope == GLOBAL_SCOPE


class MemoryStore:
    async def get(
        self,
        key: str,
        *,
        scope: str = GLOBAL_SCOPE,
    ) -> MemoryEntry | None:
        raise NotImplementedError

    async def set(self, entry: MemoryEntry) -> None:
        raise NotImplementedError

    async def delete(
        self,
        key: str,
        *,
        scope: str = GLOBAL_SCOPE,
    ) -> None:
        raise NotImplementedError

    async def all(
        self,
        *,
        scope: str | None = None,
    ) -> tuple[MemoryEntry, ...]:
        """Every entry, or the ones visible from ``scope``.

        ``None`` means everything, which is what an operator listing
        storage wants. A concrete scope means what one task may see:
        its own notes plus the global ones.
        """

        raise NotImplementedError

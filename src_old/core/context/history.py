from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

MAX_HISTORY_CONTENT_CHARS = 4_000


class HistoryKind(StrEnum):
    """What happened, at event granularity."""

    USER_MESSAGE = "user_message"
    ASSISTANT_MESSAGE = "assistant_message"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    PHASE_CHANGE = "phase_change"
    EVIDENCE_REF = "evidence_ref"
    CHECKPOINT = "checkpoint"


@dataclass(slots=True, frozen=True)
class HistoryItem:
    """One persisted event.

    Hierarchy: task -> run -> window -> item. Items are
    append-only; the active context only ever receives
    explicitly retrieved ones.
    """

    item_id: str
    task_id: str
    run_id: str
    window_id: str
    kind: HistoryKind
    content: str
    reference: str | None = None

    def __post_init__(self) -> None:
        if not self.item_id:
            raise ValueError(
                "item_id must not be empty",
            )

        if not self.task_id:
            raise ValueError(
                "task_id must not be empty",
            )

        if not self.window_id:
            raise ValueError(
                "window_id must not be empty",
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "item_id": self.item_id,
            "task_id": self.task_id,
            "run_id": self.run_id,
            "window_id": self.window_id,
            "kind": self.kind.value,
            "content": self.content,
            "reference": self.reference,
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, object],
    ) -> HistoryItem:
        reference = data.get("reference")

        if reference is not None and not isinstance(
            reference,
            str,
        ):
            raise TypeError(
                "reference must be a string or None",
            )

        return cls(
            item_id=str(data["item_id"]),
            task_id=str(data["task_id"]),
            run_id=str(data.get("run_id", "")),
            window_id=str(data["window_id"]),
            kind=HistoryKind(str(data["kind"])),
            content=str(data.get("content", "")),
            reference=reference,
        )


class HistoryStore(Protocol):
    """Append-only event storage."""

    def append(
        self,
        task_id: str,
        run_id: str,
        window_id: str,
        kind: HistoryKind,
        content: str,
        reference: str | None = None,
    ) -> HistoryItem: ...

    def get(
        self,
        item_id: str,
    ) -> HistoryItem | None: ...

    def window_items(
        self,
        task_id: str,
        window_id: str,
    ) -> tuple[HistoryItem, ...]: ...

    def task_items(
        self,
        task_id: str,
    ) -> tuple[HistoryItem, ...]: ...

    def recent_items(
        self,
        task_id: str,
        limit: int,
    ) -> tuple[HistoryItem, ...]: ...

    def search(
        self,
        task_id: str,
        query: str,
        limit: int = 20,
    ) -> tuple[HistoryItem, ...]: ...

    def by_reference(
        self,
        reference: str,
    ) -> tuple[HistoryItem, ...]: ...


class InMemoryHistoryStore:
    """In-memory HistoryStore. SQLite later, same protocol."""

    def __init__(self) -> None:
        self._items: dict[str, HistoryItem] = {}
        self._order: list[str] = []
        self._counter: int = 0

    def __len__(self) -> int:
        return len(self._items)

    def append(
        self,
        task_id: str,
        run_id: str,
        window_id: str,
        kind: HistoryKind,
        content: str,
        reference: str | None = None,
    ) -> HistoryItem:
        if not task_id:
            raise ValueError(
                "task_id must not be empty",
            )

        if not window_id:
            raise ValueError(
                "window_id must not be empty",
            )

        self._counter += 1

        if len(content) > MAX_HISTORY_CONTENT_CHARS:
            content = (
                content[:MAX_HISTORY_CONTENT_CHARS]
                + "\n[history content truncated]"
            )

        item = HistoryItem(
            item_id=f"hist-{self._counter:06d}",
            task_id=task_id,
            run_id=run_id,
            window_id=window_id,
            kind=kind,
            content=content,
            reference=reference,
        )

        self._items[item.item_id] = item
        self._order.append(item.item_id)

        return item

    def get(
        self,
        item_id: str,
    ) -> HistoryItem | None:
        return self._items.get(item_id)

    def window_items(
        self,
        task_id: str,
        window_id: str,
    ) -> tuple[HistoryItem, ...]:
        return tuple(
            self._items[item_id]
            for item_id in self._order
            if (
                self._items[item_id].task_id == task_id
                and self._items[item_id].window_id
                == window_id
            )
        )

    def task_items(
        self,
        task_id: str,
    ) -> tuple[HistoryItem, ...]:
        return tuple(
            self._items[item_id]
            for item_id in self._order
            if self._items[item_id].task_id == task_id
        )

    def recent_items(
        self,
        task_id: str,
        limit: int,
    ) -> tuple[HistoryItem, ...]:
        if limit <= 0:
            raise ValueError(
                "limit must be greater than 0",
            )

        matched = [
            self._items[item_id]
            for item_id in self._order
            if self._items[item_id].task_id == task_id
        ]

        return tuple(matched[-limit:])

    def search(
        self,
        task_id: str,
        query: str,
        limit: int = 20,
    ) -> tuple[HistoryItem, ...]:
        if not query:
            raise ValueError(
                "query must not be empty",
            )

        if limit <= 0:
            raise ValueError(
                "limit must be greater than 0",
            )

        lowered = query.lower()
        found: list[HistoryItem] = []

        for item_id in self._order:
            item = self._items[item_id]

            if item.task_id != task_id:
                continue

            if lowered in item.content.lower():
                found.append(item)

            if len(found) >= limit:
                break

        return tuple(found)

    def by_reference(
        self,
        reference: str,
    ) -> tuple[HistoryItem, ...]:
        return tuple(
            self._items[item_id]
            for item_id in self._order
            if self._items[item_id].reference == reference
        )

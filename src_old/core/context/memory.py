from __future__ import annotations

import re
from dataclasses import dataclass

from core.context.context_item import (
    ContextItem,
    ContextSource,
)
from core.entities.models.memory import MemoryEntry

WORD_PATTERN = re.compile(r"[a-z0-9]+")

DEFAULT_SNAPSHOT_LIMIT = 5
MIN_TOKEN_LENGTH = 3


def query_terms(
    query: str,
) -> frozenset[str]:
    return frozenset(
        token
        for token in WORD_PATTERN.findall(query.lower())
        if len(token) >= MIN_TOKEN_LENGTH
    )


def entry_terms(
    entry: MemoryEntry,
) -> frozenset[str]:
    return frozenset(
        token
        for token in WORD_PATTERN.findall(
            f"{entry.key} {entry.value}".lower()
        )
        if len(token) >= MIN_TOKEN_LENGTH
    )


def score_entry(
    entry: MemoryEntry,
    terms: frozenset[str],
) -> int:
    if not terms:
        return 0

    return len(entry_terms(entry) & terms)


def memory_to_item(
    entry: MemoryEntry,
) -> ContextItem:
    return ContextItem(
        source=ContextSource.MEMORY,
        content=f"{entry.key}: {entry.value}",
        reference=entry.key,
    )


@dataclass(slots=True, frozen=True)
class MemorySnapshot:
    """Explicit retrieval result for one LLM request."""

    items: tuple[ContextItem, ...] = ()

    def __len__(self) -> int:
        return len(self.items)


def retrieve_snapshot(
    entries: tuple[MemoryEntry, ...],
    query: str,
    limit: int = DEFAULT_SNAPSHOT_LIMIT,
) -> MemorySnapshot:
    """
    Select memories relevant to the query.

    Deterministic keyword overlap, no embeddings: an entry is
    selected only if it shares at least one significant token
    with the query. Highest overlap first, stable for ties.
    """

    if limit <= 0:
        raise ValueError(
            "limit must be greater than 0",
        )

    terms = query_terms(query)

    scored = sorted(
        (
            (score_entry(entry, terms), index, entry)
            for index, entry in enumerate(entries)
        ),
        key=lambda scored_entry: (
            -scored_entry[0],
            scored_entry[1],
        ),
    )

    return MemorySnapshot(
        items=tuple(
            memory_to_item(entry)
            for score, _, entry in scored[:limit]
            if score > 0
        ),
    )

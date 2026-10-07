from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from agent_workflow.core.context.context_item import (
    ContextItem,
    ContextSource,
)
from agent_workflow.core.entities.models.memory import (
    GLOBAL_SCOPE,
    MemoryEntry,
)

WORD_PATTERN = re.compile(r"[a-z0-9]+")

DEFAULT_SNAPSHOT_LIMIT = 5
MIN_TOKEN_LENGTH = 3

# Words that are long enough to pass the length filter but say nothing
# about what a note is about.
#
# Without this, "the" matched on its own: a note reading "the red key
# opens the blue door" was selected for the query "play the game",
# purely because both contained an article. Retrieval by keyword overlap
# is only as good as its definition of a significant word, and this is
# that definition. Deliberately short -- a longer list starts dropping
# words that carry meaning in a short note.
STOP_WORDS = frozenset({
    "about", "after", "all", "and", "any", "are", "been", "before",
    "being", "but", "can", "did", "does", "for", "from", "had", "has",
    "have", "her", "him", "his", "how", "into", "its", "just", "like",
    "make", "made", "many", "more", "most", "much", "not", "now", "one",
    "only", "other", "our", "out", "over", "she", "should", "some",
    "such", "than", "that", "the", "their", "them", "then", "there",
    "these", "they", "this", "those", "too", "very", "was", "were",
    "what", "when", "which", "while", "who", "will", "with", "would",
    "you", "your",
})


def _significant(token: str) -> bool:
    return (
        len(token) >= MIN_TOKEN_LENGTH
        and token not in STOP_WORDS
    )


# Endings that mean the word is already singular, so stripping an "s"
# would produce a different word rather than the same one.
_NOT_SINGULAR = ("ss", "us", "is", "as", "os")


def _with_variants(token: str) -> set[str]:
    """The token, plus a crude singular.

    Only the plural "s", and only when removing it is unambiguous. A
    note saying "collect 3 keys" and a request asking "where is the
    key" are about the same thing, and without this the two share no
    token at all -- which is the whole failure this module exists to
    avoid. Deliberately not a real stemmer: an over-eager one maps
    unrelated words together, and false matches are worse here than a
    missed one, because they fill the block with noise.
    """

    forms = {token}

    if (
        len(token) >= 4
        and token.endswith("s")
        and not token.endswith(_NOT_SINGULAR)
    ):
        forms.add(token[:-1])

    return forms

# How much of the recent conversation is mined for retrieval terms.
#
# Deliberately small. The point is that a fact learned two turns ago
# shares words with what is happening now; a hundred turns of history
# would match everything and select nothing in particular.
RETRIEVAL_TAIL_MESSAGES = 6

# A message can be a wall of tool output. Long values are truncated so
# one verbose result cannot fill the query on its own and pull every
# note in.
RETRIEVAL_VALUE_CHARS = 600


def query_terms(
    query: str,
) -> frozenset[str]:
    return frozenset(
        form
        for token in WORD_PATTERN.findall(query.lower())
        if _significant(token)
        for form in _with_variants(token)
    )


def entry_terms(
    entry: MemoryEntry,
) -> frozenset[str]:
    return frozenset(
        form
        for token in WORD_PATTERN.findall(
            f"{entry.key} {entry.value}".lower()
        )
        if _significant(token)
        for form in _with_variants(token)
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


def retrieval_query(
    objective: str,
    dialogue: list[dict[str, Any]] | tuple[dict[str, Any], ...] = (),
    *,
    tail: int = RETRIEVAL_TAIL_MESSAGES,
) -> str:
    """What to match notes against: the task, plus what just happened.

    The original prompt is the wrong target on its own, and not
    slightly wrong. ``TaskAnchor`` is immutable by design, so it always
    describes what the user asked at the start and never what the
    agent has since worked out. Ask an agent to play a game, let it
    discover that a red key opens a blue door, and the words for that
    discovery appear nowhere in the prompt -- so a note recording it
    shares no term with the query and is never selected, no matter how
    carefully it was written.

    Matching against the live tail fixes that case rather than
    generalising it: a fact learned recently shares vocabulary with the
    conversation that produced it. The objective is still included so
    that a note written from the original request is still reachable
    when nothing recent is relevant.
    """

    parts: list[str] = [objective]

    if tail > 0:
        for message in dialogue[-tail:]:
            text = message_text(message)

            if text:
                parts.append(text[:RETRIEVAL_VALUE_CHARS])

    return "\n".join(part for part in parts if part)


def message_text(message: dict[str, Any]) -> str:
    """The readable part of one dialogue message.

    Handles both shapes seen in practice: a plain ``{"content": str}``
    and the provider form where content is a list of blocks.
    """

    content = message.get("content")

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        chunks: list[str] = []

        for block in content:
            if isinstance(block, dict):
                text = block.get("text")

                if isinstance(text, str):
                    chunks.append(text)

            elif isinstance(block, str):
                chunks.append(block)

        return " ".join(chunks)

    return ""


def visible_entries(
    entries: tuple[MemoryEntry, ...],
    scope: str,
) -> tuple[MemoryEntry, ...]:
    """The notes one task is allowed to be offered.

    A note belonging to another task is not merely ranked lower, it is
    absent: the alternative is a game note surfacing under an
    unrelated request, which teaches the model to distrust the block
    entirely.
    """

    return tuple(
        entry
        for entry in entries
        if entry.scope in (scope, GLOBAL_SCOPE)
    )

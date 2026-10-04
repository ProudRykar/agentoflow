from __future__ import annotations

from dataclasses import dataclass

from agent_workflow.core.entities.models.research_contract import ResearchPage


@dataclass(slots=True, frozen=True)
class Evidence:
    """One unit of trusted data found by the runtime.

    ``ResearchPage`` is just one evidence kind. Later kinds will be
    shell results, test results, API responses, subagent results.
    The conversation only ever carries the ``evidence_id``;
    the content lives here.
    """

    evidence_id: str
    kind: str
    source: str
    title: str
    content: str
    depth: int = 0
    content_bytes: int = 0
    links: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.evidence_id:
            raise ValueError(
                "evidence_id must not be empty",
            )

        if not self.kind:
            raise ValueError(
                "kind must not be empty",
            )

        object.__setattr__(
            self,
            "links",
            tuple(self.links),
        )

    @classmethod
    def from_research_page(
        cls,
        page: ResearchPage,
        evidence_id: str,
        source: str,
    ) -> Evidence:
        return cls(
            evidence_id=evidence_id,
            kind="research.page",
            source=source,
            title=page.title,
            content=page.content,
            depth=page.depth,
            content_bytes=page.content_bytes,
            links=tuple(page.links),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "evidence_id": self.evidence_id,
            "kind": self.kind,
            "source": self.source,
            "title": self.title,
            "content": self.content,
            "depth": self.depth,
            "content_bytes": self.content_bytes,
            "links": list(self.links),
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, object],
    ) -> Evidence:
        raw_links = data.get("links", ())

        if not isinstance(
            raw_links,
            (list, tuple),
        ):
            raise TypeError(
                "links must be a list or tuple",
            )

        return cls(
            evidence_id=str(data["evidence_id"]),
            kind=str(data["kind"]),
            source=str(data["source"]),
            title=str(data.get("title", "")),
            content=str(data.get("content", "")),
            depth=int(data.get("depth", 0)),
            content_bytes=int(
                data.get("content_bytes", 0),
            ),
            links=tuple(
                str(url) for url in raw_links
            ),
        )


@dataclass(slots=True, frozen=True)
class EvidenceReceipt:
    """Compact conversation representation of stored evidence."""

    evidence_ids: tuple[str, ...]
    page_count: int
    total_bytes: int
    max_depth_reached: int
    satisfied: bool

    def to_text(
        self,
        tool_name: str,
    ) -> str:
        ids = ", ".join(self.evidence_ids)

        return (
            f"[research] {tool_name} succeeded: "
            f"{self.page_count} page(s), "
            f"{self.total_bytes} bytes, "
            f"depth {self.max_depth_reached}. "
            f"Evidence registered: {ids}. "
            f"Full content is held by the runtime; "
            f"do not re-fetch these URLs."
        )


DEFAULT_MAX_EVIDENCE = 200


class EvidenceStore:
    """Append-only task-owned storage for evidence.

    In-memory implementation. A persistent implementation can
    replace it later without changing callers, because evidence
    already serializes via ``to_dict()`` / ``from_dict()``.
    """

    def __init__(
        self,
        max_items: int = DEFAULT_MAX_EVIDENCE,
    ) -> None:
        self._items: dict[str, Evidence] = {}
        self._counter: int = 0
        self._max_items = max(1, max_items)
        self._pinned: set[str] = set()

    @property
    def max_items(self) -> int:
        """Soft limit: the age-based eviction target."""

        return self._max_items

    @property
    def hard_max_items(self) -> int:
        """Ceiling that even pinned items may not cross."""

        return self._max_items * 2

    def __len__(self) -> int:
        return len(self._items)

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(self._items)

    def pin(self, evidence_id: str) -> None:
        """Keep an item regardless of age.

        Used for evidence a checkpoint cites: dropping it would leave
        the checkpoint referring to something the model can no longer
        read.
        """

        if evidence_id in self._items:
            self._pinned.add(evidence_id)

    def unpin(self, evidence_id: str) -> None:
        self._pinned.discard(evidence_id)

    def trim_to(self, max_items: int) -> None:
        """Apply a limit chosen by the caller.

        Used when restoring from disk: the live cap is sized for one
        run, but a restored store should not immediately drop the
        evidence a checkpoint cites.
        """

        self._max_items = max(1, max_items)
        self._evict()

    def _evict(self) -> None:
        """Drop the oldest unpinned items down to the soft limit.

        The limit applies to unpinned evidence only: pinned items are
        referenced by checkpoints, so they are held on top of the
        budget rather than counting against it. A hard ceiling still
        applies to the total, so pinning cannot grow without bound.
        """

        unpinned = [
            evidence_id
            for evidence_id in self._items
            if evidence_id not in self._pinned
        ]

        overflow = len(unpinned) - self._max_items

        for evidence_id in unpinned[:max(0, overflow)]:
            self._items.pop(evidence_id, None)

        while len(self._items) > self.hard_max_items:
            oldest = next(iter(self._items), None)

            if oldest is None:
                break

            del self._items[oldest]

    def append_page(
        self,
        page: ResearchPage,
        source: str,
    ) -> Evidence:
        self._counter += 1

        evidence_id = f"ev-{self._counter:04d}"

        evidence = Evidence.from_research_page(
            page,
            evidence_id,
            source,
        )

        self._items[evidence_id] = evidence
        self._evict()

        return evidence

    def get(
        self,
        evidence_id: str,
    ) -> Evidence | None:
        return self._items.get(evidence_id)

    def all(self) -> tuple[Evidence, ...]:
        return tuple(self._items.values())

    def to_dict(self) -> dict[str, object]:
        return {
            "items": [
                evidence.to_dict()
                for evidence in self._items.values()
            ],
            "counter": self._counter,
            "max_items": self._max_items,
            "pinned": sorted(self._pinned),
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, object],
    ) -> EvidenceStore:
        store = cls()

        raw_items = data.get("items", [])

        if not isinstance(raw_items, list):
            raise TypeError(
                "items must be a list",
            )

        for raw in raw_items:
            if not isinstance(raw, dict):
                raise TypeError(
                    "evidence item must be a dict",
                )

            evidence = Evidence.from_dict(raw)
            store._items[evidence.evidence_id] = evidence

        store._counter = int(data.get("counter", len(store._items)))
        store._max_items = int(
            data.get("max_items", store._max_items)  # type: ignore[arg-type]
        )
        store._pinned = {
            str(item)
            for item in (data.get("pinned") or [])  # type: ignore[union-attr]
        }

        return store

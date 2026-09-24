from __future__ import annotations

from dataclasses import dataclass

from core.entities.models.research_contract import ResearchPage


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


class EvidenceStore:
    """Append-only task-owned storage for evidence.

    In-memory implementation. A persistent implementation can
    replace it later without changing callers, because evidence
    already serializes via ``to_dict()`` / ``from_dict()``.
    """

    def __init__(self) -> None:
        self._items: dict[str, Evidence] = {}
        self._counter: int = 0

    def __len__(self) -> int:
        return len(self._items)

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(self._items)

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

        return store

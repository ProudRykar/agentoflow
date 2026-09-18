from __future__ import annotations

import json

from dataclasses import dataclass, field
from urllib.parse import SplitResult, urlsplit, urlunsplit


def canonicalize_url(url: str) -> str:
    value = str(url).strip()

    if not value:
        raise ValueError(
            "URL must not be empty"
        )

    parsed = urlsplit(value)

    scheme = parsed.scheme.lower()

    if scheme not in {"http", "https"}:
        raise ValueError(
            "Unsupported URL scheme: "
            f"{parsed.scheme or '<missing>'}"
        )

    if not parsed.hostname:
        raise ValueError(
            "URL must contain a host"
        )

    if (
        parsed.username is not None
        or parsed.password is not None
    ):
        raise ValueError(
            "URLs with embedded credentials are not allowed"
        )

    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError(
            f"Invalid URL port: {value!r}"
        ) from exc

    hostname = parsed.hostname.lower().rstrip(".")

    if ":" in hostname and not hostname.startswith("["):
        hostname = f"[{hostname}]"

    netloc = hostname

    if port is not None and not (
        (
            scheme == "http"
            and port == 80
        )
        or (
            scheme == "https"
            and port == 443
        )
    ):
        netloc = f"{netloc}:{port}"

    return urlunsplit(
        SplitResult(
            scheme,
            netloc,
            parsed.path or "/",
            parsed.query,
            "",
        )
    )


@dataclass(slots=True, frozen=True)
class ResearchPage:
    """
    One page actually fetched by the research runtime.
    """

    url: str
    depth: int
    title: str
    content: str
    links: tuple[str, ...]
    content_bytes: int

    def __post_init__(self) -> None:
        if self.depth < 0:
            raise ValueError(
                "page depth must be >= 0"
            )

        if self.content_bytes < 0:
            raise ValueError(
                "content_bytes must be >= 0"
            )

        object.__setattr__(
            self,
            "url",
            canonicalize_url(self.url),
        )

        object.__setattr__(
            self,
            "links",
            tuple(
                canonicalize_url(url)
                for url in self.links
            ),
        )


@dataclass(slots=True, frozen=True)
class ResearchResult:
    """
    Structured evidence produced by the trusted research runtime.

    This is deliberately not a boolean such as research_completed=True.
    Completion is decided later by ResearchContract + ResearchCoverage.
    """

    root_url: str

    pages: tuple[ResearchPage, ...]

    discovered_urls: tuple[str, ...]

    failed_urls: tuple[str, ...]

    max_depth_reached: int

    total_bytes: int

    page_limit_reached: bool

    byte_limit_reached: bool

    schema_version: int = 1

    kind: str = (
        "agentoflow.research_result"
    )

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError(
                "Unsupported research result schema: "
                f"{self.schema_version}"
            )

        if self.max_depth_reached < 0:
            raise ValueError(
                "max_depth_reached must be >= 0"
            )

        if self.total_bytes < 0:
            raise ValueError(
                "total_bytes must be >= 0"
            )

        object.__setattr__(
            self,
            "root_url",
            canonicalize_url(self.root_url),
        )

        object.__setattr__(
            self,
            "discovered_urls",
            tuple(
                canonicalize_url(url)
                for url in self.discovered_urls
            ),
        )

        object.__setattr__(
            self,
            "failed_urls",
            tuple(
                canonicalize_url(url)
                for url in self.failed_urls
            ),
        )

    @property
    def fetched_urls(self) -> frozenset[str]:
        return frozenset(
            page.url
            for page in self.pages
        )

    def to_json(self) -> str:
        payload = {
            "kind": self.kind,
            "schema_version": self.schema_version,
            "root_url": self.root_url,
            "pages": [
                {
                    "url": page.url,
                    "depth": page.depth,
                    "title": page.title,
                    "content": page.content,
                    "links": list(page.links),
                    "content_bytes": page.content_bytes,
                }
                for page in self.pages
            ],
            "discovered_urls": list(
                self.discovered_urls
            ),
            "failed_urls": list(
                self.failed_urls
            ),
            "max_depth_reached": (
                self.max_depth_reached
            ),
            "total_bytes": self.total_bytes,
            "page_limit_reached": (
                self.page_limit_reached
            ),
            "byte_limit_reached": (
                self.byte_limit_reached
            ),
        }

        return json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
        )

    @classmethod
    def from_json(
        cls,
        value: str | None,
    ) -> ResearchResult | None:
        if not value:
            return None

        try:
            payload = json.loads(value)
        except (
            TypeError,
            json.JSONDecodeError,
        ):
            return None

        if not isinstance(
            payload,
            dict,
        ):
            return None

        if (
            payload.get("kind")
            != "agentoflow.research_result"
        ):
            return None

        if (
            payload.get("schema_version")
            != 1
        ):
            return None

        raw_pages = payload.get(
            "pages"
        )

        if not isinstance(
            raw_pages,
            list,
        ):
            return None

        pages: list[ResearchPage] = []

        try:
            for raw_page in raw_pages:
                if not isinstance(
                    raw_page,
                    dict,
                ):
                    return None

                raw_links = raw_page.get(
                    "links",
                    [],
                )

                if not isinstance(
                    raw_links,
                    list,
                ):
                    return None

                pages.append(
                    ResearchPage(
                        url=str(
                            raw_page["url"]
                        ),
                        depth=int(
                            raw_page["depth"]
                        ),
                        title=str(
                            raw_page.get(
                                "title",
                                "",
                            )
                        ),
                        content=str(
                            raw_page.get(
                                "content",
                                "",
                            )
                        ),
                        links=tuple(
                            str(url)
                            for url in raw_links
                        ),
                        content_bytes=int(
                            raw_page[
                                "content_bytes"
                            ]
                        ),
                    )
                )

            return cls(
                root_url=str(
                    payload["root_url"]
                ),
                pages=tuple(pages),
                discovered_urls=tuple(
                    str(url)
                    for url in payload.get(
                        "discovered_urls",
                        [],
                    )
                ),
                failed_urls=tuple(
                    str(url)
                    for url in payload.get(
                        "failed_urls",
                        [],
                    )
                ),
                max_depth_reached=int(
                    payload.get(
                        "max_depth_reached",
                        0,
                    )
                ),
                total_bytes=int(
                    payload.get(
                        "total_bytes",
                        0,
                    )
                ),
                page_limit_reached=bool(
                    payload.get(
                        "page_limit_reached",
                        False,
                    )
                ),
                byte_limit_reached=bool(
                    payload.get(
                        "byte_limit_reached",
                        False,
                    )
                ),
            )

        except (
            KeyError,
            TypeError,
            ValueError,
        ):
            return None


@dataclass(slots=True, frozen=True)
class ResearchContract:
    """
    Explicit research requirements for a task.

    root_urls:
        Entry points that must be researched.

    required_urls:
        Specific URLs that must actually be fetched.

    min_pages:
        Minimum number of unique successfully fetched pages.

    min_depth:
        Minimum crawl depth that must actually be reached.

    require_all_roots:
        Whether every root URL must be fetched.
    """

    root_urls: tuple[str, ...]

    required_urls: tuple[str, ...] = ()

    min_pages: int = 1

    min_depth: int = 0

    require_all_roots: bool = True

    def __post_init__(self) -> None:
        if not self.root_urls:
            raise ValueError(
                "ResearchContract requires "
                "at least one root URL"
            )

        if self.min_pages < 1:
            raise ValueError(
                "min_pages must be >= 1"
            )

        if self.min_depth < 0:
            raise ValueError(
                "min_depth must be >= 0"
            )

        roots = tuple(
            dict.fromkeys(
                canonicalize_url(url)
                for url in self.root_urls
            )
        )

        required = tuple(
            dict.fromkeys(
                canonicalize_url(url)
                for url in self.required_urls
            )
        )

        object.__setattr__(
            self,
            "root_urls",
            roots,
        )

        object.__setattr__(
            self,
            "required_urls",
            required,
        )

    def is_satisfied(
        self,
        coverage: ResearchCoverage,
    ) -> bool:
        fetched = coverage.fetched_urls

        if len(fetched) < self.min_pages:
            return False

        if (
            coverage.max_depth_reached
            < self.min_depth
        ):
            return False

        if self.require_all_roots:
            if not set(
                self.root_urls
            ).issubset(fetched):
                return False

        if not set(
            self.required_urls
        ).issubset(fetched):
            return False

        return True


@dataclass(slots=True)
class ResearchCoverage:
    """
    Accumulated evidence over multiple crawl calls.
    """

    fetched_urls: set[str] = field(
        default_factory=set,
    )

    failed_urls: set[str] = field(
        default_factory=set,
    )

    discovered_urls: set[str] = field(
        default_factory=set,
    )

    max_depth_reached: int = 0

    total_bytes: int = 0

    def reset(self) -> None:
        self.fetched_urls.clear()

        self.failed_urls.clear()

        self.discovered_urls.clear()

        self.max_depth_reached = 0

        self.total_bytes = 0

    def merge(
        self,
        result: ResearchResult,
    ) -> None:
        for page in result.pages:
            url = canonicalize_url(
                page.url
            )

            if url in self.fetched_urls:
                continue

            self.fetched_urls.add(url)

            self.failed_urls.discard(url)

            self.total_bytes += (
                page.content_bytes
            )

            self.max_depth_reached = max(
                self.max_depth_reached,
                page.depth,
            )

        self.failed_urls.update(
            canonicalize_url(url)
            for url in result.failed_urls
        )

        self.failed_urls.difference_update(
            self.fetched_urls
        )

        self.discovered_urls.update(
            canonicalize_url(url)
            for url in result.discovered_urls
        )

        self.max_depth_reached = max(
            self.max_depth_reached,
            result.max_depth_reached,
        )
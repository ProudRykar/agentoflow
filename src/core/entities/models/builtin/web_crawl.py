from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import PurePosixPath
from urllib.parse import urljoin, urlsplit

import httpx

from core.entities.models.research_contract import (
    ResearchPage,
    ResearchResult,
    canonicalize_url,
)
from core.entities.models.tool import (
    Tool,
    ToolContext,
    ToolPolicy,
)


USER_AGENT = "agentoflow/0.1"

_HTML_CONTENT_TYPES = frozenset({
    "text/html",
    "application/xhtml+xml",
})


class WebFetchError(RuntimeError):
    pass


@dataclass(slots=True, frozen=True)
class WebFetchInput:
    url: str


@dataclass(slots=True, frozen=True)
class WebCrawlInput:
    url: str


@dataclass(slots=True, frozen=True)
class WebCrawlPolicy:
    """
    Trusted runtime limits.

    These are deliberately not exposed as LLM-controlled tool
    arguments.
    """

    max_depth: int = 2

    max_pages: int = 8

    max_total_bytes: int = 2_000_000

    max_page_bytes: int = 512_000

    max_content_chars: int = 10_000

    max_links_per_page: int = 64

    same_host_only: bool = True

    timeout: float = 30.0

    def __post_init__(self) -> None:
        if self.max_depth < 0:
            raise ValueError(
                "max_depth must be >= 0"
            )

        if self.max_pages < 1:
            raise ValueError(
                "max_pages must be >= 1"
            )

        if self.max_total_bytes < 1:
            raise ValueError(
                "max_total_bytes must be >= 1"
            )

        if self.max_page_bytes < 1:
            raise ValueError(
                "max_page_bytes must be >= 1"
            )

        if self.max_content_chars < 1:
            raise ValueError(
                "max_content_chars must be >= 1"
            )

        if self.max_links_per_page < 1:
            raise ValueError(
                "max_links_per_page must be >= 1"
            )

        if self.timeout <= 0:
            raise ValueError(
                "timeout must be > 0"
            )


@dataclass(slots=True, frozen=True)
class WebPage:
    url: str

    title: str

    content: str

    links: tuple[str, ...]

    status_code: int

    content_type: str

    content_bytes: int

    def __post_init__(self) -> None:
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

    def to_model_text(self) -> str:
        result = [
            f"URL: {self.url}",
            f"HTTP status: {self.status_code}",
        ]

        if self.title:
            result.append(
                f"Title: {self.title}"
            )

        result.extend(
            (
                "",
                "Content:",
                self.content,
            )
        )

        if self.links:
            result.extend(
                (
                    "",
                    "Discovered links:",
                )
            )

            result.extend(
                f"- {url}"
                for url in self.links
            )

        return "\n".join(result)


class _HTMLParser(HTMLParser):
    _BLOCK_TAGS = frozenset({
        "address",
        "article",
        "aside",
        "blockquote",
        "br",
        "dd",
        "div",
        "dl",
        "dt",
        "figcaption",
        "figure",
        "footer",
        "form",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "header",
        "hr",
        "li",
        "main",
        "nav",
        "ol",
        "p",
        "pre",
        "section",
        "table",
        "td",
        "th",
        "tr",
        "ul",
    })

    _IGNORED_TAGS = frozenset({
        "canvas",
        "noscript",
        "script",
        "style",
        "svg",
        "template",
    })

    def __init__(
        self,
        *,
        base_url: str,
    ) -> None:
        super().__init__(
            convert_charrefs=True,
        )

        self._base_url = base_url

        self._ignored_depth = 0

        self._title_depth = 0

        self._title_parts: list[str] = []

        self._text_parts: list[str] = []

        self._links: list[str] = []

        self._seen_links: set[str] = set()

    @property
    def title(self) -> str:
        return " ".join(
            "".join(
                self._title_parts
            ).split()
        )

    @property
    def text(self) -> str:
        raw = "".join(
            self._text_parts
        )

        lines = [
            " ".join(line.split())
            for line in raw.splitlines()
        ]

        return "\n".join(
            line
            for line in lines
            if line
        ).strip()

    @property
    def links(self) -> tuple[str, ...]:
        return tuple(
            self._links
        )

    def handle_starttag(
        self,
        tag: str,
        attrs,
    ) -> None:
        tag_lower = tag.lower()

        if (
            tag_lower
            in self._IGNORED_TAGS
        ):
            self._ignored_depth += 1
            return

        if self._ignored_depth:
            return

        if tag_lower == "title":
            self._title_depth += 1

        if tag_lower == "a":
            href: str | None = None
            rel = ""

            for key, value in attrs:
                key_lower = key.lower()

                if key_lower == "href":
                    href = value

                elif key_lower == "rel":
                    rel = value or ""

            rel_values = {
                item.strip().lower()
                for item in rel.split()
            }

            if (
                href
                and "nofollow"
                not in rel_values
            ):
                self._add_link(href)

        if (
            tag_lower
            in self._BLOCK_TAGS
        ):
            self._append_newline()

    def handle_endtag(
        self,
        tag: str,
    ) -> None:
        tag_lower = tag.lower()

        if (
            tag_lower
            in self._IGNORED_TAGS
        ):
            if self._ignored_depth:
                self._ignored_depth -= 1

            return

        if self._ignored_depth:
            return

        if (
            tag_lower == "title"
            and self._title_depth
        ):
            self._title_depth -= 1

        if (
            tag_lower
            in self._BLOCK_TAGS
        ):
            self._append_newline()

    def handle_startendtag(
        self,
        tag: str,
        attrs,
    ) -> None:
        self.handle_starttag(
            tag,
            attrs,
        )

        self.handle_endtag(
            tag
        )

    def handle_data(
        self,
        data: str,
    ) -> None:
        if self._ignored_depth:
            return

        if self._title_depth:
            self._title_parts.append(
                data
            )

            return

        self._text_parts.append(
            data
        )

    def _append_newline(self) -> None:
        if not self._text_parts:
            return

        if self._text_parts[-1].endswith(
            "\n"
        ):
            return

        self._text_parts.append(
            "\n"
        )

    def _add_link(
        self,
        href: str,
    ) -> None:
        resolved = urljoin(
            self._base_url,
            href.strip(),
        )

        try:
            normalized = canonicalize_url(
                resolved
            )
        except ValueError:
            return

        if (
            normalized
            in self._seen_links
        ):
            return

        self._seen_links.add(
            normalized
        )

        self._links.append(
            normalized
        )


class WebFetcher:
    """
    Atomic HTTP page fetcher.

    web_fetch and web_crawl both use this implementation.
    """

    def __init__(
        self,
        *,
        timeout: float = 30.0,
        max_redirects: int = 5,
    ) -> None:
        if timeout <= 0:
            raise ValueError(
                "timeout must be > 0"
            )

        if max_redirects < 0:
            raise ValueError(
                "max_redirects must be >= 0"
            )

        self._timeout = timeout

        self._max_redirects = (
            max_redirects
        )

    @property
    def timeout(self) -> float:
        return self._timeout

    async def fetch(
        self,
        url: str,
        *,
        max_bytes: int = 512_000,
        allowed_host: str | None = None,
    ) -> WebPage:
        current_url = (
            canonicalize_url(url)
        )

        if max_bytes <= 0:
            raise ValueError(
                "max_bytes must be > 0"
            )

        root_host = (
            allowed_host.lower()
            if allowed_host
            else None
        )

        timeout = httpx.Timeout(
            self._timeout,
            connect=min(
                self._timeout,
                10.0,
            ),
        )

        async with httpx.AsyncClient(
            headers={
                "User-Agent": USER_AGENT,
                "Accept": (
                    "text/html,"
                    "application/xhtml+xml;"
                    "q=0.9,"
                    "text/plain;q=0.8,"
                    "*/*;q=0.1"
                ),
                "Accept-Language": (
                    "en-US,en;q=0.8"
                ),
            },
            timeout=timeout,
            follow_redirects=False,
        ) as client:
            for redirect_index in range(
                self._max_redirects + 1
            ):
                try:
                    response = await client.get(
                        current_url,
                        follow_redirects=False,
                    )
                except httpx.HTTPError as exc:
                    raise WebFetchError(
                        "HTTP request failed for "
                        f"{current_url}: {exc}"
                    ) from exc

                if (
                    300
                    <= response.status_code
                    < 400
                ):
                    location = (
                        response.headers.get(
                            "location"
                        )
                    )

                    if not location:
                        raise WebFetchError(
                            "Redirect without "
                            "Location header: "
                            f"{current_url}"
                        )

                    if (
                        redirect_index
                        >= self._max_redirects
                    ):
                        raise WebFetchError(
                            "Too many redirects: "
                            f"{current_url}"
                        )

                    redirected = (
                        canonicalize_url(
                            urljoin(
                                current_url,
                                location,
                            )
                        )
                    )

                    if root_host is not None:
                        redirected_host = (
                            urlsplit(
                                redirected
                            ).hostname
                        )

                        if (
                            redirected_host
                            != root_host
                        ):
                            raise WebFetchError(
                                "Redirect leaves "
                                "allowed host: "
                                f"{redirected}"
                            )

                    current_url = (
                        redirected
                    )

                    continue

                if not (
                    200
                    <= response.status_code
                    < 300
                ):
                    raise WebFetchError(
                        f"HTTP "
                        f"{response.status_code} "
                        f"for {current_url}"
                    )

                content_type = (
                    response.headers
                    .get(
                        "content-type",
                        "",
                    )
                    .split(
                        ";",
                        1,
                    )[0]
                    .strip()
                    .lower()
                )

                content_length = (
                    response.headers.get(
                        "content-length"
                    )
                )

                if content_length:
                    try:
                        declared_size = int(
                            content_length
                        )
                    except ValueError:
                        declared_size = 0

                    if (
                        declared_size
                        > max_bytes
                    ):
                        raise WebFetchError(
                            "Response exceeds "
                            "page size limit: "
                            f"{declared_size} "
                            f"> {max_bytes} bytes"
                        )

                body = bytearray()

                async for chunk in (
                    response.aiter_bytes()
                ):
                    body.extend(chunk)

                    if (
                        len(body)
                        > max_bytes
                    ):
                        raise WebFetchError(
                            "Response exceeds "
                            "page size limit: "
                            f"> {max_bytes} bytes"
                        )

                raw = bytes(body)

                encoding = (
                    response.encoding
                    or "utf-8"
                )

                text = raw.decode(
                    encoding,
                    errors="replace",
                )

                is_html = (
                    content_type
                    in _HTML_CONTENT_TYPES
                    or text.lstrip()
                    .lower()
                    .startswith(
                        (
                            "<!doctype html",
                            "<html",
                        )
                    )
                )

                if is_html:
                    parser = _HTMLParser(
                        base_url=str(
                            response.url
                        ),
                    )

                    try:
                        parser.feed(text)
                        parser.close()
                    except Exception as exc:
                        raise WebFetchError(
                            "HTML parsing failed "
                            f"for {current_url}: "
                            f"{exc}"
                        ) from exc

                    return WebPage(
                        url=str(
                            response.url
                        ),
                        title=parser.title,
                        content=parser.text,
                        links=parser.links,
                        status_code=(
                            response.status_code
                        ),
                        content_type=(
                            content_type
                            or "text/html"
                        ),
                        content_bytes=len(
                            raw
                        ),
                    )

                if content_type.startswith(
                    "text/"
                ):
                    plain_text = "\n".join(
                        line.strip()
                        for line
                        in text.splitlines()
                        if line.strip()
                    )
                else:
                    plain_text = (
                        "Non-HTML response: "
                        f"{content_type or 'unknown'}"
                    )

                return WebPage(
                    url=str(
                        response.url
                    ),
                    title="",
                    content=plain_text,
                    links=(),
                    status_code=(
                        response.status_code
                    ),
                    content_type=content_type,
                    content_bytes=len(raw),
                )

        raise WebFetchError(
            f"Fetch failed for {current_url}"
        )


class WebCrawler:
    """
    Deterministic bounded breadth-first crawler.

    The LLM controls only the root URL.

    Crawl limits and same-host rules belong to trusted runtime policy.
    """

    _SKIP_EXTENSIONS = frozenset({
        ".7z",
        ".avi",
        ".bmp",
        ".css",
        ".csv",
        ".doc",
        ".docx",
        ".gif",
        ".gz",
        ".ico",
        ".jpeg",
        ".jpg",
        ".js",
        ".m4a",
        ".m4v",
        ".mkv",
        ".mov",
        ".mp3",
        ".mp4",
        ".mpeg",
        ".pdf",
        ".png",
        ".svg",
        ".tar",
        ".tgz",
        ".tif",
        ".tiff",
        ".wav",
        ".webm",
        ".webp",
        ".woff",
        ".woff2",
        ".xls",
        ".xlsx",
        ".xml",
        ".zip",
    })

    def __init__(
        self,
        fetcher: WebFetcher | None = None,
        policy: WebCrawlPolicy | None = None,
    ) -> None:
        self._policy = (
            policy
            or WebCrawlPolicy()
        )

        self._fetcher = (
            fetcher
            or WebFetcher(
                timeout=(
                    self._policy.timeout
                ),
            )
        )

    @property
    def policy(self) -> WebCrawlPolicy:
        return self._policy

    async def crawl(
        self,
        root_url: str,
    ) -> ResearchResult:
        root = canonicalize_url(
            root_url
        )

        root_host = urlsplit(
            root
        ).hostname

        if root_host is None:
            raise ValueError(
                "Root URL must contain "
                "a host"
            )

        queue: deque[
            tuple[str, int]
        ] = deque(
            [(root, 0)]
        )

        queued: set[str] = {
            root
        }

        attempted: set[str] = set()

        pages: list[
            ResearchPage
        ] = []

        discovered: set[str] = set()

        failed: list[str] = []

        total_bytes = 0

        max_depth_reached = 0

        page_limit_reached = False

        byte_limit_reached = False

        while (
            queue
            and len(attempted)
            < self._policy.max_pages
        ):
            url, depth = (
                queue.popleft()
            )

            if url in attempted:
                continue

            remaining_bytes = (
                self._policy
                .max_total_bytes
                - total_bytes
            )

            if remaining_bytes <= 0:
                byte_limit_reached = True
                break

            attempted.add(url)

            try:
                page = await (
                    self._fetcher.fetch(
                        url,
                        max_bytes=min(
                            self._policy
                            .max_page_bytes,
                            remaining_bytes,
                        ),
                        allowed_host=(
                            root_host
                            if self._policy
                            .same_host_only
                            else None
                        ),
                    )
                )
            except (
                WebFetchError,
                ValueError,
            ):
                failed.append(url)
                continue

            if (
                page.content_bytes
                > remaining_bytes
            ):
                byte_limit_reached = True
                break

            total_bytes += (
                page.content_bytes
            )

            max_depth_reached = max(
                max_depth_reached,
                depth,
            )

            links = self._select_links(
                page.links,
                root_host=root_host,
            )

            discovered.update(links)

            pages.append(
                ResearchPage(
                    url=page.url,
                    depth=depth,
                    title=page.title,
                    content=page.content[
                        : self._policy
                        .max_content_chars
                    ],
                    links=links,
                    content_bytes=(
                        page.content_bytes
                    ),
                )
            )

            if (
                depth
                >= self._policy.max_depth
            ):
                continue

            for link in links:
                if (
                    link in attempted
                    or link in queued
                ):
                    continue

                if (
                    len(attempted)
                    + len(queue)
                    >= self._policy.max_pages
                ):
                    page_limit_reached = True
                    break

                queued.add(link)

                queue.append(
                    (
                        link,
                        depth + 1,
                    )
                )

        if (
            queue
            and len(attempted)
            >= self._policy.max_pages
        ):
            page_limit_reached = True

        if (
            queue
            and total_bytes
            >= self._policy.max_total_bytes
        ):
            byte_limit_reached = True

        return ResearchResult(
            root_url=root,
            pages=tuple(pages),
            discovered_urls=tuple(
                sorted(discovered)
            ),
            failed_urls=tuple(failed),
            max_depth_reached=(
                max_depth_reached
            ),
            total_bytes=total_bytes,
            page_limit_reached=(
                page_limit_reached
            ),
            byte_limit_reached=(
                byte_limit_reached
            ),
        )

    def _select_links(
        self,
        links: tuple[str, ...],
        *,
        root_host: str,
    ) -> tuple[str, ...]:
        selected: list[str] = []

        for link in links:
            try:
                normalized = (
                    canonicalize_url(link)
                )
            except ValueError:
                continue

            parsed = urlsplit(
                normalized
            )

            host = parsed.hostname

            if host is None:
                continue

            if (
                self._policy.same_host_only
                and host != root_host
            ):
                continue

            suffix = PurePosixPath(
                parsed.path
            ).suffix.lower()

            if suffix in (
                self._SKIP_EXTENSIONS
            ):
                continue

            if normalized in selected:
                continue

            selected.append(
                normalized
            )

            if (
                len(selected)
                >= self._policy
                .max_links_per_page
            ):
                break

        return tuple(selected)


def create_web_fetch_tool(
    fetcher: WebFetcher,
) -> Tool:
    async def handler(
        arguments: WebFetchInput,
        context: ToolContext,
    ) -> str:
        del context

        page = await fetcher.fetch(
            arguments.url,
        )

        return page.to_model_text()

    return Tool(
        name="web_fetch",
        description=(
            "Fetch one HTTP or HTTPS "
            "page and return its "
            "extracted content and "
            "discovered links."
        ),
        input_type=WebFetchInput,
        handler=handler,
        policy=ToolPolicy(
            permissions=frozenset({
                "web.fetch",
            }),
            timeout=fetcher.timeout,
            max_output_size=20_000,
        ),
    )


def create_web_crawl_tool(
    crawler: WebCrawler,
) -> Tool:
    async def handler(
        arguments: WebCrawlInput,
        context: ToolContext,
    ) -> str:
        del context

        result = await crawler.crawl(
            arguments.url,
        )

        return result.to_json()

    return Tool(
        name="web_crawl",
        description=(
            "Run a bounded breadth-first "
            "crawl from one URL. Follow "
            "same-host HTML links within "
            "the trusted runtime limits "
            "and return structured "
            "research evidence."
        ),
        input_type=WebCrawlInput,
        handler=handler,
        policy=ToolPolicy(
            permissions=frozenset({
                "web.fetch",
            }),
            timeout=(
                crawler.policy.timeout
                * crawler.policy.max_pages
            ),
            max_output_size=120_000,
        ),
    )
from __future__ import annotations

import asyncio
from dataclasses import dataclass

from agent_workflow.core.entities.models.builtin.web_crawl import (
    WebFetcher,
    WebPage,
)
from agent_workflow.core.entities.models.tool import (
    Tool,
    ToolContext,
    ToolPolicy,
)


class WebBatchError(RuntimeError):
    """The batch could not be started at all."""


@dataclass(slots=True, frozen=True)
class WebFetchManyInput:
    urls: tuple[str, ...]
    """Two or more URLs to download at the same time."""

    max_pages: int = 0
    """Fewer pages than the runtime allows. 0 means the maximum."""


@dataclass(slots=True, frozen=True)
class WebBatchPolicy:
    """Trusted runtime limits for parallel fetching.

    Concurrency is capped so that a batch cannot be used to hammer a
    third-party site or to exhaust the agent's own sockets and file
    descriptors.
    """

    max_pages: int = 8

    max_concurrency: int = 4

    max_chars_per_page: int = 6_000

    max_total_chars: int = 40_000

    max_page_bytes: int = 512_000

    timeout: float = 25.0

    def __post_init__(self) -> None:
        if self.max_pages < 1:
            raise ValueError("max_pages must be >= 1")

        if self.max_concurrency < 1:
            raise ValueError(
                "max_concurrency must be >= 1"
            )

        if self.max_chars_per_page < 1:
            raise ValueError(
                "max_chars_per_page must be >= 1"
            )

        if self.max_total_chars < 1:
            raise ValueError(
                "max_total_chars must be >= 1"
            )

        if self.max_page_bytes < 1:
            raise ValueError(
                "max_page_bytes must be >= 1"
            )

        if self.timeout <= 0:
            raise ValueError("timeout must be > 0")


@dataclass(slots=True, frozen=True)
class FetchedPage:
    url: str

    final_url: str = ""

    title: str = ""

    text: str = ""

    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error


class WebBatchFetcher:
    """Downloads several pages concurrently.

    Each URL is fetched independently and its failure is recorded on
    that page instead of failing the batch: one dead link out of six
    should not throw away the other five. Concurrency is bounded and
    the total output is capped so the result stays inside the tool's
    output budget.
    """

    def __init__(
        self,
        policy: WebBatchPolicy | None = None,
        *,
        fetcher: WebFetcher | None = None,
    ) -> None:
        self._policy = policy or WebBatchPolicy()
        self._fetcher = fetcher or WebFetcher(
            timeout=self._policy.timeout
        )

    @property
    def policy(self) -> WebBatchPolicy:
        return self._policy

    async def fetch_many(
        self,
        urls: tuple[str, ...] | list[str],
        *,
        max_pages: int | None = None,
    ) -> list[FetchedPage]:
        limit = self._policy.max_pages

        if max_pages is not None and max_pages > 0:
            limit = min(max_pages, limit)

        unique: list[str] = []
        seen: set[str] = set()

        for url in urls:
            cleaned = url.strip().split("#", 1)[0]

            if not cleaned or cleaned in seen:
                continue

            seen.add(cleaned)
            unique.append(cleaned)

            if len(unique) >= limit:
                break

        if not unique:
            raise WebBatchError(
                "No URLs supplied"
            )

        semaphore = asyncio.Semaphore(
            min(
                self._policy.max_concurrency,
                len(unique),
            )
        )

        async def one(url: str) -> FetchedPage:
            async with semaphore:
                return await self._fetch_one(url)

        # gather without return_exceptions would discard the pages
        # that did succeed, so failures are collected as values.
        return list(
            await asyncio.gather(
                *(one(url) for url in unique)
            )
        )

    async def _fetch_one(self, url: str) -> FetchedPage:
        try:
            page: WebPage = await self._fetcher.fetch(
                url,
                max_bytes=self._policy.max_page_bytes,
            )
        except Exception as exc:
            # Includes WebPolicyError: a URL that is refused is
            # reported as a failed page, so the batch still returns
            # the pages that were allowed.
            return FetchedPage(
                url=url,
                error=str(exc),
            )

        return FetchedPage(
            url=url,
            final_url=page.url,
            title=page.title,
            text=page.content[
                : self._policy.max_chars_per_page
            ],
        )


def format_pages(
    pages: list[FetchedPage],
    *,
    query: str = "",
    max_total_chars: int = 40_000,
) -> str:
    """Render a batch for the model, keeping the source of every
    block visible so a claim can be traced back to a URL."""

    header = f'Search: "{query}"' if query else ""
    succeeded = [page for page in pages if page.ok]
    failed = [page for page in pages if not page.ok]

    lines = [
        part
        for part in (
            header,
            (
                f"Pages: {len(succeeded)} fetched, "
                f"{len(failed)} failed"
            ),
        )
        if part
    ]

    remaining = max_total_chars

    for index, page in enumerate(succeeded, start=1):
        if remaining <= 0:
            lines.append("")
            lines.append(
                "[output budget reached; "
                "remaining pages omitted]"
            )
            break

        body = page.text[:remaining]
        remaining -= len(body)

        lines.append("")
        lines.append(
            f"--- [{index}] {page.title or page.final_url or page.url}"
        )

        if page.final_url and page.final_url != page.url:
            lines.append(f"    {page.final_url}")

        lines.append("")
        lines.append(body.strip() or "(no text content)")

    if failed:
        lines.append("")
        lines.append("Failed:")
        lines.append("")

        for page in failed:
            lines.append(f"  {page.url}")
            lines.append(f"    {page.error}")

    return "\n".join(lines).rstrip()


def create_web_fetch_many_tool(
    batch: WebBatchFetcher,
) -> Tool:
    async def handler(
        arguments: WebFetchManyInput,
        context: ToolContext,
    ) -> str:
        del context

        pages = await batch.fetch_many(
            arguments.urls,
            max_pages=arguments.max_pages or None,
        )

        return format_pages(
            pages,
            max_total_chars=batch.policy.max_total_chars,
        )

    return Tool(
        name="web_fetch_many",
        description=(
            "Download several web pages at the same time and return "
            "their extracted text. Use it after web_search to read "
            "several sources in one call instead of fetching them "
            "one at a time. URLs may be on different hosts. Every "
            "URL is still checked against the network policy, and a "
            "URL that fails is reported without discarding the pages "
            "that succeeded."
        ),
        input_type=WebFetchManyInput,
        handler=handler,
        policy=ToolPolicy(
            permissions=frozenset({
                "web.fetch",
            }),
            timeout=(
                batch.policy.timeout
                * batch.policy.max_concurrency
                + 10.0
            ),
            max_output_size=120_000,
        ),
    )
from __future__ import annotations

from dataclasses import dataclass

from agent_workflow.core.entities.models.tool import (
    Tool,
    ToolContext,
    ToolPolicy,
)
from agent_workflow.core.entities.models.web_batch import (
    FetchedPage,
    WebBatchFetcher,
    WebBatchPolicy,
    format_pages,
)
from agent_workflow.core.entities.models.web_search import (
    DuckDuckGoSearch,
    SearchResult,
    WebSearchError,
    WebSearchPolicy,
)


class WebResearchError(RuntimeError):
    """The research step could not be completed."""


@dataclass(slots=True, frozen=True)
class WebResearchInput:
    query: str

    max_pages: int = 0
    """How many top results to actually download."""


@dataclass(slots=True, frozen=True)
class WebResearchPolicy:
    """Trusted runtime limits for the combined search-then-read step."""

    search: WebSearchPolicy = WebSearchPolicy()

    batch: WebBatchPolicy = WebBatchPolicy()

    def __post_init__(self) -> None:
        if self.batch.max_pages > self.search.max_results:
            raise ValueError(
                "batch.max_pages cannot exceed "
                "search.max_results: there is nothing to fetch "
                "beyond the number of results returned"
            )


class WebResearcher:
    """Search, then read the top results concurrently.

    The composition the model would otherwise have to drive by hand
    across two tool calls: it returns the search results together
    with the pages it managed to download, so the model can see what
    was found even when some sources could not be read.
    """

    def __init__(
        self,
        search: DuckDuckGoSearch | None = None,
        batch: WebBatchFetcher | None = None,
        *,
        policy: WebResearchPolicy | None = None,
    ) -> None:
        self._policy = policy or WebResearchPolicy()
        self._search = search or DuckDuckGoSearch(
            self._policy.search
        )
        self._batch = batch or WebBatchFetcher(
            self._policy.batch
        )

    @property
    def policy(self) -> WebResearchPolicy:
        return self._policy

    async def research(
        self,
        query: str,
        *,
        max_pages: int | None = None,
    ) -> tuple[list[SearchResult], list[FetchedPage]]:
        try:
            results = await self._search.search(query)
        except WebSearchError as exc:
            raise WebResearchError(str(exc)) from exc

        if not results:
            return results, []

        limit = len(results)

        if max_pages is not None and max_pages > 0:
            limit = min(max_pages, limit)

        chosen = results[:limit]

        pages = await self._batch.fetch_many(
            [result.url for result in chosen]
        )

        return chosen, pages


def create_web_research_tool(
    researcher: WebResearcher,
) -> Tool:
    async def handler(
        arguments: WebResearchInput,
        context: ToolContext,
    ) -> str:
        del context

        results, pages = await researcher.research(
            arguments.query,
            max_pages=arguments.max_pages or None,
        )

        if not results:
            return (
                f'No search results for '
                f'"{arguments.query}". Try different or '
                "broader wording, or use web_fetch on a URL "
                "you already know."
            )

        fetched = sum(1 for page in pages if page.ok)

        listing = "\n".join(
            f"[{index}] {result.title or '(no title)'}"
            f"\n    {result.url}"
            for index, result in enumerate(results, start=1)
        )

        body = format_pages(
            pages,
            query=arguments.query,
            max_total_chars=(
                researcher.policy.batch.max_total_chars
            ),
        )

        return (
            f'Search "{arguments.query}" returned '
            f"{len(results)} results; "
            f"{fetched} pages downloaded.\n\n"
            f"All results:\n{listing}\n\n{body}"
        )

    return Tool(
        name="web_research",
        description=(
            "Search the web and read the top results in one call. "
            "Runs a search, then downloads the best matches "
            "concurrently and returns their extracted text "
            "alongside the full result list. Use it for a question "
            "that needs outside information; use web_search alone "
            "when you only need links, and web_fetch_many when you "
            "already have the URLs. The pages are returned as text "
            "for you to read and reason about; no answer is "
            "pre-computed."
        ),
        input_type=WebResearchInput,
        handler=handler,
        policy=ToolPolicy(
            permissions=frozenset({
                "web.fetch",
                "web.search",
            }),
            timeout=(
                researcher.policy.search.timeout
                + researcher.policy.batch.timeout
                * researcher.policy.batch.max_concurrency
                + 10.0
            ),
            max_output_size=120_000,
        ),
    )
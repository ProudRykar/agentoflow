"""The web research tools: search, parallel fetch, and the pair.

These tests never touch the network. The search backend is driven
through an injected client and the fetcher through an injected
policy, because a test that depends on DuckDuckGo or on a live site
is a test that fails for reasons unrelated to the code.
"""

from __future__ import annotations

import asyncio
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import httpx
import pytest

from agent_workflow.core.entities.models.builtin.web_crawl import (
    WebFetcher,
    WebPage,
)
from agent_workflow.core.entities.models.web_batch import (
    FetchedPage,
    WebBatchError,
    WebBatchFetcher,
    WebBatchPolicy,
    format_pages,
)
from agent_workflow.core.entities.models.web_policy import (
    WebPolicy,
    WebPolicyError,
)
from agent_workflow.core.entities.models.web_research import (
    WebResearchPolicy,
    WebResearcher,
    create_web_research_tool,
)
from agent_workflow.core.entities.models.web_search import (
    DuckDuckGoSearch,
    WebSearchError,
    WebSearchPolicy,
    create_web_search_tool,
    format_results,
)


# ======================================================================
# Fixtures and doubles
# ======================================================================


RESULT_PAGE = """
<html><body>
<table>
<tr>
  <td>
    <a rel="nofollow" class="result-link"
       href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fone">
      First <b>result</b>
    </a>
  </td>
  <td class='result-snippet'>Snippet about one.</td>
</tr>
<tr>
  <td>
    <a rel="nofollow" class="result-link"
       href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.org%2Ftwo">
      Second result
    </a>
  </td>
  <td class='result-snippet'>Snippet about two.</td>
</tr>
<tr>
  <td>
    <a rel="nofollow" class="result-link"
       href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.net%2Fthree">
      Third result
    </a>
  </td>
  <td class='result-snippet'>Snippet about three.</td>
</tr>
</table>
</body></html>
"""


class _StubClient:
    """Minimal stand-in for httpx.AsyncClient.get."""

    def __init__(
        self,
        text: str,
        status_code: int = 200,
    ) -> None:
        self._text = text
        self._status_code = status_code
        self.calls: list[dict] = []

    async def get(self, url: str, **kwargs) -> httpx.Response:
        self.calls.append({"url": url, **kwargs})

        request = httpx.Request("GET", url)

        return httpx.Response(
            self._status_code,
            text=self._text,
            request=request,
        )


class _PermissiveEntryPolicy(WebPolicy):
    """Allows exactly one origin, delegating everything else.

    Used to prove the redirect target is validated on its own. The
    allowlist is an origin rather than a hostname so that a redirect
    to the same host on a different port still reaches the real
    policy and is refused there.
    """

    def __init__(self, allowed_origin: str) -> None:
        self._allowed = allowed_origin

    async def validate_url(self, url: str) -> None:
        from urllib.parse import urlparse

        parsed = urlparse(url)
        port = parsed.port or (
            443 if parsed.scheme == "https" else 80
        )

        if (
            f"{parsed.scheme}://{parsed.hostname}:{port}"
            == self._allowed
        ):
            return

        await super().validate_url(url)


class _StubFetcher:
    """Returns canned pages, optionally failing some."""

    def __init__(
        self,
        pages: dict[str, str],
        *,
        errors: dict[str, str] | None = None,
        delay: float = 0.0,
    ) -> None:
        self._pages = pages
        self._errors = errors or {}
        self._delay = delay
        self.calls: list[str] = []

    async def fetch(
        self,
        url: str,
        *,
        max_bytes: int = 512_000,
    ) -> WebPage:
        self.calls.append(url)

        if self._delay:
            await asyncio.sleep(self._delay)

        if url in self._errors:
            raise WebFetchFailure(self._errors[url])

        return WebPage(
            url=url,
            title=f"Title for {url}",
            content=f"Content of {url}",
            links=(),
            status_code=200,
            content_type="text/html",
            content_bytes=10,
        )


class WebFetchFailure(RuntimeError):
    pass


def _origin(url: str) -> str:
    from urllib.parse import urlparse

    parsed = urlparse(url)
    port = parsed.port or (443 if parsed.scheme == "https" else 80)

    return f"{parsed.scheme}://{parsed.hostname}:{port}"


def _redirect_server(
    location: str,
    status: int = 302,
) -> tuple[HTTPServer, str]:
    """Serve one redirect so redirect handling can be tested."""

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            self.send_response(status)
            self.send_header("Location", location)
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, *args) -> None:
            return

    server = HTTPServer(("127.0.0.1", 0), Handler)

    threading.Thread(
        target=server.serve_forever,
        daemon=True,
    ).start()

    host, port = server.server_address[0], server.server_address[1]

    return server, f"http://{host}:{port}/"


# ======================================================================
# Search parsing
# ======================================================================


async def test_search_parses_titles_snippets_and_urls() -> None:
    search = DuckDuckGoSearch(
        WebSearchPolicy(max_results=10),
        client=_StubClient(RESULT_PAGE),  # type: ignore[arg-type]
    )

    results = await search.search("anything")

    assert len(results) == 3
    assert results[0].url == "https://example.com/one"
    assert results[0].title == "First result"
    assert results[0].snippet == "Snippet about one."
    assert results[2].url == "https://example.net/three"


async def test_search_respects_the_result_limit() -> None:
    search = DuckDuckGoSearch(
        WebSearchPolicy(max_results=2),
        client=_StubClient(RESULT_PAGE),  # type: ignore[arg-type]
    )

    assert len(await search.search("q")) == 2


async def test_caller_cannot_raise_the_limit() -> None:
    search = DuckDuckGoSearch(
        WebSearchPolicy(max_results=1),
        client=_StubClient(RESULT_PAGE),  # type: ignore[arg-type]
    )

    assert len(await search.search("q", max_results=99)) == 1


async def test_search_deduplicates_and_strips_fragments() -> None:
    page = RESULT_PAGE.replace(
        "https%3A%2F%2Fexample.com%2Fone",
        "https%3A%2F%2Fexample.com%2Fone%23section",
    )

    client = _StubClient(page)
    search = DuckDuckGoSearch(
        client=client,  # type: ignore[arg-type]
    )

    results = await search.search("q")

    assert results[0].url == "https://example.com/one"


async def test_empty_query_is_refused() -> None:
    search = DuckDuckGoSearch(client=_StubClient(RESULT_PAGE))  # type: ignore[arg-type]

    with pytest.raises(WebSearchError, match="empty"):
        await search.search("   ")


async def test_long_query_is_truncated_not_rejected() -> None:
    client = _StubClient(RESULT_PAGE)
    search = DuckDuckGoSearch(
        WebSearchPolicy(max_query_chars=20),
        client=client,  # type: ignore[arg-type]
    )

    await search.search("a" * 500)

    sent = client.calls[0]["params"]["q"]

    assert len(sent) <= 20


async def test_no_results_is_not_an_error() -> None:
    search = DuckDuckGoSearch(
        client=_StubClient("<html>nothing here</html>"),  # type: ignore[arg-type]
    )

    assert await search.search("q") == []


async def test_antibot_page_is_reported_clearly() -> None:
    # A silent empty list would read to the model as "no matches",
    # and it would retry instead of telling the user search is down.
    blocked = (
        "<html><script src='/anomaly.js'></script>"
        "<p>please complete the following challenge</p></html>"
    )

    search = DuckDuckGoSearch(client=_StubClient(blocked))  # type: ignore[arg-type]

    with pytest.raises(WebSearchError, match="anti-bot"):
        await search.search("q")


async def test_http_error_is_reported() -> None:
    search = DuckDuckGoSearch(
        client=_StubClient("", status_code=503),  # type: ignore[arg-type]
    )

    with pytest.raises(WebSearchError, match="503"):
        await search.search("q")


async def test_safe_search_flag_is_sent_when_enabled() -> None:
    client = _StubClient(RESULT_PAGE)
    search = DuckDuckGoSearch(
        WebSearchPolicy(safe_search=True),
        client=client,  # type: ignore[arg-type]
    )

    await search.search("q")

    assert client.calls[0]["params"]["kp"] == "1"


def test_results_are_formatted_with_numbers_and_urls() -> None:
    rendered = format_results("q", _parse(RESULT_PAGE))

    assert "[1]" in rendered
    assert "https://example.com/one" in rendered


def _parse(payload: str):
    search = DuckDuckGoSearch(
        client=_StubClient(payload),  # type: ignore[arg-type]
    )

    return search._parse(payload, limit=10)


def test_empty_result_message_suggests_a_next_step() -> None:
    rendered = format_results("obscure query", [])

    assert "No search results" in rendered
    assert "web_fetch" in rendered


def test_policy_rejects_nonsense_values() -> None:
    with pytest.raises(ValueError, match="max_results"):
        WebSearchPolicy(max_results=0)

    with pytest.raises(ValueError, match="timeout"):
        WebSearchPolicy(timeout=0)


# ======================================================================
# Parallel batch fetch
# ======================================================================


async def test_batch_fetches_every_url() -> None:
    fetcher = _StubFetcher({})
    batch = WebBatchFetcher(fetcher=fetcher)  # type: ignore[arg-type]

    pages = await batch.fetch_many(
        ["https://a.test/1", "https://a.test/2"]
    )

    assert len(pages) == 2
    assert all(page.ok for page in pages)
    assert len(fetcher.calls) == 2


async def test_batch_deduplicates_urls() -> None:
    fetcher = _StubFetcher({})
    batch = WebBatchFetcher(fetcher=fetcher)  # type: ignore[arg-type]

    pages = await batch.fetch_many(
        ["https://a.test/1", "https://a.test/1", "https://a.test/1"]
    )

    assert len(pages) == 1


async def test_batch_truncates_at_the_policy_limit() -> None:
    fetcher = _StubFetcher({})
    batch = WebBatchFetcher(
        WebBatchPolicy(max_pages=2),
        fetcher=fetcher,  # type: ignore[arg-type]
    )

    urls = [f"https://a.test/{index}" for index in range(5)]

    assert len(await batch.fetch_many(urls)) == 2


async def test_batch_respects_the_caller_limit() -> None:
    batch = WebBatchFetcher(
        WebBatchPolicy(max_pages=5),
        fetcher=_StubFetcher({}),  # type: ignore[arg-type]
    )

    urls = [f"https://a.test/{index}" for index in range(5)]

    assert len(await batch.fetch_many(urls, max_pages=1)) == 1


async def test_one_failure_does_not_discard_the_others() -> None:
    # The whole point of the batch: a dead link out of three must not
    # throw away the two pages that worked.
    batch = WebBatchFetcher(
        fetcher=_StubFetcher(  # type: ignore[arg-type]
            {},
            errors={"https://a.test/bad": "HTTP 500"},
        )
    )

    pages = await batch.fetch_many(
        ["https://a.test/ok", "https://a.test/bad"]
    )

    by_url = {page.url: page for page in pages}

    assert by_url["https://a.test/ok"].ok
    assert not by_url["https://a.test/bad"].ok
    assert "HTTP 500" in by_url["https://a.test/bad"].error


async def test_batch_really_runs_concurrently() -> None:
    # Sequential execution would take 3 * delay.
    batch = WebBatchFetcher(
        WebBatchPolicy(max_concurrency=3),
        fetcher=_StubFetcher({}, delay=0.2),  # type: ignore[arg-type]
    )

    loop = asyncio.get_running_loop()
    started = loop.time()

    await batch.fetch_many(
        [
            "https://a.test/1",
            "https://a.test/2",
            "https://a.test/3",
        ]
    )

    elapsed = loop.time() - started

    assert elapsed < 0.5


async def test_batch_respects_the_concurrency_cap() -> None:
    batch = WebBatchFetcher(
        WebBatchPolicy(max_concurrency=1),
        fetcher=_StubFetcher({}, delay=0.05),  # type: ignore[arg-type]
    )

    loop = asyncio.get_running_loop()
    started = loop.time()

    await batch.fetch_many(
        [
            "https://a.test/1",
            "https://a.test/2",
            "https://a.test/3",
        ]
    )

    # Serialised by the semaphore, so at least 2 * delay.
    assert loop.time() - started >= 0.1


async def test_batch_truncates_page_text() -> None:
    fetcher = _StubFetcher({})
    batch = WebBatchFetcher(
        WebBatchPolicy(max_chars_per_page=5),
        fetcher=fetcher,  # type: ignore[arg-type]
    )

    pages = await batch.fetch_many(["https://a.test/1"])

    assert len(pages[0].text) == 5


async def test_batch_requires_at_least_one_url() -> None:
    batch = WebBatchFetcher(fetcher=_StubFetcher({}))  # type: ignore[arg-type]

    with pytest.raises(WebBatchError, match="No URLs"):
        await batch.fetch_many([])


def test_batch_policy_rejects_nonsense_values() -> None:
    with pytest.raises(ValueError, match="max_pages"):
        WebBatchPolicy(max_pages=0)

    with pytest.raises(ValueError, match="max_concurrency"):
        WebBatchPolicy(max_concurrency=0)


# ======================================================================
# Formatting
# ======================================================================


def test_format_lists_failures_separately() -> None:
    pages = [
        FetchedPage(url="https://a.test/ok", text="body"),
        FetchedPage(url="https://a.test/bad", error="boom"),
    ]

    rendered = format_pages(pages, query="q")

    assert "1 fetched, 1 failed" in rendered
    assert "https://a.test/bad" in rendered
    assert "boom" in rendered
    assert "body" in rendered


def test_format_respects_the_total_budget() -> None:
    pages = [
        FetchedPage(url=f"https://a.test/{index}", text="x" * 100)
        for index in range(5)
    ]

    rendered = format_pages(pages, max_total_chars=150)

    assert "output budget reached" in rendered


def test_format_notes_a_redirect() -> None:
    pages = [
        FetchedPage(
            url="https://a.test/old",
            final_url="https://a.test/new",
            text="body",
        )
    ]

    assert "https://a.test/new" in format_pages(pages)


def test_format_reports_an_empty_page() -> None:
    rendered = format_pages(
        [FetchedPage(url="https://a.test/1", text="")],
        query="q",
    )

    assert "no text content" in rendered


# ======================================================================
# Research composition
# ======================================================================


async def test_research_searches_then_downloads() -> None:
    search = DuckDuckGoSearch(
        client=_StubClient(RESULT_PAGE),  # type: ignore[arg-type]
    )
    batch = WebBatchFetcher(fetcher=_StubFetcher({}))  # type: ignore[arg-type]

    results, pages = await WebResearcher(search, batch).research("q")

    assert len(results) == 3
    assert len(pages) == 3
    assert [page.url for page in pages] == [
        result.url for result in results
    ]


async def test_research_honours_max_pages() -> None:
    search = DuckDuckGoSearch(
        client=_StubClient(RESULT_PAGE),  # type: ignore[arg-type]
    )
    batch = WebBatchFetcher(fetcher=_StubFetcher({}))  # type: ignore[arg-type]

    results, pages = await WebResearcher(search, batch).research(
        "q",
        max_pages=1,
    )

    assert len(results) == 1
    assert len(pages) == 1


async def test_research_reports_pages_that_failed_to_download() -> None:
    search = DuckDuckGoSearch(
        client=_StubClient(RESULT_PAGE),  # type: ignore[arg-type]
    )
    batch = WebBatchFetcher(
        fetcher=_StubFetcher(  # type: ignore[arg-type]
            {},
            errors={"https://example.org/two": "HTTP 403"},
        )
    )

    _, pages = await WebResearcher(search, batch).research("q")

    failed = [page for page in pages if not page.ok]

    assert len(failed) == 1
    assert "403" in failed[0].error


async def test_research_with_no_results_returns_nothing() -> None:
    search = DuckDuckGoSearch(
        client=_StubClient("<html>empty</html>"),  # type: ignore[arg-type]
    )
    batch = WebBatchFetcher(fetcher=_StubFetcher({}))  # type: ignore[arg-type]

    results, pages = await WebResearcher(search, batch).research("q")

    assert results == []
    assert pages == []


async def test_research_surfaces_a_search_failure() -> None:
    search = DuckDuckGoSearch(
        client=_StubClient("", status_code=500),  # type: ignore[arg-type]
    )
    batch = WebBatchFetcher(fetcher=_StubFetcher({}))  # type: ignore[arg-type]

    from agent_workflow.core.entities.models.web_research import (
        WebResearchError,
    )

    with pytest.raises(WebResearchError, match="500"):
        await WebResearcher(search, batch).research("q")


def test_research_policy_rejects_asking_for_more_than_it_searches() -> None:
    with pytest.raises(ValueError, match="max_results"):
        WebResearchPolicy(
            search=WebSearchPolicy(max_results=2),
            batch=WebBatchPolicy(max_pages=5),
        )


# ======================================================================
# Redirect handling: the SSRF fix
# ======================================================================


async def test_redirect_to_a_blocked_address_is_refused() -> None:
    """A public page must not be able to steer a fetch inward.

    Without per-hop validation, any reachable URL could 302 to
    169.254.169.254 or to a service on the agent's own machine and
    have the agent read it.
    """

    server, entry = _redirect_server(
        "http://169.254.169.254/latest/meta-data/"
    )

    try:
        fetcher = WebFetcher(
            policy=_PermissiveEntryPolicy(_origin(entry))
        )

        with pytest.raises(WebPolicyError):
            await fetcher.fetch(entry)
    finally:
        server.shutdown()


async def test_redirect_to_a_local_port_is_refused() -> None:
    server, entry = _redirect_server(
        "http://127.0.0.1:9999/"
    )

    try:
        fetcher = WebFetcher(
            policy=_PermissiveEntryPolicy(_origin(entry))
        )

        with pytest.raises(WebPolicyError, match="Port 9999"):
            await fetcher.fetch(entry)
    finally:
        server.shutdown()


async def test_entry_url_is_validated_before_any_request() -> None:
    # Fail closed: the check must not depend on a caller having
    # validated the URL already.
    fetcher = WebFetcher()

    with pytest.raises(WebPolicyError):
        await fetcher.fetch("http://127.0.0.1:9999/")


async def test_a_permissive_policy_is_honoured() -> None:
    # Proves the tests above fail because of the redirect check and
    # not because the fetch itself is broken.
    server, entry = _redirect_server(
        "https://example.com/landed"
    )

    try:
        fetcher = WebFetcher(
            policy=_PermissiveEntryPolicy(_origin(entry))
        )

        with pytest.raises(Exception) as caught:
            await fetcher.fetch(entry)

        # example.com is public, so the fetch proceeds past the
        # policy and fails on the network or on the same-host rule.
        assert not isinstance(caught.value, WebPolicyError)
    finally:
        server.shutdown()


# ======================================================================
# Tools
# ======================================================================


async def test_search_tool_returns_formatted_results() -> None:
    tool = create_web_search_tool(
        DuckDuckGoSearch(
            client=_StubClient(RESULT_PAGE),  # type: ignore[arg-type]
        )
    )

    output = await tool.handler(
        _arguments(tool, {"query": "q"}),
        _context(),
    )

    assert "[1]" in output
    assert "https://example.com/one" in output


async def test_research_tool_lists_results_and_pages() -> None:
    search = DuckDuckGoSearch(
        client=_StubClient(RESULT_PAGE),  # type: ignore[arg-type]
    )
    tool = create_web_research_tool(
        WebResearcher(
            search,
            WebBatchFetcher(fetcher=_StubFetcher({})),  # type: ignore[arg-type]
        )
    )

    output = await tool.handler(
        _arguments(tool, {"query": "q"}),
        _context(),
    )

    assert "All results:" in output
    assert "Content of https://example.com/one" in output


async def test_research_tool_says_so_when_nothing_is_found() -> None:
    tool = create_web_research_tool(
        WebResearcher(
            DuckDuckGoSearch(
                client=_StubClient("<html>none</html>"),  # type: ignore[arg-type]
            ),
            WebBatchFetcher(fetcher=_StubFetcher({})),  # type: ignore[arg-type]
        )
    )

    output = await tool.handler(
        _arguments(tool, {"query": "obscure"}),
        _context(),
    )

    assert "No search results" in output
    assert "web_fetch" in output


def test_search_tool_requires_the_search_permission() -> None:
    tool = create_web_search_tool(DuckDuckGoSearch())

    assert "web.search" in tool.policy.permissions


def test_research_tool_requires_both_permissions() -> None:
    tool = create_web_research_tool(WebResearcher())

    assert tool.policy.permissions == frozenset(
        {"web.fetch", "web.search"}
    )


def _arguments(tool, values):
    from agent_workflow.core.entities.models.arguments import (
        ArgumentDecoder,
    )

    return ArgumentDecoder().decode(values, tool.input_type)


def _context():
    from agent_workflow.core.entities.models.tool import ToolContext

    return ToolContext(
        working_directory=None,  # type: ignore[arg-type]
        environment={},
        allowed_path=(),
        permissions=frozenset(),
    )
"""A SearXNG instance is the supported way to actually get web results.

Every keyless search front-end on the open internet answers a scripted
client with a JavaScript challenge, so the tests here are mostly about
the ways a *self-hosted* instance fails differently: JSON not enabled,
rate limited, private, or simply not running. Each has to be named,
because "no results" and "not configured for this client" look the
same from the caller's side otherwise.
"""

from __future__ import annotations

import json

import httpx
import pytest

from agent_workflow.core.entities.models import web_failure
from agent_workflow.core.entities.models.builtin.tools import (
    build_search_backend,
)
from agent_workflow.core.entities.models.searxng_search import (
    SearxngPolicy,
    SearxngSearch,
)
from agent_workflow.core.entities.models.web_research import WebResearcher
from agent_workflow.core.entities.models.web_batch import WebBatchFetcher
from agent_workflow.core.infrastructure.config import WebConfig


@pytest.fixture(autouse=True)
def _clean_outage():
    web_failure.reset()

    yield

    web_failure.reset()


class Transport:
    def __init__(self, response: httpx.Response) -> None:
        self.response = response
        self.calls = 0

    async def handle_async_request(self, request):
        self.calls += 1

        return self.response


def search_with(response: httpx.Response, endpoint: str = "http://searx:8080"):
    transport = Transport(response)
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(transport.handle_async_request)
    )

    return SearxngSearch(
        SearxngPolicy(endpoint=endpoint),
        client=client,
    ), transport


def json_response(rows: list[dict], status: int = 200) -> httpx.Response:
    return httpx.Response(
        status,
        text=json.dumps({"results": rows}),
    )


async def test_it_returns_results_in_the_shared_shape() -> None:
    """Same shape as the other backend, so research needs no changes."""

    search, _transport = search_with(
        json_response(
            [
                {
                    "url": "https://www.reddit.com/r/x",
                    "title": "Shrooms Q on Reddit",
                    "content": "a thread",
                },
                {
                    "url": "https://bsky.app/profile/shrooms",
                    "title": "Bluesky",
                    "content": "profile",
                },
            ]
        )
    )

    results = await search.search("shrooms q")

    assert [result.url for result in results] == [
        "https://www.reddit.com/r/x",
        "https://bsky.app/profile/shrooms",
    ]
    assert results[0].title == "Shrooms Q on Reddit"
    assert results[0].snippet == "a thread"


async def test_it_requests_the_json_format() -> None:
    """format=json is the whole point: it turns a scrape into an API."""

    seen: list[str] = []

    async def handler(request):
        seen.append(str(request.url))

        return json_response([])

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    search = SearxngSearch(SearxngPolicy(endpoint="http://searx:8080"), client=client)

    await search.search("shrooms")

    assert "format=json" in seen[0]
    assert "q=shrooms" in seen[0]
    assert seen[0].startswith("http://searx:8080/search?")


async def test_it_honours_the_limit() -> None:
    search, _transport = search_with(
        json_response(
            [{"url": f"https://e.dev/{index}", "title": str(index)} for index in range(20)]
        )
    )

    # max_results is the keyword the tools pass; limit stays internal.
    assert len(await search.search("x", max_results=4)) == 4


async def test_it_drops_unusable_rows() -> None:
    """A javascript: target would be handed to a fetcher that follows it."""

    search, _transport = search_with(
        json_response(
            [
                {"url": "javascript:alert(1)", "title": "bad"},
                {"url": "https://ok.dev/a", "title": ""},
                {"url": "https://ok.dev/b", "title": "good"},
            ]
        )
    )

    results = await search.search("x")

    assert [result.url for result in results] == ["https://ok.dev/b"]


async def test_it_deduplicates() -> None:
    search, _transport = search_with(
        json_response(
            [
                {"url": "https://a.dev", "title": "one"},
                {"url": "https://a.dev", "title": "two"},
            ]
        )
    )

    assert len(await search.search("x")) == 1


async def test_html_means_the_json_format_is_not_enabled() -> None:
    """The classic setup mistake, and it reads as zero results otherwise."""

    search, _transport = search_with(
        httpx.Response(200, text="<!doctype html><html><body>searx</body></html>")
    )

    with pytest.raises(Exception) as caught:
        await search.search("x")

    message = str(caught.value)

    assert "JSON format is not enabled" in message
    assert "formats" in message
    assert "json" in message
    # The caller must not go looking for a better query.
    assert "No rewrite of the query will help" in message


async def test_rate_limiting_says_waiting_does_help() -> None:
    """Unlike a public anti-bot challenge, this one is the instance's own."""

    search, _transport = search_with(httpx.Response(429, text=""))

    with pytest.raises(Exception) as caught:
        await search.search("x")

    message = str(caught.value)

    assert "rate limiting" in message
    assert "waiting does help" in message


async def test_a_private_instance_is_named_as_such() -> None:
    search, _transport = search_with(httpx.Response(403, text=""))

    with pytest.raises(Exception) as caught:
        await search.search("x")

    message = str(caught.value)

    assert "403" in message
    assert "private" in message


async def test_an_unreachable_instance_reports_the_host() -> None:
    class Dead(Transport):
        async def handle_async_request(self, request):
            raise httpx.ConnectError(
                "[Errno 111] Connection refused"
            )

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(Dead(None).handle_async_request)
    )
    search = SearxngSearch(SearxngPolicy(endpoint="http://localhost:9"), client=client)

    with pytest.raises(Exception) as caught:
        await search.search("x")

    assert "could not connect" in str(caught.value)


async def test_a_response_without_results_is_not_read_as_empty() -> None:
    """An upstream engine failure is not the same as no matches."""

    search, _transport = search_with(
        httpx.Response(200, text=json.dumps({"query": "x"}))
    )

    with pytest.raises(Exception) as caught:
        await search.search("x")

    assert "no `results` key" in str(caught.value)


async def test_a_genuinely_empty_result_set_is_empty() -> None:
    search, _transport = search_with(json_response([]))

    assert await search.search("x") == []


async def test_an_empty_query_is_refused_before_the_request() -> None:
    search, transport = search_with(json_response([]))

    with pytest.raises(Exception, match="query is empty"):
        await search.search("   ")

    assert transport.calls == 0


def test_policy_rejects_nonsense() -> None:
    with pytest.raises(ValueError):
        SearxngPolicy(endpoint="  ")

    with pytest.raises(ValueError):
        SearxngPolicy(timeout=0)


async def test_the_backend_is_swappable_without_touching_the_tools() -> None:
    """The research tool takes the backend by injection."""

    search, _transport = search_with(
        json_response([{"url": "https://a.dev", "title": "A", "content": "c"}])
    )

    researcher = WebResearcher(search, WebBatchFetcher())

    assert researcher is not None


def test_configuration_selects_the_backend() -> None:
    default = build_search_backend(WebConfig())

    assert type(default).__name__ == "DuckDuckGoSearch"

    chosen = build_search_backend(
        WebConfig(backend="searxng", searxng_endpoint="http://searx.local:8080")
    )

    assert type(chosen).__name__ == "SearxngSearch"
    assert chosen.endpoint == "http://searx.local:8080"


def test_configuration_rejects_an_unknown_backend() -> None:
    with pytest.raises(ValueError, match="web.backend"):

        WebConfig(backend="google")


def test_searxng_without_an_endpoint_is_refused() -> None:
    with pytest.raises(ValueError, match="searxng_endpoint"):

        WebConfig(backend="searxng", searxng_endpoint="")
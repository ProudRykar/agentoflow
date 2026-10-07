"""Both search backends must be interchangeable.

The tool calls were written against DuckDuckGo's keyword-only
`max_results`. A second backend that took a positional `limit` looked
fine in isolation and passed every test it had, because nothing ever
exercised the tools *with* it: the moment it became the configured
backend, `web_search` raised `unexpected keyword argument` and web
search was simply gone, with nothing in the output to say that
swapping the engine had caused it.

So these tests drive the real tools, through the real handlers, with
each backend in turn. That is the only arrangement that catches a
mismatch in the shared calling convention.
"""

from __future__ import annotations

import json

import httpx
import pytest

from agent_workflow.core.entities.models.builtin.tools import (
    build_search_backend,
)
from agent_workflow.core.entities.models.searxng_search import (
    SearxngPolicy,
    SearxngSearch,
)
from agent_workflow.core.entities.models.tool import ToolContext
from agent_workflow.core.entities.models.web_batch import WebBatchFetcher
from agent_workflow.core.entities.models.web_research import WebResearcher
from agent_workflow.core.entities.models.web_search import (
    DuckDuckGoSearch,
    WebSearchInput,
    WebSearchPolicy,
    create_web_search_tool,
)
from agent_workflow.core.infrastructure.config import WebConfig


RESULTS = [
    {
        "url": "https://www.reddit.com/r/x/comments/1/",
        "title": "A thread about her",
        "content": "found her there",
    },
    {
        "url": "https://bsky.app/profile/her",
        "title": "On Bluesky",
        "content": "profile",
    },
    {
        "url": "https://x.com/her",
        "title": "On X",
        "content": "profile",
    },
]

# DuckDuckGo wraps results in its own redirect, so the parser has to
# unwrap the target; this is the shape it expects.
DDG_HTML = """
<html><body><table>
<tr><td>
  <a rel="nofollow" class="result-link"
     href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fwww.reddit.com%2Fr%2Fx%2F">
    A thread about her
  </a>
</td><td class='result-snippet'>found her there</td></tr>
<tr><td>
  <a rel="nofollow" class="result-link"
     href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fbsky.app%2Fprofile%2Fher">
    On Bluesky
  </a>
</td><td class='result-snippet'>profile</td></tr>
<tr><td>
  <a rel="nofollow" class="result-link"
     href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fx.com%2Fher">
    On X
  </a>
</td><td class='result-snippet'>profile</td></tr>
</table></body></html>
"""

SSEARX_JSON = json.dumps({"results": RESULTS})


def _ddg() -> DuckDuckGoSearch:
    def handler(request):
        return httpx.Response(200, text=DDG_HTML)

    return DuckDuckGoSearch(
        client=httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ),
        policy=WebSearchPolicy(),
    )


def _searxng() -> SearxngSearch:
    def handler(request):
        return httpx.Response(200, text=SSEARX_JSON)

    return SearxngSearch(
        SearxngPolicy(endpoint="http://searx:8080"),
        client=httpx.AsyncClient(
            transport=httpx.MockTransport(handler)
        ),
    )


BACKENDS = {"duckduckgo": _ddg, "searxng": _searxng}


def _context() -> ToolContext:
    from pathlib import Path

    return ToolContext(
        working_directory=Path.cwd(),
        environment={},
        allowed_path=Path.cwd(),
        permissions=frozenset({"web.search"}),
    )


@pytest.mark.parametrize("name", sorted(BACKENDS))
async def test_web_search_accepts_every_backend(name: str) -> None:
    """The call shape the tool uses, against each engine.

    This is the test whose absence let a keyword mismatch ship.
    """

    search = BACKENDS[name]()
    tool = create_web_search_tool(search)

    output = await tool.handler(
        tool.input_type(query="her name", max_results=3),
        _context(),
    )

    # Both engines must have produced the same page of results, and
    # neither may have raised through the handler.
    assert "unexpected keyword argument" not in output
    assert "reddit.com" in output


@pytest.mark.parametrize("name", sorted(BACKENDS))
async def test_the_handler_never_passes_a_positional_limit(name: str) -> None:
    """A second positional would break the other backend.

    DuckDuckGo's limit is keyword-only, so anything else that changes
    is a mismatch waiting to surface on the other engine.
    """

    search = BACKENDS[name]()

    with pytest.raises(TypeError) as caught:
        await search.search("query", 3)  # type: ignore[misc]

    assert "positional" in str(caught.value)


@pytest.mark.parametrize("name", sorted(BACKENDS))
async def test_research_accepts_every_backend(name: str) -> None:
    search = BACKENDS[name]()

    researcher = WebResearcher(search, WebBatchFetcher())

    assert researcher is not None


def test_the_default_backend_builds_without_a_client() -> None:
    """The real runtime path: no injected client at all."""

    assert type(build_search_backend(WebConfig())).__name__ == (
        "DuckDuckGoSearch"
    )

    assert type(
        build_search_backend(
            WebConfig(backend="searxng", searxng_endpoint="http://searx:8080")
        )
    ).__name__ == "SearxngSearch"


async def test_max_results_is_honoured_by_both() -> None:
    for factory in (_ddg, _searxng):
        search = factory()

        rows = await search.search("her", max_results=1)

        assert len(rows) == 1, factory.__name__


async def test_a_default_limit_applies_when_none_is_given() -> None:
    for factory in (_ddg, _searxng):
        search = factory()

        rows = await search.search("her")

        assert rows, factory.__name__


def test_the_search_input_still_exposes_max_results() -> None:
    """The tool's own argument, which the handler forwards."""

    arguments = WebSearchInput(query="x", max_results=7)

    assert arguments.max_results == 7

DEGRADED_BODY = json.dumps(
    {
        "results": [],
        "unresponsive_engines": [
            ["brave", "Suspended: too many requests"],
            ["duckduckgo", "timeout"],
        ],
    }
)


async def test_an_unanswered_search_is_not_reported_as_no_match() -> None:
    """The same false negative as the sweep, one layer up.

    A refused engine returns an empty result set, and reporting that
    as "nothing found" states as fact that the subject does not
    exist. Worse, the advice attached to it -- try other wording --
    sends the caller round a rephrasing loop that cannot succeed,
    because the wording was never the problem.
    """

    def handler(request):
        return httpx.Response(200, text=DEGRADED_BODY)

    search = SearxngSearch(
        SearxngPolicy(endpoint="http://searx:8080"),
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )

    tool = create_web_search_tool(search)

    output = await tool.handler(
        tool.input_type(query="Eveline Dellai", max_results=5),
        _context(),
    )

    assert "NOTHING WAS ACTUALLY ASKED" in output
    assert "brave" in output
    assert "not a statement about whether anything exists" in output
    # The advice that provokes the rephrasing loop must be gone.
    assert "broader wording" not in output


async def test_a_genuinely_empty_answer_still_suggests_rephrasing() -> None:
    """When the engines did answer, "try other wording" is correct."""

    def handler(request):
        return httpx.Response(
            200,
            text=json.dumps({"results": [], "unresponsive_engines": []}),
        )

    search = SearxngSearch(
        SearxngPolicy(endpoint="http://searx:8080"),
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )

    tool = create_web_search_tool(search)

    output = await tool.handler(
        tool.input_type(query="nothing indexed", max_results=5),
        _context(),
    )

    assert "No search results" in output
    assert "broader wording" in output
    assert "NOTHING WAS ACTUALLY ASKED" not in output


@pytest.mark.parametrize("name", sorted(BACKENDS))
async def test_every_backend_answers_the_detailed_question(name: str) -> None:
    """Both must be able to say that nothing was asked."""

    results, degraded = await BACKENDS[name]().search_detailed("x")

    assert isinstance(results, list)
    assert isinstance(degraded, tuple)

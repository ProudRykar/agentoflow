from __future__ import annotations

import html
import re
from dataclasses import dataclass
from html.parser import HTMLParser
from urllib.parse import parse_qs, urlparse

import httpx

from agent_workflow.core.entities.models import web_failure as failure
from agent_workflow.core.entities.models.tool import (
    Tool,
    ToolContext,
    ToolPolicy,
)
from agent_workflow.core.entities.models.web_policy import WebPolicy


class WebSearchError(RuntimeError):
    """The search backend could not answer."""


@dataclass(slots=True, frozen=True)
class WebSearchInput:
    query: str

    max_results: int = 0
    """Fewer results than requested. 0 means the runtime default."""


@dataclass(slots=True, frozen=True)
class WebSearchPolicy:
    """Trusted runtime limits for search.

    Not exposed as tool arguments. The model chooses what to look
    for; the runtime decides how much is allowed.
    """

    max_results: int = 8

    max_query_chars: int = 400

    timeout: float = 20.0

    safe_search: bool = True

    def __post_init__(self) -> None:
        if self.max_results < 1:
            raise ValueError("max_results must be >= 1")

        if self.max_query_chars < 1:
            raise ValueError(
                "max_query_chars must be >= 1"
            )

        if self.timeout <= 0:
            raise ValueError("timeout must be > 0")


@dataclass(slots=True, frozen=True)
class SearchResult:
    url: str
    title: str = ""
    snippet: str = ""

    def to_json(self) -> dict[str, str]:
        return {
            "url": self.url,
            "title": self.title,
            "snippet": self.snippet,
        }


# DuckDuckGo's HTML endpoint wraps every result in a redirector whose
# target sits in the uddg query parameter. The lite endpoint is used
# because it answers without the anti-bot interstitial.
_LITE_ENDPOINT = "https://lite.duckduckgo.com/lite/"

_RESULT_LINK = re.compile(
    r'<a[^>]+class="[^"]*result-link[^"]*"[^>]*'
    r'href="(?P<href>[^"]+)"[^>]*>(?P<title>.*?)</a>',
    re.DOTALL | re.IGNORECASE,
)

_SNIPPET = re.compile(
    r"<td[^>]+class=\"?'?result-snippet'?\"?[^>]*>"
    r"(?P<body>.*?)</td>",
    re.DOTALL | re.IGNORECASE,
)

_TAG = re.compile(r"<[^>]+>")

# Signals that the endpoint answered with an anti-bot page rather
# than results. Reported as an explicit error so the model stops
# retrying instead of concluding there were no matches.
_BLOCK_MARKERS = (
    "anomaly.js",
    "unfortunately, bots",
    "please complete the following challenge",
)


def _text(raw: str) -> str:
    return " ".join(
        html.unescape(_TAG.sub(" ", raw)).split()
    )


def _unwrap(href: str) -> str:
    """Recover the real target from DuckDuckGo's redirector."""

    if href.startswith("//"):
        href = f"https:{href}"

    parsed = urlparse(href)

    if parsed.path.startswith("/l/"):
        target = parse_qs(parsed.query).get(
            "uddg",
            [""],
        )[0]

        if target:
            return target

    return href


class _ResultParser(HTMLParser):
    """Collects result links and their snippets.

    A parser rather than a regex over the whole document because the
    snippet cells are siblings of the links, so the two have to be
    matched up in document order.
    """

    def __init__(self) -> None:
        super().__init__()

        self.links: list[tuple[str, str]] = []
        self.snippets: list[str] = []

        self._in_link = False
        self._in_snippet = False
        self._href = ""
        self._buffer: list[str] = []

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        attributes = {
            key.lower(): value or ""
            for key, value in attrs
        }

        if tag == "a" and "uddg=" in attributes.get(
            "href",
            "",
        ):
            self._in_link = True
            self._href = attributes["href"]
            self._buffer = []
            return

        if tag == "td" and "result-snippet" in attributes.get(
            "class",
            "",
        ):
            self._in_snippet = True
            self._buffer = []

    def handle_data(self, data: str) -> None:
        if self._in_link or self._in_snippet:
            self._buffer.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._in_link:
            self._in_link = False
            self.links.append(
                (self._href, _text("".join(self._buffer)))
            )
            self._buffer = []
            return

        if tag == "td" and self._in_snippet:
            self._in_snippet = False
            self.snippets.append(
                _text("".join(self._buffer))
            )
            self._buffer = []


class DuckDuckGoSearch:
    """Keyless web search.

    Scraping an HTML endpoint, not an API, so it can break without
    notice and rate-limit under load. The failure is reported as an
    error with a distinct message for that reason: a silent empty
    result list would read to the model as "nothing matched".
    """

    def __init__(
        self,
        policy: WebSearchPolicy | None = None,
        *,
        endpoint: str = _LITE_ENDPOINT,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._policy = policy or WebSearchPolicy()
        self._endpoint = endpoint
        self._client = client
        self._web_policy = WebPolicy()

    @property
    def policy(self) -> WebSearchPolicy:
        return self._policy

    async def search(
        self,
        query: str,
        *,
        max_results: int | None = None,
    ) -> list[SearchResult]:
        results, _degraded = await self.search_detailed(
            query,
            max_results=max_results,
        )

        return results

    async def search_detailed(
        self,
        query: str,
        *,
        max_results: int | None = None,
    ) -> tuple[list[SearchResult], tuple[str, ...]]:
        """Results, plus the engines that did not answer.

        Always empty here: this backend either returns results or
        raises. It exists so both backends answer the same question,
        because a tool that cannot tell "nothing matched" from
        "nothing was asked" reports the second as the first.
        """

        cleaned = " ".join(query.split())

        if not cleaned:
            raise WebSearchError("Search query is empty")

        limit = self._policy.max_results

        if max_results is not None and max_results > 0:
            limit = min(max_results, limit)

        truncated = cleaned[: self._policy.max_query_chars]

        if len(truncated) < len(cleaned):
            truncated = truncated.rstrip()

        # The endpoint is public and fixed, but it is still an
        # outbound request, so it goes through the same policy.
        await self._web_policy.validate_url(
            self._endpoint
        )

        payload = await self._request(truncated)

        results = self._parse(payload, limit)

        return results, ()

    async def _request(self, query: str) -> str:
        parameters = {"q": query}

        if self._policy.safe_search:
            parameters["kp"] = "1"

        blocked = failure.blocked_message(self._endpoint)

        if blocked is not None:
            # A previous call already established that the network is
            # unreachable; waiting again would just repeat the timeout.
            raise WebSearchError(blocked)

        try:
            if self._client is not None:
                response = await self._client.get(
                    self._endpoint,
                    params=parameters,
                )
            else:
                async with httpx.AsyncClient(
                    timeout=httpx.Timeout(
                        self._policy.timeout,
                        connect=min(
                            self._policy.timeout,
                            10.0,
                        ),
                    ),
                    follow_redirects=True,
                    trust_env=False,
                    headers=_HEADERS,
                ) as client:
                    response = await client.get(
                        self._endpoint,
                        params=parameters,
                    )
        except httpx.HTTPError as exc:
            raise WebSearchError(
                failure.note_failure(exc, self._endpoint)
            ) from exc

        failure.note_success()

        if response.status_code >= 400:
            raise WebSearchError(
                "Search endpoint returned HTTP "
                f"{response.status_code}"
            )

        return response.text

    def _parse(
        self,
        payload: str,
        limit: int,
    ) -> list[SearchResult]:
        lowered = payload.lower()

        for marker in _BLOCK_MARKERS:
            if marker in lowered:
                # Not a rate limit, whatever it looks like: this is a
                # JavaScript challenge served to this address, and it
                # is returned with HTTP 202 as a success. Waiting does
                # not clear it, and neither does rephrasing the query.
                # Saying "rate limited, wait a moment" sent a caller
                # round a retry loop that could not succeed.
                raise WebSearchError(
                    "DuckDuckGo served an anti-bot challenge instead "
                    "of results. This is not a rate limit and the "
                    "request was not at fault: every keyless search "
                    "front-end reachable from here answers this way, "
                    "because they all gate on this address rather "
                    "than on the request headers.\n"
                    "Do not retry, and do not try another phrasing or "
                    "another scraping target; the same answer will "
                    "come back. A User-Agent will not help either.\n"
                    "To get web results, configure a search API key "
                    "(see the [web] section of config.toml), or "
                    "answer from what is already available."
                )

        parser = _ResultParser()

        try:
            parser.feed(payload)
        except Exception as exc:
            raise WebSearchError(
                f"Could not parse search results: {exc}"
            ) from exc

        # The regexes are a fallback: the parser above is the primary
        # path, and DuckDuckGo has changed its markup before.
        if not parser.links:
            parser.links = [
                (match.group("href"), match.group("title"))
                for match in _RESULT_LINK.finditer(payload)
            ]

        if not parser.snippets:
            parser.snippets = [
                match.group("body")
                for match in _SNIPPET.finditer(payload)
            ]

        results: list[SearchResult] = []
        seen: set[str] = set()

        for index, (href, title) in enumerate(parser.links):
            url = _unwrap(href).split("#", 1)[0].rstrip()

            if not url.startswith(("http://", "https://")):
                continue

            if url in seen:
                continue

            seen.add(url)

            snippet = ""

            if index < len(parser.snippets):
                snippet = parser.snippets[index]

            results.append(
                SearchResult(
                    url=url,
                    title=title,
                    snippet=snippet,
                )
            )

            if len(results) >= limit:
                break

        return results


_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "en-US,en;q=0.9",
}


def format_results(
    query: str,
    results: list[SearchResult],
    degraded: tuple[str, ...] = (),
) -> str:
    """Render results for the model.

    Numbered so the model can refer to "result 3" when it follows
    up with web_fetch_many.

    ``degraded`` names the engines that did not answer. Without it an
    empty result set is reported as "nothing matched", which is how a
    refused engine becomes a confident claim that the subject does not
    exist. Telling the caller to try other wording, when nothing was
    actually asked, sends it round the same loop.
    """

    if not results and degraded:
        listed = ", ".join(sorted(degraded)[:6])

        return (
            f"NOTHING WAS ACTUALLY ASKED. The search for "
            f'"{query}" reached the engine, but these engines did '
            f"not answer: {listed}.\n"
            "This is not a statement about whether anything exists "
            "on the subject. Wait a few minutes and try again, and "
            "do not rephrase: the wording was never the problem."
        )

    if not results:
        return (
            f'No search results for "{query}". '
            "Try different or broader wording, or use "
            "web_fetch on a URL you already know."
        )

    lines = [
        f'Search: "{query}"',
        f"Results: {len(results)}",
        "",
    ]

    for index, result in enumerate(results, start=1):
        lines.append(f"[{index}] {result.title or '(no title)'}")

        if result.snippet:
            lines.append(f"    {result.snippet}")

        lines.append(f"    {result.url}")
        lines.append("")

    return "\n".join(lines).rstrip()

def create_web_search_tool(
    search: DuckDuckGoSearch,
) -> Tool:
    """The web_search tool.

    Kept separate from the research tool on purpose: looking up links
    is cheap and does not spend the download budget, so the model
    can decide whether it actually needs the pages.
    """

    async def handler(
        arguments: WebSearchInput,
        context: ToolContext,
    ) -> str:
        del context

        results, degraded = await search.search_detailed(
            arguments.query,
            max_results=arguments.max_results or None,
        )

        return format_results(arguments.query, results, degraded)

    return Tool(
        name="web_search",
        description=(
            "Search the web and return a numbered list of results "
            "with titles, snippets and URLs. Use it to find sources "
            "before reading them. Follow up with web_fetch_many to "
            "download several of the URLs in one call, or use "
            "web_research to do both at once.\n\n"
            "When it fails, read the message before trying again. A "
            "report that DNS cannot resolve a host, or that the "
            "connection is refused, means the network is unavailable "
            "rather than the query being wrong: rephrasing, widening "
            "or switching tools will fail identically and only spend "
            "time. The same applies to the note that repeated calls "
            "were not attempted, which means an earlier call already "
            "established that. In that case answer from what is "
            "already to hand, or say plainly that the lookup could "
            "not be done."
        ),
        input_type=WebSearchInput,
        handler=handler,
        policy=ToolPolicy(
            permissions=frozenset({
                "web.search",
            }),
            timeout=search.policy.timeout + 10.0,
            max_output_size=30_000,
        ),
    )

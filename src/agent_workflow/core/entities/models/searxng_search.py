"""Search through a self-hosted SearXNG instance.

Why this exists rather than another scraper: every keyless search
front-end on the open internet answers a scripted client with a
JavaScript challenge instead of results. DuckDuckGo serves one with
HTTP 202, Mojeek serves a captcha page, Brave serves an empty app
shell. Changing the User-Agent does not change any of that, because the
gate is on the address rather than on the request.

An instance you run answers to you. Its rate limits are yours, its
formats are yours, and turning on the JSON format turns a scraping
problem into an API problem, which is the only kind that keeps working.

SearXNG serves ``format=json`` only when the administrator enables it,
so the two setup mistakes are named exactly, because "no results" is
what a missing format looks like and it is indistinguishable from a
genuinely empty result set otherwise.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote, urljoin

import httpx

from agent_workflow.core.entities.models import web_failure as failure
from agent_workflow.core.entities.models.web_search import (
    SearchResult,
    WebSearchError,
)


# What the tools ask for when the caller sets no limit.
DEFAULT_RESULTS = 5


class SearxngError(WebSearchError):
    """The instance could not be searched."""


@dataclass(slots=True, frozen=True)
class SearxngPolicy:
    """How the instance is called."""

    endpoint: str = "http://localhost:8080"
    timeout: float = 20.0
    categories: str = "general"
    language: str = "en"

    def __post_init__(self) -> None:
        if not self.endpoint.strip():
            raise ValueError("endpoint is required")

        if self.timeout <= 0:
            raise ValueError("timeout must be > 0")


class SearxngSearch:
    """A search backend backed by one SearXNG instance.

    Returns the same :class:`SearchResult` shape as the DuckDuckGo
    backend, so the research and batch tools need no changes and the
    two can be swapped by configuration.
    """

    def __init__(
        self,
        policy: SearxngPolicy | None = None,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._policy = policy or SearxngPolicy()
        self._client = client

    @property
    def policy(self) -> SearxngPolicy:
        return self._policy

    @property
    def client(self) -> httpx.AsyncClient | None:
        return self._client

    @property
    def endpoint(self) -> str:
        return self._policy.endpoint

    def _search_url(self, query: str) -> str:
        # format=json is what makes this an API rather than a scrape.
        return urljoin(
            self._policy.endpoint.rstrip("/") + "/",
            "search",
        ) + f"?q={_quote(query)}&format=json&categories={_quote(self._policy.categories)}&language={_quote(self._policy.language)}"

    async def search(
        self,
        query: str,
        *,
        max_results: int | None = None,
    ) -> list[SearchResult]:
        """Results from the instance, or an error that says why not.

        The keyword is ``max_results`` and it is keyword-only because
        this is the signature the tools call. An earlier version took a
        positional ``limit``, which every call site then got wrong the
        moment this replaced the default backend: the tools raise
        ``unexpected keyword argument`` and web search is simply gone,
        with no hint that swapping the engine caused it.
        """

        results, _degraded = await self.search_detailed(
            query,
            limit=max_results or DEFAULT_RESULTS,
        )

        return results

    async def search_detailed(
        self,
        query: str,
        limit: int = DEFAULT_RESULTS,
        *,
        max_results: int | None = None,
    ) -> tuple[list[SearchResult], tuple[str, ...]]:
        """Results and the engines that did not answer."""

        if not query.strip():
            raise SearxngError("Search query is empty")

        effective = max_results or limit

        if effective < 1:
            raise SearxngError("max_results must be >= 1")

        target = self._search_url(query)

        try:
            if self._client is not None:
                response = await self._client.get(target)
            else:
                async with httpx.AsyncClient(
                    timeout=httpx.Timeout(
                        self._policy.timeout,
                        connect=min(self._policy.timeout, 10.0),
                    ),
                    follow_redirects=True,
                    trust_env=False,
                ) as client:
                    response = await client.get(target)
        except httpx.HTTPError as exc:
            raise SearxngError(
                failure.note_failure(exc, target)
            ) from exc

        failure.note_success()

        results, degraded = self._parse(response, limit)

        return results, degraded

    def _parse(
        self,
        response: httpx.Response,
        limit: int,
    ) -> tuple[list[SearchResult], tuple[str, ...]]:
        """Results, plus the engines that did not answer.

        The second value matters: an instance whose engines are all
        timing out returns an empty result set that looks exactly like
        a genuine "nothing matched". A sweep that reported that as
        "no account on this platform" would be stating a falsehood as
        evidence, and it would do so most often right after a restart,
        when the engines are still warming up.
        """
        if response.status_code == 429:
            raise SearxngError(
                f"The SearXNG instance at {self._policy.endpoint} is "
                "rate limiting this address. That is the instance's own "
                "limiter rather than a public anti-bot challenge, so "
                "waiting does help; try again shortly or lower the "
                "instance's rate limit."
            )

        if response.status_code in (401, 403):
            raise SearxngError(
                f"The SearXNG instance at {self._policy.endpoint} "
                f"answered HTTP {response.status_code}. SearXNG "
                "instances are often private; check that this URL is "
                "reachable without a login and that the instance "
                "allows this address."
            )

        if response.status_code >= 400:
            raise SearxngError(
                "The SearXNG instance returned HTTP "
                f"{response.status_code}."
            )

        body = response.text

        try:
            payload = json.loads(body)
        except json.JSONDecodeError as exc:
            # This is the JSON format not being enabled, which is the
            # single most common setup mistake and looks exactly like
            # an empty result set if it is not named.
            if "<html" in body[:400].lower():
                raise SearxngError(
                    f"{self._policy.endpoint} answered with HTML rather "
                    "than JSON, which means the JSON format is not "
                    "enabled on the instance. Add json to `formats` in "
                    "the instance's settings.yml under "
                    "`search.formats` and restart it:\n"
                    "  search:\n"
                    "    formats:\n"
                    "      - html\n"
                    "      - json\n"
                    "No rewrite of the query will help until that is "
                    "done."
                ) from exc

            raise SearxngError(
                f"{self._policy.endpoint} returned something that is "
                "neither JSON nor a recognisable HTML page."
            ) from exc

        if not isinstance(payload, dict):
            raise SearxngError(
                "The SearXNG instance returned JSON of an unexpected "
                "shape."
            )

        if "results" not in payload:
            raise SearxngError(
                "The SearXNG response carried no `results` key, which "
                "is what a blocked or erroring engine returns. Check "
                "the instance logs for the upstream failure."
            )

        unresponsive = payload.get("unresponsive_engines")
        degraded: tuple[str, ...] = ()

        if isinstance(unresponsive, list):
            names: list[str] = []

            for entry in unresponsive:
                if isinstance(entry, list) and entry:
                    names.append(str(entry[0]))
                elif isinstance(entry, str):
                    names.append(entry)

            degraded = tuple(sorted(set(names)))

        raw_results = payload.get("results")

        if not isinstance(raw_results, list):
            return [], degraded

        results: list[SearchResult] = []
        seen: set[str] = set()

        for row in raw_results:
            if not isinstance(row, dict):
                continue

            url = str(row.get("url") or "").strip()

            # An off-site redirect is not a result worth passing on to
            # a fetcher that will follow it.
            if not url.startswith(("http://", "https://")):
                continue

            title = str(row.get("title") or "").strip()

            if not title or url in seen:
                continue

            seen.add(url)

            snippet = str(
                row.get("content") or row.get("snippet") or ""
            ).strip()

            results.append(
                SearchResult(
                    url=url,
                    title=title,
                    snippet=snippet,
                )
            )

            if len(results) >= limit:
                break

        return results, degraded


def _quote(value: str) -> str:
    return quote(value, safe="")


def describe_for_model(endpoint: str) -> str:
    """A one-liner naming the instance, for error messages."""

    return f"SearXNG at {endpoint}"


__all__: list[Any] = [
    "SearxngError",
    "SearxngPolicy",
    "SearxngSearch",
    "describe_for_model",
]
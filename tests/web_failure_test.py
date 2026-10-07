"""A web tool that cannot say why it failed gets called again.

The symptom this guards against: an agent whose search times out
rephrases the query, widens it, switches tools, and spends minutes
waiting on DNS that will never answer. Each attempt burns a full connect
timeout and produces a message that reads like a transient glitch
rather than a hard outage.

So the failure has to be named, and the repeat has to be cheap.
"""

from __future__ import annotations

import httpx
import pytest

from agent_workflow.core.entities.models import web_failure as failure
from agent_workflow.core.entities.models.web_search import (
    DuckDuckGoSearch,
    WebSearchError,
    WebSearchPolicy,
)


@pytest.fixture(autouse=True)
def _clean_state():
    failure.reset()

    yield

    failure.reset()


def test_a_dns_failure_names_the_host_and_the_diagnosis() -> None:
    message = failure.describe(
        httpx.ConnectError("[Errno -2] Name or service not known"),
        "https://html.duckduckgo.com/html/?q=x",
    )

    assert "DNS could not resolve html.duckduckgo.com" in message
    assert "no web tool will work" in message


def test_a_connection_refusal_is_not_reported_as_dns() -> None:
    """They need different fixes, so they must not read the same."""

    message = failure.describe(
        httpx.ConnectError("All connection attempts failed"),
        "https://html.duckduckgo.com/",
    )

    assert "could not connect to html.duckduckgo.com" in message
    assert "DNS" not in message


def test_a_read_timeout_says_the_host_answered() -> None:
    message = failure.describe(httpx.ReadTimeout(""), "https://example.com")

    assert "connected to example.com" in message
    assert "stalled" in message


def test_a_connect_timeout_is_its_own_case() -> None:
    message = failure.describe(httpx.ConnectTimeout(""), "https://example.com")

    assert "not answering on the expected port" in message


def test_an_empty_exception_still_produces_a_message() -> None:
    """httpx leaves some exceptions with no text at all.

    Reporting `Search request failed: ` with nothing after it is
    indistinguishable from no report, which is what sent the caller
    round the loop in the first place.
    """

    message = failure.describe(httpx.ConnectError(""), "https://example.com")

    assert message.strip()
    assert "example.com" in message


def test_an_unknown_exception_keeps_its_class_name() -> None:
    message = failure.describe(ValueError(""), "https://example.com")

    assert "ValueError" in message


def test_a_hard_failure_suppresses_immediate_repeats() -> None:
    first = failure.note_failure(
        httpx.ConnectError("[Errno -2] Name or service not known"),
        "https://html.duckduckgo.com/",
    )

    assert failure.blocked_message("https://html.duckduckgo.com/") is not None

    blocked = failure.blocked_message("https://html.duckduckgo.com/")

    assert blocked is not None
    assert "DNS" in blocked
    assert "were not attempted" in blocked
    # The diagnosis is preserved, not replaced by a generic refusal.
    assert "Name or service" not in blocked
    assert first


def test_a_read_timeout_does_not_block_the_next_attempt() -> None:
    """The host was reachable, so the next call is worth making."""

    failure.note_failure(httpx.ReadTimeout(""), "https://example.com")

    assert failure.blocked_message("https://example.com") is None


def test_a_success_clears_a_recorded_outage() -> None:
    failure.note_failure(
        httpx.ConnectError("[Errno -2] Name or service not known"),
        "https://html.duckduckgo.com/",
    )

    assert failure.blocked_message("https://html.duckduckgo.com/") is not None

    failure.note_success()

    assert failure.blocked_message("https://html.duckduckgo.com/") is None


def test_the_cooldown_expires(monkeypatch: pytest.MonkeyPatch) -> None:
    failure.note_failure(
        httpx.ConnectError("[Errno -2] Name or service not known"),
        "https://html.duckduckgo.com/",
    )

    assert failure.blocked_message("https://html.duckduckgo.com/") is not None

    later = failure.time.monotonic() + failure.COOLDOWN_SECONDS + 1
    monkeypatch.setattr(failure.time, "monotonic", lambda: later)

    assert failure.blocked_message("https://html.duckduckgo.com/") is None


class _FailingTransport:
    """An httpx transport that always fails to connect."""

    def __init__(self, exc: Exception) -> None:
        self.exc = exc
        self.calls = 0

    async def handle_async_request(self, request):
        self.calls += 1

        raise self.exc


async def test_a_second_search_fails_without_waiting() -> None:
    """The whole point: four attempts, one timeout."""

    transport = _FailingTransport(
        httpx.ConnectError("[Errno -2] Name or service not known")
    )

    client = httpx.AsyncClient(transport=httpx.MockTransport(transport.handle_async_request))
    search = DuckDuckGoSearch(client=client, policy=WebSearchPolicy(timeout=30.0))

    with pytest.raises(WebSearchError) as first:
        await search.search("shrooms q")

    with pytest.raises(WebSearchError) as second:
        await search.search("shrooms q daisy love")

    assert "DNS" in str(first.value)
    assert "were not attempted" in str(second.value)
    # One request went out. The second was refused on the evidence of
    # the first.
    assert transport.calls == 1


async def test_a_read_timeout_is_reported_and_retried_next_time() -> None:
    transport = _FailingTransport(httpx.ReadTimeout(""))

    client = httpx.AsyncClient(transport=httpx.MockTransport(transport.handle_async_request))
    search = DuckDuckGoSearch(client=client, policy=WebSearchPolicy(timeout=30.0))

    with pytest.raises(WebSearchError) as first:
        await search.search("one")

    with pytest.raises(WebSearchError) as second:
        await search.search("two")

    assert "stalled" in str(first.value)
    # No suppression here, so both really attempted.
    assert "were not attempted" not in str(second.value)
    assert transport.calls == 2

class _StaticTransport:
    def __init__(self, status: int, text: str) -> None:
        self.status = status
        self.text = text
        self.calls = 0

    async def handle_async_request(self, request):
        self.calls += 1

        return httpx.Response(self.status, text=self.text)


def _challenge() -> str:
    """The page DuckDuckGo actually serves, captured from this host."""

    from pathlib import Path

    return Path(__file__).parent.joinpath(
        "fixtures/ddg_challenge.html"
    ).read_text(encoding="utf-8")


async def test_the_real_challenge_page_is_reported_honestly() -> None:
    """It used to claim a rate limit, which sent the caller round a loop.

    The response is HTTP 202, which reads as success, and the page
    carries an `anomaly.js` challenge. Nothing about it clears with
    time or with a different query, so telling the caller to wait was
    the one thing that guaranteed another wasted attempt.
    """

    transport = _StaticTransport(202, _challenge())
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(transport.handle_async_request)
    )

    search = DuckDuckGoSearch(client=client, policy=WebSearchPolicy())

    with pytest.raises(WebSearchError) as caught:
        await search.search("shrooms q reddit bluesky")

    message = str(caught.value)

    assert "not a rate limit" in message
    assert "Do not retry" in message
    assert "User-Agent will not help" in message
    # The old wording, which was wrong and caused the retries.
    assert "wait a moment" not in message


async def test_a_challenge_is_not_treated_as_a_network_outage() -> None:
    """The host answered, so fetch and research still get a fair try."""

    transport = _StaticTransport(202, _challenge())
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(transport.handle_async_request)
    )

    search = DuckDuckGoSearch(client=client, policy=WebSearchPolicy())

    with pytest.raises(WebSearchError):
        await search.search("one")

    assert failure.blocked_message("https://html.duckduckgo.com/") is None

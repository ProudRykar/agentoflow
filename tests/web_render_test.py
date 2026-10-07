"""The fallback has to be safe and boring when there is no browser.

Rendering cannot be part of the default test run: CI has no Chromium,
and a test that quietly passes because the render was skipped is worth
nothing. So these cover the decision and the guards with a fake, and
the one test that needs a real browser is marked and skips itself.

What is actually being protected here is not the crawl. It is that a
browser is a much larger hole than httpx: one page can ask for an image
on the link-local metadata address and httpx would never have seen the
request at all, because WebPolicy only checks the entry URL and its
redirects.
"""

from __future__ import annotations

import os

import pytest

from agent_workflow.core.entities.models.builtin.web_crawl import (
    WebPage,
)
from agent_workflow.core.entities.models.web_policy import (
    WebPolicyError,
)
from agent_workflow.core.entities.models.web_render import (
    BrowserFetcher,
    RenderFallbackFetcher,
    RenderPolicy,
    RenderUnavailable,
    THIN_PAGE_LINKS,
    _is_noise,
    resolve_executable,
)

# A test that drives a real browser against the real internet does not
# belong in the default run: it fails when the network is slow, and a
# suite whose other tests are hermetic has no business carrying that.
# Opt in with AGENTOFLOW_RENDER_TESTS=1.
RUN_LIVE_RENDER_TESTS = (
    os.environ.get("AGENTOFLOW_RENDER_TESTS") == "1"
)


def page(links: tuple[str, ...]) -> WebPage:
    return WebPage(
        url="https://example.com/",
        title="t",
        content="<html></html>",
        links=links,
        status_code=200,
        content_type="text/html",
        content_bytes=11,
    )


class FakeStatic:
    timeout = 30.0

    def __init__(self, links: tuple[str, ...] = ()) -> None:
        self._page = page(links)
        self.calls: list[str] = []

    async def fetch(self, url: str, *args, **kwargs) -> WebPage:
        self.calls.append(url)
        return self._page


class FakeBrowser:
    timeout = 30.0

    def __init__(
        self,
        links: tuple[str, ...] = (),
        available: bool = True,
        raises: Exception | None = None,
    ) -> None:
        self._links = links
        self.available = available
        self._raises = raises
        self.calls: list[str] = []
        self.closed = False

    async def fetch(self, url: str, *args, **kwargs) -> WebPage:
        self.calls.append(url)

        if self._raises is not None:
            raise self._raises

        return page(self._links)

    async def aclose(self) -> None:
        self.closed = True


async def test_a_page_with_links_is_not_rendered() -> None:
    """The whole point: do not spend a browser on pages that work."""

    static = FakeStatic(
        tuple(f"https://other.test/{i}" for i in range(20))
    )
    browser = FakeBrowser(
        tuple(f"https://rendered.test/{i}" for i in range(90))
    )
    fetcher = RenderFallbackFetcher(static, browser)  # type: ignore[arg-type]

    result = await fetcher.fetch("https://example.com/")

    assert browser.calls == [], "a full page must not be rendered"
    assert len(result.links) == 20


async def test_an_empty_page_is_rendered() -> None:
    static = FakeStatic()
    browser = FakeBrowser(
        (
            "https://instagram.com/her",
            "https://x.com/her",
        )
    )
    fetcher = RenderFallbackFetcher(static, browser)  # type: ignore[arg-type]

    result = await fetcher.fetch("https://linktr.ee/her")

    assert browser.calls == ["https://linktr.ee/her"]
    assert "https://instagram.com/her" in result.links


async def test_a_render_that_finds_nothing_does_not_lose_the_static_page() -> None:
    """Rendering can produce an error page; the static one still stands."""

    static = FakeStatic(("https://keep.test/a",))
    browser = FakeBrowser(("https://extra.test/b",))
    # Only one link, under the threshold, so it does render.
    fetcher = RenderFallbackFetcher(
        static,
        browser,  # type: ignore[arg-type]
        thin_links=5,
    )
    result = await fetcher.fetch("https://example.com/")

    assert result.links == ("https://keep.test/a",)


async def test_a_broken_render_leaves_the_static_page_alone() -> None:
    static = FakeStatic()
    browser = FakeBrowser(
        raises=RuntimeError("browser died"),
    )
    fetcher = RenderFallbackFetcher(static, browser)  # type: ignore[arg-type]

    result = await fetcher.fetch("https://example.com/")

    assert result.links == ()
    assert browser.calls, "the render should have been attempted"


async def test_a_failed_render_is_reported_not_hidden() -> None:
    """"Found nothing" and "could not look" must not read the same.

    The whole output of this tool is a list of accounts, so a render
    that failed quietly becomes a claim that a performer has no other
    profiles. Two runs of the same call gave 0 and 5 links before this
    was surfaced, with nothing in the output to tell them apart.
    """

    browser = FakeBrowser(raises=RuntimeError("timeout"))
    fetcher = RenderFallbackFetcher(
        FakeStatic(),
        browser,  # type: ignore[arg-type]
    )

    await fetcher.fetch("https://linktr.ee/her")

    failures = fetcher.take_failures()

    assert len(failures) == 1
    url, why = failures[0]
    assert url == "https://linktr.ee/her"
    assert "RuntimeError" in why


async def test_a_missing_browser_is_reported_too() -> None:
    fetcher = RenderFallbackFetcher(FakeStatic(), None)

    await fetcher.fetch("https://linktr.ee/her")

    assert fetcher.take_failures() == [
        ("https://linktr.ee/her", "no browser installed")
    ]


async def test_a_good_render_reports_nothing() -> None:
    browser = FakeBrowser(("https://instagram.com/her",))
    fetcher = RenderFallbackFetcher(
        FakeStatic(),
        browser,  # type: ignore[arg-type]
    )

    await fetcher.fetch("https://linktr.ee/her")

    assert fetcher.take_failures() == []


async def test_no_browser_means_no_error() -> None:
    """Missing Playwright is a degraded tool, not a failed page."""

    static = FakeStatic()
    fetcher = RenderFallbackFetcher(static, None)

    result = await fetcher.fetch("https://example.com/")

    assert result.links == ()
    assert len(static.calls) == 1


async def test_an_unavailable_browser_is_never_asked() -> None:
    static = FakeStatic()
    browser = FakeBrowser(available=False)
    fetcher = RenderFallbackFetcher(static, browser)  # type: ignore[arg-type]

    await fetcher.fetch("https://example.com/")

    assert browser.calls == []


async def test_closing_reaches_the_browser() -> None:
    browser = FakeBrowser()
    fetcher = RenderFallbackFetcher(
        FakeStatic(),
        browser,  # type: ignore[arg-type]
    )

    await fetcher.aclose()

    assert browser.closed


def test_policy_refuses_nonsense() -> None:
    for kwargs in (
        {"timeout": 0},
        {"settle_ms": -1},
        {"max_links": 0},
        {"max_pages": 0},
    ):
        with pytest.raises(ValueError):
            RenderPolicy(**kwargs)


def test_a_negative_thin_page_threshold_is_refused() -> None:
    with pytest.raises(ValueError):
        RenderFallbackFetcher(FakeStatic(), thin_links=-1)


def test_the_default_threshold_is_not_trivial() -> None:
    """A one-link page is almost always an unrendered shell.

    If this drops to 1, every single page pays for a browser.
    """

    assert THIN_PAGE_LINKS > 2


@pytest.mark.parametrize(
    "link",
    [
        "javascript: void(0)",
        "about:blank",
        "data:text/html,x",
        "chrome-extension://abc/page.html",
    ],
)
def test_browser_chrome_is_filtered_out(link: str) -> None:
    assert _is_noise(link)


@pytest.mark.parametrize(
    "link",
    [
        "https://example.com/real",
        "http://example.com/real",
    ],
)
def test_real_links_survive_the_filter(link: str) -> None:
    assert not _is_noise(link)


async def test_the_browser_refuses_a_loopback_entry_url() -> None:
    """The entry URL is checked before a browser is ever started."""

    fetcher = BrowserFetcher()

    with pytest.raises(WebPolicyError):
        await fetcher.fetch("http://127.0.0.1:9999/graphql")


async def test_every_subresource_is_checked_not_just_the_entry_url() -> None:
    """The reason the filter exists at all.

    A page can pull an image from the metadata address while its own
    URL is a perfectly ordinary public site. Nothing about the entry
    check catches that, because the request never goes through it.
    """

    fetcher = BrowserFetcher()
    fetcher._host_verdicts["cdn.example.com"] = True

    assert await fetcher._host_allowed(
        "https://cdn.example.com/app.js"
    )

    # A host that resolves into private space is refused, and cached so
    # the page cannot ask again.
    assert not await fetcher._host_allowed(
        "http://127.0.0.1:9999/secret"
    )
    assert fetcher._host_verdicts["127.0.0.1"] is False

    # A verdict is remembered rather than re-resolved per request, and
    # a cached refusal stays a refusal.
    assert not await fetcher._host_allowed(
        "http://127.0.0.1:9999/other"
    )
    assert fetcher._host_verdicts["127.0.0.1"] is False

    # Page-local schemes never leave the machine.
    assert not await fetcher._host_allowed("about:blank")
    assert not await fetcher._host_allowed("blob:https://x.test/1")


async def test_a_missing_browser_is_reported_as_unavailable() -> None:
    """Not as a failed page, so the caller falls back."""

    fetcher = BrowserFetcher(executable_path="/nonexistent/chromium")

    with pytest.raises(RenderUnavailable):
        await fetcher.fetch("https://example.com/")


def test_availability_is_false_without_playwright() -> None:
    import sys

    saved = sys.modules.get("playwright")
    sys.modules["playwright"] = None  # type: ignore[assignment]

    try:
        assert BrowserFetcher(executable_path="/usr/bin/chromium").available is False
    finally:
        if saved is None:
            sys.modules.pop("playwright", None)
        else:
            sys.modules["playwright"] = saved


def test_resolve_executable_rejects_a_bad_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "AGENTOFLOW_CHROMIUM",
        "/definitely/not/here/chromium",
    )

    with pytest.raises(RenderUnavailable):
        resolve_executable()


@pytest.mark.render
@pytest.mark.skipif(
    not RUN_LIVE_RENDER_TESTS or not BrowserFetcher().available,
    reason="set AGENTOFLOW_RENDER_TESTS=1 with a browser installed",
)
async def test_a_real_render_finds_links_static_html_does_not_have() -> None:
    """Opt-in: proves the browser path end to end.

    Everything that can be checked without a browser or a network is
    checked above; this only exists to catch a render that silently
    returns an empty page.
    """

    browser = BrowserFetcher(
        render_policy=RenderPolicy(timeout=45.0, settle_ms=3000)
    )

    try:
        result = await browser.fetch("https://example.com/")
    finally:
        await browser.aclose()

    assert result.status_code == 200
    assert result.links, "a rendered page should carry links"
    assert all(not _is_noise(link) for link in result.links)

class RefusingStatic:
    """Stands in for httpx being told 403 by a host that a browser likes.

    This is what babepedia and iafd actually do: the static path is
    refused outright, so a fallback that only handles a thin page never
    gets to try the browser at all.
    """

    timeout = 30.0

    def __init__(self) -> None:
        self.calls = 0

    async def fetch(self, url: str, *args, **kwargs):
        self.calls += 1
        raise RuntimeError("HTTP 403 for " + url)


async def test_a_refused_static_fetch_still_reaches_the_browser() -> None:
    static = RefusingStatic()
    browser = FakeBrowser(
        (
            "https://babepedia.com/related/1",
            "https://instagram.com/her",
        )
    )
    fetcher = RenderFallbackFetcher(
        static,
        browser,  # type: ignore[arg-type]
    )

    result = await fetcher.fetch("https://www.babepedia.com/babe/her")

    assert browser.calls == ["https://www.babepedia.com/babe/her"]
    assert "https://instagram.com/her" in result.links


async def test_a_refusal_with_no_browser_still_raises() -> None:
    """The original error must survive, not become an empty page."""

    fetcher = RenderFallbackFetcher(RefusingStatic(), None)

    with pytest.raises(RuntimeError, match="403"):
        await fetcher.fetch("https://www.babepedia.com/babe/her")


async def test_a_refusal_is_recorded_for_the_report() -> None:
    fetcher = RenderFallbackFetcher(
        RefusingStatic(),
        FakeBrowser(("https://instagram.com/her",)),
    )  # type: ignore[arg-type]

    await fetcher.fetch("https://www.babepedia.com/babe/her")

    # The render succeeded, so there is nothing to report even though
    # the static request was refused.
    assert fetcher.take_failures() == []


@pytest.mark.parametrize(
    "link",
    [
        "mailto:her@example.com",
        "javascript: login()",
        "tel:+15550100",
        "about:blank",
        "data:text/html,x",
        "chrome-extension://abc/x.html",
    ],
)
def test_non_web_links_never_reach_the_page(link: str) -> None:
    """WebPage canonicalises every link, and mailto: is not a URL.

    One mailto link on a page was enough to fail the whole fetch.
    """

    assert _is_noise(link)


@pytest.mark.parametrize(
    "link",
    [
        "https://instagram.com/her",
        "http://example.com/x",
    ],
)
def test_web_links_are_kept(link: str) -> None:
    assert not _is_noise(link)


def test_the_age_gate_is_off_unless_asked_for() -> None:
    """The click asserts something about a person.

    It stays opt-in even though the operator has said they are an
    adult, because the next person to run this should have to think
    about it rather than inherit someone else's decision.
    """

    assert RenderPolicy().confirm_age_gate is False
    assert RenderPolicy(confirm_age_gate=True).confirm_age_gate is True


async def test_failures_do_not_leak_into_the_next_call() -> None:
    """One performer's broken pages must not be reported on the next.

    The fetcher is built once and lives for the life of the process,
    so anything left on it is visible to whoever asks next. This is
    how a search for LikaBusy reported failures for a page belonging
    to someone else entirely, and duplicated them at that.
    """

    browser = FakeBrowser(raises=RuntimeError("timeout"))
    fetcher = RenderFallbackFetcher(
        FakeStatic(),
        browser,  # type: ignore[arg-type]
    )

    await fetcher.fetch("https://linktr.ee/first")
    first = fetcher.take_failures()

    assert len(first) == 1
    assert fetcher.render_failures == {}

    # A second caller asking about their own URLs sees nothing.
    second = fetcher.take_failures(
        ["https://linktr.ee/second"]
    )

    assert second == []


async def test_a_page_that_rendered_is_not_reported_as_failed() -> None:
    """The 403 that would not go away.

    babepedia and iafd refuse httpx and open normally in a browser. The
    first version recorded the refusal and never cleared it, so the
    report kept claiming pages were unreadable that had in fact been
    read -- the complaint that "it still shows 403" was the report
    lying, not the fetch failing.
    """

    class Refuses(FakeStatic):
        async def fetch(self, url, *a, **k):
            raise RuntimeError("HTTP 403 for " + url)

    fetcher = RenderFallbackFetcher(
        Refuses(),
        FakeBrowser(("https://instagram.com/her",)),
    )  # type: ignore[arg-type]

    result = await fetcher.fetch("https://www.babepedia.com/babe/her")

    assert "https://instagram.com/her" in result.links
    assert fetcher.take_failures() == []


async def test_take_failures_reports_the_original_refusal() -> None:
    """When it really does fail, say both halves."""

    fetcher = RenderFallbackFetcher(
        RefusingStatic(),
        FakeBrowser(raises=RuntimeError("TimeoutError")),
    )  # type: ignore[arg-type]

    # Unreadable both ways, so the page really is unreadable and the
    # caller has to be told rather than handed an empty page.
    with pytest.raises(RuntimeError):
        await fetcher.fetch("https://www.iafd.com/person.rme/1")

    failures = fetcher.take_failures()

    assert len(failures) == 1
    url, why = failures[0]
    assert url == "https://www.iafd.com/person.rme/1"
    assert "403" in why
    assert "TimeoutError" in why

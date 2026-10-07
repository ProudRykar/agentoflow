"""Reading pages that only exist after JavaScript runs.

A link-in-bio page is the cheapest possible source of social accounts:
one page lists every profile its owner collected. But those pages, and
the social networks themselves, serve an empty shell. Ask for the HTML
and you get a few hundred bytes of scaffolding and a login prompt, which
is why the static fetcher found nothing on exactly the pages this is
built for.

So this renders, in a real browser, and returns the same ``WebPage`` the
static path returns. Nothing downstream changes: the crawler, the batch
fetcher and the social expander keep reading ``page.links``.

What this deliberately does not do
----------------------------------
It does not defeat bot detection. X, Instagram and most of the large
aggregators answer a headless browser with a 403 just as they answer
httpx, and working around that means forging fingerprints or solving
challenges on the user's behalf. That is out of scope. What this fixes
is the class of sites that are merely JavaScript, not sites that are
actively refusing.

It also does not click through age gates or consent walls. "I'm over
18" is an attestation made by a person, and an agent that auto-confirms
it is making a legal statement the user never gave. Where a gate hides
the links, they stay hidden and the tool reports the page as blocked.

Why the request filter is not optional
--------------------------------------
``WebFetcher`` validates the entry URL and every redirect, because
httpx cannot do anything a checked URL does not allow. A browser can:
one page can pull an image from 169.254.169.254, a script from
127.0.0.1:9999, or a stylesheet from an internal host, and none of that
goes through the entry URL check. Every single request the page makes is
therefore validated here, and blocking is the fail-closed default.
"""

from __future__ import annotations

import asyncio
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_workflow.core.entities.models.builtin.web_crawl import (
    WebFetcher,
    WebPage,
)
from agent_workflow.core.entities.models.web_policy import (
    WebPolicy,
    WebPolicyError,
)

# Request types that cannot contribute a link and are the usual way a
# page reaches something it should not. Blocking them also makes a
# render several times faster.
HEAVY_RESOURCE_TYPES = frozenset(
    {"image", "media", "font", "stylesheet", "manifest"}
)

SYSTEM_CHROMIUM_PATHS = (
    "/usr/bin/chromium",
    "/usr/bin/chromium-browser",
    "/usr/bin/google-chrome",
    "/usr/bin/google-chrome-stable",
    "/usr/bin/chrome",
    "/snap/bin/chromium",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
)

# Playwright downloads its own browser here. If it is present, prefer it:
# it is the build Playwright is actually tested against.
PLAYWRIGHT_BROWSER_CACHE = Path(
    os.environ.get(
        "PLAYWRIGHT_BROWSERS_PATH",
        "~/.cache/ms-playwright",
    )
).expanduser()


class RenderUnavailable(Exception):
    """Raised when rendering was asked for but cannot be done.

    Separate from a failed render: this means the machine has no
    browser, so the caller should fall back to a static fetch rather
    than report that the page is unreadable.
    """


def resolve_executable() -> str | None:
    """Find a Chromium to drive, or None for Playwright's own.

    The environment variable is the escape hatch for a browser in an
    unusual place, and for a CI image that pins one.
    """

    override = os.environ.get("AGENTOFLOW_CHROMIUM")

    if override:
        if Path(override).exists():
            return override

        raise RenderUnavailable(
            f"AGENTOFLOW_CHROMIUM points at {override!r}, "
            "which does not exist"
        )

    if PLAYWRIGHT_BROWSER_CACHE.is_dir():
        # None asks Playwright for the browser it installed.
        return None

    for path in SYSTEM_CHROMIUM_PATHS:
        if Path(path).exists():
            return path

    return None


@dataclass(frozen=True, slots=True)
class RenderPolicy:
    """How much the browser is allowed to spend on one page."""

    timeout: float = 30.0
    settle_ms: int = 2500
    """Quiet period after load, for the last script to write links."""

    max_links: int = 600

    max_pages: int = 4
    """Pages rendered at once. A browser is not a web server."""

    confirm_age_gate: bool = False
    """Click through an "I am over 18" interstitial.

    Off by default, and the default should stay off: the click is an
    attestation about a person, and a tool that makes it on the user's
    behalf is making a claim they never typed. A person who is
    actually an adult can enable it knowingly, which is what it is for.

    What it does not do is solve a CAPTCHA, and it only ever clicks a
    button whose own text says the user is an adult.
    """

    block_resources: bool = True

    user_agent: str = (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36"
    )

    def __post_init__(self) -> None:
        if self.timeout <= 0:
            raise ValueError("timeout must be > 0")

        if self.settle_ms < 0:
            raise ValueError("settle_ms must be >= 0")

        if self.max_links <= 0:
            raise ValueError("max_links must be > 0")

        if self.max_pages <= 0:
            raise ValueError("max_pages must be > 0")


def _is_noise(link: str) -> bool:
    """True for anything that is not a fetchable web link.

    Everything here is filtered before the page is built rather than
    after, because WebPage canonicalises each link on construction and
    a mailto: or javascript: href is not a URL it can canonicalise --
    one mailto link on the page would fail the whole fetch.
    """

    return not link.startswith(("http://", "https://"))


class BrowserFetcher:
    """Renders one page and reports it as a ``WebPage``.

    The browser is started on first use and kept. Launching Chromium
    costs a few hundred milliseconds, which is the whole budget of a
    static fetch, so paying it per page would make the fallback slower
    than the thing it is falling back from.
    """

    def __init__(
        self,
        *,
        policy: WebPolicy | None = None,
        render_policy: RenderPolicy | None = None,
        executable_path: str | None | Any = None,
    ) -> None:
        self._policy = policy or WebPolicy()

        self._render = render_policy or RenderPolicy()

        # None is meaningful here: it means "use Playwright's own".
        self._executable = (
            executable_path
            if executable_path is not None
            else resolve_executable()
        )

        self._playwright: Any = None
        self._browser: Any = None
        self._lock = asyncio.Lock()
        self._semaphore = asyncio.Semaphore(
            self._render.max_pages
        )

        # A page asks for the same CDN host dozens of times, and
        # resolving it again for each one is most of the cost of the
        # filter. Verdicts are cached per host for the life of the
        # process; a host that was public a minute ago does not become
        # loopback in between two requests on one page.
        self._host_verdicts: dict[str, bool] = {}

    @property
    def available(self) -> bool:
        """Whether a render could plausibly be attempted."""

        try:
            import playwright  # noqa: F401
        except ImportError:
            return False

        if self._executable is None:
            return True

        return Path(self._executable).exists()

    async def _ensure_browser(self) -> Any:
        if self._browser is not None:
            return self._browser

        async with self._lock:
            if self._browser is not None:
                return self._browser

            try:
                from playwright.async_api import (
                    async_playwright,
                )
            except ImportError as exc:
                raise RenderUnavailable(
                    "JavaScript rendering needs Playwright. Install it "
                    "with: uv sync --extra render"
                ) from exc

            self._playwright = await async_playwright().start()

            launch: dict[str, Any] = {
                "headless": True,
                # Required to run as root in a container, which is how
                # this mostly runs.
                "args": [
                    "--no-sandbox",
                    "--disable-dev-shm-usage",
                    # The image, font and analytics traffic a render
                    # never looks at.
                    "--disable-extensions",
                    "--disable-background-networking",
                ],
            }

            if self._executable is not None:
                launch["executable_path"] = self._executable

            try:
                self._browser = (
                    await self._playwright.chromium.launch(**launch)
                )
            except Exception as exc:
                await self.aclose()
                raise RenderUnavailable(
                    "Could not start Chromium. With Playwright's own "
                    "browser: playwright install chromium. Or point "
                    "AGENTOFLOW_CHROMIUM at an existing one. "
                    f"({type(exc).__name__})"
                ) from exc

            return self._browser

    async def _host_allowed(self, url: str) -> bool:
        """Whether one request is allowed to leave the machine.

        Runs for every single request the page makes, including the
        subresources the entry-URL check never sees.
        """

        from urllib.parse import urlsplit

        parsed = urlsplit(url)

        if parsed.scheme not in ("http", "https"):
            # about:, blob:, data: never leave; they are page-local.
            return False

        host = (parsed.hostname or "").lower()

        if not host:
            return False

        cached = self._host_verdicts.get(host)

        if cached is not None:
            return cached

        try:
            await self._policy.validate_url(url)
        except (WebPolicyError, Exception):
            allowed = False
        else:
            allowed = True

        self._host_verdicts[host] = allowed

        return allowed

    async def _install_filter(self, page: Any) -> None:
        render = self._render

        async def handle(route: Any) -> None:
            # A filter that raises takes the navigation down with it,
            # and a page may legitimately ask for a scheme that has no
            # host at all (mailto:, a blob it just created). Blocking
            # those is the answer; letting the error escape is not.
            try:
                request = route.request
                heavy = (
                    request.resource_type in HEAVY_RESOURCE_TYPES
                )

                if render.block_resources and heavy:
                    await route.abort()
                    return

                if await self._host_allowed(request.url):
                    await route.continue_()
                else:
                    await route.abort()
            except Exception:
                try:
                    await route.abort()
                except Exception:
                    pass

        await page.route("**/*", handle)

    async def _confirm_age(self, page: Any) -> None:
        """Click an age interstitial, and only an age interstitial.

        Scoped to text that literally claims adulthood. A generic
        "Enter" or "Continue" is somebody's consent wall or paywall
        and is left alone, because those are not the user's to click
        for them either.
        """

        pattern = (
            r"(i'?m|iam|i am)\s*(over\s*)?18"
            r"|(over|above|at least)\s*18"
            r"|18\s*\+"
        )

        try:
            control = page.get_by_text(
                re.compile(pattern, re.IGNORECASE)
            ).first

            if await control.count() == 0:
                return

            await control.click(timeout=8000)
            await page.wait_for_timeout(
                min(self._render.settle_ms or 1000, 3000)
            )
        except Exception:
            # No gate, or a gate that would not move. Either way the
            # page is still read, just without whatever it hid.
            return

    async def fetch(self, url: str) -> WebPage:
        await self._policy.validate_url(url)

        browser = await self._ensure_browser()

        async with self._semaphore:
            context = await browser.new_context(
                user_agent=self._render.user_agent,
                java_script_enabled=True,
            )

            try:
                page = await context.new_page()

                await self._install_filter(page)

                response = await page.goto(
                    url,
                    wait_until="domcontentloaded",
                    timeout=self._render.timeout * 1000,
                )

                # Best effort. A page with polling never goes idle, and
                # the links are usually there the moment load fires.
                try:
                    await page.wait_for_load_state(
                        "networkidle",
                        timeout=self._render.settle_ms,
                    )
                except Exception:
                    pass

                if self._render.settle_ms:
                    await page.wait_for_timeout(
                        self._render.settle_ms
                    )

                if self._render.confirm_age_gate:
                    await self._confirm_age(page)

                html = ""
                title = ""
                links: list[str] = []

                # Each of these can independently fail on a hostile
                # page. A title that will not render must not cost us
                # the links, which are the entire point.
                try:
                    html = await page.content()
                except Exception:
                    pass

                try:
                    title = await page.title()
                except Exception:
                    pass

                try:
                    raw = await page.eval_on_selector_all(
                        "a[href]",
                        "els => els.map(e => e.href)",
                    )
                    links = list(raw or ())
                except Exception:
                    pass
            finally:
                await context.close()

        kept: list[str] = []

        for link in links or ():
            if not isinstance(link, str):
                continue

            link = link.strip()

            if not link or _is_noise(link):
                continue

            kept.append(link)

            if len(kept) >= self._render.max_links:
                break

        return WebPage(
            url=url,
            title=title or "",
            content=html or "",
            links=tuple(kept),
            status_code=response.status
            if response is not None
            else 200,
            content_type="text/html",
            content_bytes=len(html or ""),
        )

    async def aclose(self) -> None:
        if self._browser is not None:
            try:
                await self._browser.close()
            except Exception:
                pass

            self._browser = None

        if self._playwright is not None:
            try:
                await self._playwright.stop()
            except Exception:
                pass

            self._playwright = None


# A static page under this many links is a page that has not finished
# loading yet. Twelve covers real prose pages, which do have navigation
# and a handful of body links; anything thinner is a shell.
THIN_PAGE_LINKS = 12


class RenderFallbackFetcher:
    """Static first, browser only where static came back empty.

    Rendering costs seconds against milliseconds, and most pages do not
    need it. But a link-in-bio page is the one place it pays, so the
    rule is deliberately dumb: if the static page had almost no links,
    it did not render for us, and try again with a real browser.
    """

    def __init__(
        self,
        static: WebFetcher,
        browser: BrowserFetcher | None = None,
        *,
        thin_links: int = THIN_PAGE_LINKS,
        render_policy: RenderPolicy | None = None,
    ) -> None:
        if thin_links < 0:
            raise ValueError("thin_links must be >= 0")

        self._static = static
        self._browser = browser
        self._thin = thin_links
        self._render_policy = render_policy
        # A render that fails and falls back silently is
        # indistinguishable from a page that genuinely has no links,
        # and this tool's whole output is "no links found". Keep the
        # failures so the caller can say which it was.
        #
        # Keyed by URL and holding the *final* outcome, not a log. An
        # earlier version appended on every static refusal and never
        # cleared it, so a page that the browser then read perfectly
        # was reported as unreadable, and because this fetcher outlives
        # a single call, the next performer was told about the last
        # one's failures. Both were wrong in the same direction: the
        # report claimed less than had happened.
        self.render_failures: dict[str, str] = {}

    @property
    def timeout(self) -> float:
        return self._static.timeout

    async def fetch(
        self,
        url: str,
        *args: Any,
        **kwargs: Any,
    ) -> WebPage:
        # A refusal is not a final answer. Some hosts answer httpx with
        # a 403 simply because it is httpx, and a browser gets the same
        # page without trouble, so a failed static fetch has to reach
        # the browser too -- otherwise the fallback only ever helps
        # pages that came back empty, which is the rarer problem.
        try:
            page = await self._static.fetch(url, *args, **kwargs)
        except Exception as exc:
            # The refusal is worth remembering only if nothing else
            # gets to read the page. A 403 from httpx on a host that a
            # browser opens without trouble is not a problem to report.
            static_error = exc

            self.render_failures.pop(url, None)

            if self._browser is None or not self._browser.available:
                if self._browser is None:
                    self.render_failures[url] = "no browser installed"

                raise

            return await self._render(
                url,
                static_error=static_error,
            )

        if len(page.links or ()) >= self._thin:
            return page

        if self._browser is None or not self._browser.available:
            if self._browser is None:
                # The page is thin and nothing can fix it, so its
                # script-drawn links are simply missing. Say so rather
                # than let a thin page read as "no accounts".
                self.render_failures[url] = "no browser installed"

            return page

        return await self._render(url, keep=page)

    async def _render(
        self,
        url: str,
        keep: WebPage | None = None,
        static_error: Exception | None = None,
    ) -> WebPage:
        """Try the browser, keeping the static page if it does no better."""

        if self._browser is None:
            if keep is not None:
                return keep

            raise RenderUnavailable("no browser configured")

        try:
            rendered = await self._browser.fetch(url)
        except RenderUnavailable as exc:
            self._record(url, static_error, str(exc))

            if keep is not None:
                return keep

            raise
        except Exception as exc:
            # A render that fails is not a reason to lose a page the
            # static path already has.
            self._record(
                url,
                static_error,
                f"{type(exc).__name__}: "
                f"{str(exc).splitlines()[0][:100]}",
            )

            if keep is not None:
                return keep

            raise

        # It rendered. Whatever the static path said is now moot, and
        # leaving it on the record is what made a working page read as
        # a failed one.
        self.render_failures.pop(url, None)

        # Keep whichever answer actually knows more, not merely the
        # rendered one: some pages render to an error page.
        if keep is not None and (
            len(rendered.links or ()) <= len(keep.links or ())
        ):
            return keep

        return rendered

    def _record(
        self,
        url: str,
        static_error: Exception | None,
        render_error: str,
    ) -> None:
        """Note a page that could not be read at all."""

        why = render_error

        if static_error is not None:
            why = (
                f"static fetch refused "
                f"({str(static_error).splitlines()[0][:70]}); "
                f"render: {render_error}"
            )

        self.render_failures[url] = why

    def take_failures(
        self,
        urls: list[str] | None = None,
    ) -> list[tuple[str, str]]:
        """The failures for these URLs, and only these.

        Scoping matters: the fetcher is shared for the life of the
        process, so anything left on it leaks into the next caller's
        report.
        """

        if urls is None:
            items = list(self.render_failures.items())
            self.render_failures.clear()
            return items

        wanted = {u.rstrip("/") for u in urls}
        items = [
            (url, why)
            for url, why in self.render_failures.items()
            if url.rstrip("/") in wanted
        ]

        for url, _why in items:
            self.render_failures.pop(url, None)

        return items

    async def aclose(self) -> None:
        if self._browser is not None:
            await self._browser.aclose()


__all__ = [
    "BrowserFetcher",
    "RenderFallbackFetcher",
    "RenderPolicy",
    "RenderUnavailable",
    "THIN_PAGE_LINKS",
    "resolve_executable",
]
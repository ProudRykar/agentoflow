"""Check a name against many platforms in one call.

The failure this replaces: asking a model to sweep forty platforms
makes it loop, lose count, and then narrate a result it never got.
A 4B model will do all three. It will say a search happened when none
did, invent a Linktree that no tool ever mentioned, and declare the
task finished having answered a fraction of it.

So the sweep is done here. One query per platform, run concurrently,
and every platform gets a verdict that distinguishes three outcomes:

- **found** -- results came back
- **none** -- the search ran and matched nothing
- **error** -- the search did not run, and the reason is given

Collapsing "none" and "error" is what produces confident wrong
answers, so they are never reported as the same thing.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field, replace
from typing import Any
from urllib.parse import urlsplit

from agent_workflow.core.entities.models.searxng_search import (
    SearxngSearch,
)
from agent_workflow.core.entities.models.tool import (
    Tool,
    ToolContext,
    ToolPolicy,
)


class PlatformScanError(RuntimeError):
    """The sweep could not run."""


@dataclass(slots=True, frozen=True)
class ScanInput:
    """Arguments for ``web_platform_scan``."""

    name: str = ""
    """The name to look for, quoted in every query."""

    sites: list[str] | None = None
    """Platform names to check. Empty uses the built-in list."""

    aliases: list[str] | None = None
    """Tried in turn when the name alone matches nothing."""

    known_urls: list[str] | None = None
    """URLs already held elsewhere, usually a library record.

    Given them, the report separates what was already known from what
    is new. Comparing a dozen found links against a dozen stored ones by
    hand is where this goes wrong: it is a set difference, and doing it
    from memory produces a confident list with silent omissions.
    """

    per_site: int = 2
    """Results to keep per platform."""

    concurrency: int = 2
    """Parallel searches. Low on purpose; see ScanPolicy."""


@dataclass(slots=True, frozen=True)
class PlatformTarget:
    """Where to look for one platform."""

    label: str
    site: str
    note: str = ""
    """Why a platform may legitimately return nothing."""


@dataclass(slots=True, frozen=True)
class PlatformResult:
    """The verdict for one platform."""

    label: str
    site: str
    note: str
    urls: tuple[str, ...] = ()
    matched_with: str = ""
    error: str = ""
    degraded: tuple[str, ...] = ()
    """Engines that did not answer for this platform."""
    known: tuple[str, ...] = ()
    """Urls that were already supplied as known."""

    @property
    def new_urls(self) -> tuple[str, ...]:
        # Both sides go through _key: `known` holds the raw url, so
        # comparing it against a normalised form never matches and
        # every known link comes back as a discovery.
        already = {_key(item) for item in self.known}

        return tuple(url for url in self.urls if _key(url) not in already)

    @property
    def status(self) -> str:
        if self.error:
            return "error"

        if self.urls:
            return "found"

        # An empty answer from an instance whose engines were timing out
        # is not evidence of absence, and must never be reported as it.
        if self.degraded:
            return "incomplete"

        return "none"


# `site:` target per platform. Several of these cannot be covered by a
# single host and say so, because a blank here is not evidence of
# absence.
BUILTIN_PLATFORMS: tuple[PlatformTarget, ...] = (
    PlatformTarget("X", "twitter.com"),
    PlatformTarget("X", "x.com"),
    PlatformTarget("Reddit", "reddit.com"),
    PlatformTarget("Bluesky", "bsky.app"),
    PlatformTarget("Bluesky", "bsky.social"),
    PlatformTarget("Telegram", "t.me"),
    PlatformTarget("Mastodon", "mastodon.social", "one instance only"),
    PlatformTarget("Mastodon", "fosstodon.org", "one instance only"),
    PlatformTarget("Tumblr", "tumblr.com"),
    PlatformTarget("FetLife", "fetlife.com", "login-walled, rarely indexed"),
    PlatformTarget("Snapchat", "snapchat.com"),
    PlatformTarget("Instagram", "instagram.com"),
    PlatformTarget("Facebook", "facebook.com"),
    PlatformTarget("Threads", "threads.net"),
    PlatformTarget("Pinterest", "pinterest.com"),
    PlatformTarget("VK", "vk.com"),
    PlatformTarget("OK", "ok.ru"),
    PlatformTarget("Discord", "discord.com/invite", "invites only, rarely indexed"),
    PlatformTarget("Twitch", "twitch.tv"),
    PlatformTarget("Kick", "kick.com"),
    PlatformTarget("LinkedIn", "linkedin.com"),
    PlatformTarget("Quora", "quora.com"),
    PlatformTarget("Medium", "medium.com"),
    PlatformTarget("Patreon", "patreon.com"),
    PlatformTarget("Fansly", "fansly.com"),
    PlatformTarget("OnlyFans", "onlyfans.com", "login-walled"),
    PlatformTarget("ManyVids", "manyvids.com"),
    PlatformTarget("LoyalFans", "loyalfans.com"),
    PlatformTarget("JustForFans", "justforfans.com"),
    PlatformTarget("Fanvue", "fanvue.com"),
    PlatformTarget("Ko-fi", "ko-fi.com"),
    PlatformTarget("BuyMeACoffee", "buymeacoffee.com"),
    PlatformTarget("RedGIFs", "redgifs.com"),
    PlatformTarget("Pornhub", "pornhub.com"),
    PlatformTarget("XVideos", "xvideos.com"),
    PlatformTarget("xHamster", "xhamster.com"),
    PlatformTarget("SpankBang", "spankbang.com"),
    PlatformTarget("Chaturbate", "chaturbate.com"),
    PlatformTarget("Stripchat", "stripchat.com"),
    PlatformTarget("Cam4", "cam4.com"),
    PlatformTarget("TikTok", "tiktok.com"),
    PlatformTarget("YouTube", "youtube.com"),
)

_BY_LABEL = {target.label: target for target in BUILTIN_PLATFORMS}


@dataclass(slots=True)
class ScanPolicy:
    """Bounds on one sweep.

    Concurrency is deliberately low and the stagger real. Each query
    reaches five or more upstream engines, so a forty-target sweep at
    concurrency 4 is roughly eight hundred outbound requests in under
    half a minute. That is enough to have the engines start refusing,
    and a refusal comes back as an empty result set rather than as an
    error -- so the sweep comes back looking like "no accounts found
    anywhere" when in fact nothing was ever asked.
    """

    concurrency: int = 2
    per_site: int = 2
    max_sites: int = 60
    timeout: float = 300.0
    stagger: float = 0.4
    """Seconds to wait before each query, spread by position."""

    def __post_init__(self) -> None:
        if self.concurrency < 1:
            raise ValueError("concurrency must be >= 1")

        if self.per_site < 1:
            raise ValueError("per_site must be >= 1")

        if self.max_sites < 1:
            raise ValueError("max_sites must be >= 1")

        if self.timeout <= 0:
            raise ValueError("timeout must be > 0")

        if self.stagger < 0:
            raise ValueError("stagger must be >= 0")


def _key(url: str) -> str:
    """A comparable form of a link.

    Both sides go through this, because normalising only the known list
    compares a normalised form against a raw one and never matches.
    """

    text = url.strip().lower()

    text = text.removeprefix("https://").removeprefix("http://")
    text = text.removeprefix("www.")

    return text.rstrip("/")


def _normalise_known(urls: list[str] | None) -> set[str]:
    """Compare links the way a person does, not byte for byte.

    ``instagram.com/x`` and ``https://www.instagram.com/x/`` are the
    same link, and treating them as different would report a stored URL
    as a new discovery, which is the opposite of what was asked.
    """

    return {key for key in (_key(entry) for entry in urls or []) if key}


def _host(entry: str) -> str:
    """Reduce whatever was passed to a bare hostname.

    Callers hand over full profile URLs, having guessed the path:
    ``https://reddit.com/r/sweetiefox``. Turning that into a filter
    yields ``site:reddit.com/r/sweetiefox``, which matches nothing and
    looks like a clean miss. The path is exactly what the search is
    supposed to discover, so only the host is kept.
    """

    text = entry.strip()

    for prefix in ("https://", "http://"):
        if text.lower().startswith(prefix):
            text = text[len(prefix) :]
            break

    if not text:
        return ""

    host = urlsplit(text).netloc or urlsplit(f"//{text}").netloc

    if not host:
        # No scheme and no slash: "reddit.com/r/x" splits as a path,
        # so take the first segment as the host.
        host = text.split("/", 1)[0]

    return host.removeprefix("www.")


def resolve(sites: list[str] | None) -> list[PlatformTarget]:
    """Turn requested platform names into search targets.

    An unknown name is kept as a bare site filter, so a platform the
    caller cares about that is not in the table is still searchable
    rather than silently dropped.
    """

    if not sites:
        return list(BUILTIN_PLATFORMS)

    # Keyed on the lowercased form so duplicates collapse, but the
    # caller's own casing is kept: the report shows the name the user
    # wrote, not "somenewplatform".
    wanted: dict[str, str] = {}

    for entry in sites:
        cleaned = entry.strip()

        if cleaned:
            wanted.setdefault(cleaned.lower(), cleaned)

    resolved: list[PlatformTarget] = []

    for key, label in wanted.items():
        targets = [t for t in BUILTIN_PLATFORMS if t.label.lower() == key]

        if targets:
            resolved.extend(targets)
            continue

        resolved.append(PlatformTarget(label, _host(label)))

    return resolved


@dataclass(slots=True)
class PlatformScanner:
    """Runs one query per platform and keeps the verdicts."""

    search: SearxngSearch
    policy: ScanPolicy = field(default_factory=ScanPolicy)

    async def _one(
        self,
        target: PlatformTarget,
        name: str,
        aliases: list[str],
        semaphore: asyncio.Semaphore,
        offset: int = 0,
    ) -> PlatformResult:
        attempts = [name, *aliases][:3]

        last_error = ""
        engines: tuple[str, ...] = ()

        if self.policy.stagger:
            await asyncio.sleep(self.policy.stagger * offset)

        async with semaphore:
            for attempt in attempts:
                query = f'"{attempt}" site:{target.site}'

                try:
                    found, degraded = await self.search.search_detailed(
                        query,
                        limit=self.policy.per_site,
                    )
                except Exception as exc:  # noqa: BLE001
                    # Recorded rather than raised: one platform being
                    # unreachable must not lose the other forty.
                    last_error = str(exc).splitlines()[0][:200]
                    continue

                engines = degraded

                if found:
                    return PlatformResult(
                        label=target.label,
                        site=target.site,
                        note=target.note,
                        urls=tuple(
                            result.url for result in found[: self.policy.per_site]
                        ),
                        matched_with=attempt,
                    )

        return PlatformResult(
            label=target.label,
            site=target.site,
            note=target.note,
            error=last_error,
            degraded=engines,
        )

    async def scan(
        self,
        name: str,
        sites: list[str] | None = None,
        aliases: list[str] | None = None,
        known_urls: list[str] | None = None,
    ) -> tuple[list[PlatformResult], int]:
        """Every platform gets a verdict; nothing is left unstated."""

        if not name.strip():
            raise PlatformScanError("give a name to search for")

        targets = resolve(sites)[: self.policy.max_sites]

        semaphore = asyncio.Semaphore(self.policy.concurrency)
        extra = [alias.strip() for alias in (aliases or []) if alias.strip()]

        results = await asyncio.gather(
            *(
                self._one(target, name.strip(), extra, semaphore, offset)
                for offset, target in enumerate(targets)
            )
        )

        known = _normalise_known(known_urls)

        if known:
            results = [
                replace(
                    result,
                    known=tuple(
                        url for url in result.urls if _key(url) in known
                    ),
                )
                for result in results
            ]

        return list(results), len(targets)


def format_scan(
    results: list[PlatformResult],
    checked: int,
    name: str,
    *,
    filtered: bool = False,
) -> str:
    """Report every platform, found or not, with the distinction kept."""

    found = [r for r in results if r.status == "found"]
    none = [r for r in results if r.status == "none"]
    incomplete = [r for r in results if r.status == "incomplete"]
    errored = [r for r in results if r.status == "error"]

    lines: list[str] = []

    lines.append(f"Platform sweep for {name!r}: {checked} targets checked.")
    lines.append(
        f"found {len(found)}, no results {len(none)}, "
        f"incomplete {len(incomplete)}, search failed {len(errored)}."
    )

    if results and len(incomplete) == len(results):
        # Every line below is a non-answer, so say so first rather than
        # leaving a reader to work it out from a wall of text.
        lines.append("")
        lines.append(
            "NOTHING WAS ACTUALLY ASKED. Every search came back with "
            "no answer from the engines behind the instance, so this "
            "run tells you nothing about any platform. It is almost "
            "always too many targets at once. Wait a few minutes and "
            "sweep six to ten platforms."
        )
    lines.append("")
    lines.append(
        "'no results' means the search ran and matched nothing. It is "
        "not proof the account does not exist: the profile may be "
        "private, walled behind a login, on a domain not covered "
        "here, or simply not indexed. 'search failed' means the "
        "lookup did not run at all, which is different and is worth "
        "retrying."
    )

    if filtered:
        # The usual ask: what is not already on file.
        fresh: list[PlatformResult] = [
            result for result in found if result.new_urls
        ]

        lines.append("")

        if fresh:
            lines.append("NEW, not in the list supplied:")

            for result in fresh:
                suffix = (
                    f"  [matched on {result.matched_with!r}]"
                    if result.matched_with
                    and result.matched_with.lower() != name.lower()
                    else ""
                )

                lines.append(f"  {result.label}{suffix}")

                for url in result.new_urls:
                    lines.append(f"    {url}")

            lines.append("")
            lines.append(
                f"{sum(len(r.new_urls) for r in fresh)} new links across "
                f"{len(fresh)} platforms."
            )
        elif found:
            lines.append(
                "NEW, not in the list supplied: none. Every link found "
                "was already in the list."
            )
        else:
            # Nothing was found at all, so nothing was ruled out either.
            # Saying "every link found was already known" would read
            # as a completed comparison when no comparison happened.
            lines.append(
                "NEW, not in the list supplied: cannot say. No links "
                "were found in this run, so nothing was confirmed new "
                "and nothing was confirmed already known."
            )

    if found:
        lines.append("")
        lines.append("FOUND:")

        for result in found:
            suffix = f"  [matched on {result.matched_with!r}]" if (
                result.matched_with and result.matched_with.lower() != name.lower()
            ) else ""

            lines.append(f"  {result.label} ({result.site}){suffix}")

            for url in result.urls:
                mark = "  (already known)" if url in result.known else ""

                lines.append(f"    {url}{mark}")

    if incomplete:
        lines.append("")
        lines.append(
            "INCOMPLETE, no answer from the engines behind these. "
            "Treat them as unknown, not as absent:"
        )

        for result in incomplete:
            engines = ", ".join(result.degraded[:6])

            lines.append(
                f"  {result.label} ({result.site}): engines did not "
                f"answer ({engines})"
            )

    if none:
        lines.append("")
        lines.append("NO RESULTS:")

        for result in none:
            note = f"  ({result.note})" if result.note else ""

            lines.append(f"  {result.label} ({result.site}){note}")

    if errored:
        lines.append("")
        lines.append("SEARCH FAILED:")

        for result in errored:
            lines.append(f"  {result.label} ({result.site}): {result.error}")

    return "\n".join(lines)


def create_web_platform_scan_tool(
    search: SearxngSearch,
) -> Tool:
    """One call that answers the whole platform list.

    Without this the sweep is forty tool calls and a synthesis, which
    is exactly the shape of task a small model cannot be trusted with:
    it loses the count, skips platforms, and then describes a search
    that never happened.
    """

    async def handler(
        arguments: ScanInput,
        context: ToolContext,
    ) -> str:
        del context

        scanner = PlatformScanner(
            search,
            ScanPolicy(
                concurrency=max(1, arguments.concurrency),
                per_site=max(1, arguments.per_site),
                timeout=search.policy.timeout * 4,
            ),
        )

        results, checked = await scanner.scan(
            arguments.name,
            arguments.sites,
            arguments.aliases,
            arguments.known_urls,
        )

        return format_scan(
            results,
            checked,
            arguments.name,
            filtered=bool(arguments.known_urls),
        )

    return Tool(
        name="web_platform_scan",
        description=(
            "Look for one name across several platforms and report a "
            "verdict for every one of them. Use it whenever the "
            "request is a list of sites to check, rather than a "
            "general question.\n"
            "\n"
            "Pass PLATFORM NAMES, never URLs. \"Reddit\", "
            "\"Bluesky\", \"Ko-fi\" are what this takes; "
            "\"reddit.com/r/someone\" is not, because guessing the "
            "profile path is the one thing this tool exists to do for "
            "you. A URL is reduced to its host, which searches the "
            "whole site instead of one guessed profile. An unrecognised "
            "name is searched rather than dropped.\n"
            "\n"
            "Pass name as the name itself and nothing else. Appending a "
            "role or a description makes it a phrase that matches "
            "nothing. Quote the name when it is uncertain, and list "
            "aliases to fall back on.\n"
            "\n"
            "Sweep six to ten platforms per call, not forty. Each "
            "target reaches several upstream engines, and asking for "
            "everything at once makes them refuse, which comes back as "
            "no answer rather than as an error. The output says so "
            "when that has happened.\n"
            "\n"
            "Pass known_urls with anything already on file when the "
            "ask is for what is missing: the report then separates new "
            "links from known ones instead of leaving the comparison to "
            "be done by memory.\n"
            "\n"
            "Report the sweep as it came back. 'no results' is not "
            "proof of absence, 'incomplete' means nothing was asked, "
            "and 'search failed' is neither of those."
        ),
        input_type=ScanInput,
        handler=handler,
        policy=ToolPolicy(
            permissions=frozenset({
                "web.search",
            }),
            # A sweep of forty platforms is deliberately slow.
            timeout=300.0,
            max_output_size=60_000,
        ),
    )


__all__: list[Any] = [
    "BUILTIN_PLATFORMS",
    "PlatformResult",
    "PlatformScanError",
    "PlatformScanner",
    "PlatformTarget",
    "ScanInput",
    "ScanPolicy",
    "create_web_platform_scan_tool",
    "format_scan",
    "resolve",
]
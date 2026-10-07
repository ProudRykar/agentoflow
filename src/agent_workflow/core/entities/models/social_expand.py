"""Walk the links a library already holds and read the accounts off them.

The platform sweep asks a search engine forty questions and burns the
quota that the engines meter by address. This does the opposite: it
opens the pages the record already points at and reads the links out of
them.

That is the cheaper direction, because those pages were built to list
accounts. A link-in-bio page is a directory of every profile someone
wants collected in one; an aggregator page built for this exact
purpose is the same thing a third time over. None of it touches a
search engine, so none of it competes with anything else for the
address's quota.

It also finds what a name search cannot. Searching for a performer
finds pages about her, which link onwards; fetching a page about her
finds the accounts themselves, in one hop.

What a hit does not prove is identity. A page can link an account for
someone else, and a search for a name that is also a common word finds
pages about a different thing entirely. Compare a returned handle
against the record before believing it.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit

from agent_workflow.core.entities.models.platform_scan import BUILTIN_PLATFORMS
from agent_workflow.core.entities.models.tool import (
    Tool,
    ToolContext,
    ToolPolicy,
)


class SocialExpandError(RuntimeError):
    """The sources could not be walked."""


@dataclass(slots=True)
class ExpandInput:
    """Arguments for ``web_social_expand``."""

    urls: list[str] = field(default_factory=list)
    """The links already on file, usually a record's urls field."""

    name: str = ""
    """Who this is about, used only for the report heading."""

    platforms: list[str] | None = None
    """Platform names to keep. Empty keeps every recognised one."""

    per_page: int = 40
    """Links to read per page."""

    hops: int = 2
    """How many links to follow.

    The second hop is the useful one: the record points at a profile,
    the profile points at a directory, and the directory lists every
    account. Only hosts known to be directories are followed, so this
    stays a handful of pages rather than a crawl.
    """


@dataclass(slots=True, frozen=True)
class ExpandPolicy:
    """Bounds on one walk.

    These are page fetches rather than searches, so the ceiling here is
    about politeness to the hosts being read, not about a quota that
    runs out. The one that does run out is patience: a page that will
    not answer is reported as such.
    """

    concurrency: int = 4
    max_sources: int = 30
    max_directories: int = 6
    per_page: int = 40
    timeout: float = 180.0
    hops: int = 2

    def __post_init__(self) -> None:
        if self.concurrency < 1:
            raise ValueError("concurrency must be >= 1")

        if self.max_sources < 1:
            raise ValueError("max_sources must be >= 1")

        if self.per_page < 1:
            raise ValueError("per_page must be >= 1")

        if self.timeout <= 0:
            raise ValueError("timeout must be > 0")

        if self.hops < 1:
            raise ValueError("hops must be >= 1")


# Host to platform label, built from the sweep's table so the two tools
# cannot drift apart on what counts as a platform.
_HOST_LABELS: dict[str, str] = {}

for _target in BUILTIN_PLATFORMS:
    _HOST_LABELS.setdefault(_target.site.replace("www.", ""), _target.label)

# Hosts that are aggregators rather than accounts. They point onward
# and are worth keeping as such, but they are not a profile of hers.
_AGGREGATORS = frozenset({
    "pics-x.com",
    "indexxx.com",
    "socialmediapornstars.com",
    "imdb.com",
    "iafd.com",
    "babepedia.com",
    "wikipedia.org",
    "ru.wikipedia.org",
    "linktree.com",
    "allmylinks.com",
    "beacons.ai",
    "solo.to",
    "bio.link",
    "about.me",
    "imdb.name",
})


def classify(url: str) -> str:
    """The platform a link belongs to, or 'other'."""

    host = urlsplit(url).netloc.lower()

    if not host:
        return "other"

    bare = host.removeprefix("www.")

    for known in _HOST_LABELS:
        if bare == known or bare.endswith(f".{known}"):
            return _HOST_LABELS[known]

    # A profile host is not a platform anyone searches for, so it is
    # labelled by its own name rather than folded into "other".
    if bare in _PROFILE_HOSTS:
        return bare.removeprefix("www.")

    for parent in _PROFILE_PARENTS:
        if bare == parent or bare.endswith(f".{parent}"):
            return parent

    return "other"


def _is_aggregator(url: str) -> bool:
    """True for a directory page, and only for the directory itself.

    An exact host or its www variant: matching on any subdomain pulls
    in a CDN, whose "links" are images rather than accounts.
    """

    host = urlsplit(url).netloc.lower().removeprefix("www.")

    if urlsplit(url).path.lower().endswith(tuple(_ASSETS)):
        return False

    return host in _AGGREGATORS


# Site furniture, not accounts. A profile page links its own login,
# legal and footer sections, and a post links a share intent; without
# this every platform contributes a handful of rows that are obviously
# not a person.
_FURNITURE = frozenset({
    "about", "about-us", "abuse", "account", "accounts", "advertise",
    "auth", "careers", "contact", "cookie", "cookies", "dashboard",
    "developers", "download", "explore", "faq", "feedback", "guidelines",
    "help", "home", "intent", "jobs", "legal", "login", "logout",
    "notifications", "policies", "popular", "press", "privacy", "privacy-policy",
    "porn", "register", "search", "settings", "signup", "static",
    "support", "tags", "terms", "tos", "trending", "watch", "web",
    "lite", "challenge", "emulate", "directory", "hashtag", "topic",
})

# Furniture that turns up as a subdomain rather than a path segment.
# A single letter is left out on purpose: m.instagram.com and
# i.instagram.com are still profiles, and filtering those would drop
# real accounts instead of nav.
_FURNITURE_SUBDOMAINS = frozenset({
    "support", "help", "helpcenter", "help-centre", "business",
    "developer", "developers", "dev", "status", "blog", "press",
    "careers", "jobs", "legal", "privacy", "terms", "tos",
    "about", "account", "accounts", "auth", "login", "signup",
    "feedback", "abuse", "safety", "transparency", "policies",
    "settings", "notifications", "messages", "mobile",
    "web", "lite", "static", "cdn", "assets", "img", "media",
    "api", "graph", "internal", "staging", "beta", "admin",
})

# Account sites that are not in the sweep's platform table, but which
# are still a profile of hers and not a directory. Without these a
# link-in-bio page renders successfully, gets read, and then
# contributes nothing, because every link on it points at a host the
# table has never heard of -- which is most of what an adult performer's
# link page actually contains.
_PROFILE_HOSTS = frozenset({
    "stripschat.com",
    "selfymodels.com",
    "clips4sale.com",
    "skyprivate.com",
    "sheer.com",
    "sheerxxx.com",
    "fancentro.com",
    "chaturbate.com",
    "myfreecams.com",
    "camsoda.com",
    "flirt4free.com",
    "bongacams.com",
    "stripchat.com",
    "myyouporn.com",
    "motherless.com",
    "xhamster.com",
    "xvideos.com",
    "manyvids.com",
    "fansly.com",
    "onlyfans.com",
    "fansdb.cc",
    "imdb.com",
})

# Hosts where any subdomain is still a profile of the person, not a
# mirror of someone else's directory. These sites host a page per
# performer under its own subdomain.
_PROFILE_PARENTS = frozenset({
    "bezoxo.com",
    "hot.com",
    "iitFlix.com",
})

# Not a page, so never a directory to follow.
_ASSETS = frozenset({
    ".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".css", ".js",
    ".mp4", ".webm", ".ico",
})


# A profile page links its own tabs as well as its own login and legal
# sections. The tabs sit *after* the handle, so checking the first
# segment misses every one of them: x.com/her/photo is the same
# account as x.com/her, not a new account.
_PROFILE_TABS = frozenset({
    "about", "activity", "answers", "bookmarks", "collections",
    "comment", "compose", "contact", "emoji", "explore", "favorites",
    "followers", "following", "header_photo", "home", "i", "intent",
    "job", "jobs", "liked", "likes", "lists", "live", "login",
    "media", "mentions", "messages", "mobile", "more", "newsletters",
    "notifications", "photo", "photos", "pinned", "posts",
    "privacy", "replies", "reposted", "reposts", "search", "settings",
    "share", "signup", "status", "stories", "subscribe", "support",
    "tagged", "terms", "topics", "video", "videos", "with_replies",
    "verified_followers", "video", "view",
})

# Consent and login interstitials live on their own hosts but are the
# same page wearing a different domain, and they carry a "continue"
# query that looks like a profile.
_INTERSTITIAL_HOSTS = frozenset({
    "consent.youtube.com",
    "accounts.google.com",
    "consent.google.com",
    "hcaptcha.com",
    "accounts.hcaptcha.com",
})

# X serves its login and signup walls under an /i/jf/ prefix, which
# reads like a profile path and lands in the report as a new account.
_ONBOARDING_PREFIXES = (
    ("i", "jf"),
    ("i", "flow"),
)


def _variant_key(url: str) -> tuple[str, str, int]:
    """Identity of a page across its language variants.

    de.pornhub.com, rt.pornhub.com and pornhub.com are one profile in
    twelve languages, and reporting each separately buries the one
    link that was actually new.

    The leading label is dropped when it is short, because that is
    what separates a language code from a site name: "de" and "rt"
    are translations, while angieelif.bezoxo.com and
    angieelifdate.bezoxo.com are two different performers' sites and
    must stay apart. The label count is carried along so the shortest
    host can be kept as the representative.

    No list of language codes is involved, on purpose. One was tried
    and it was wrong three times in a single run (rt, then cn, then
    whatever came next).
    """

    parsed = urlsplit(url)
    host = parsed.netloc.lower().removeprefix("www.")
    labels = host.split(".")

    if len(labels) > 2 and _is_short_label(labels[0]):
        labels = labels[1:]

    return ".".join(labels), parsed.path.rstrip("/").lower(), len(
        parsed.netloc.lower().removeprefix("www.").split(".")
    )


def _is_short_label(label: str) -> bool:
    """Whether a subdomain looks like a code rather than a name."""

    return 0 < len(label) <= 3 and label.isalpha()


def _brand_token(host: str) -> str:
    """The distinctive part of a domain name.

    "www.iafd.com" -> "iafd". Short ones are refused: a two-letter
    token matches half the web by accident.
    """

    for label in host.lower().removeprefix("www.").split("."):
        if label in ("com", "net", "org", "co", "cc", "info"):
            continue

        if len(label) >= 4 and not label.isdigit():
            return label

    return ""


def _mentions(url: str, token: str) -> bool:
    """Whether a link's own path carries the site's brand token."""

    if not token:
        return False

    path = urlsplit(url).path.lower()

    return token in path


def _dedupe_variants(
    bucket: list[tuple[str, str]],
) -> list[tuple[str, str]]:
    """One entry per page, however many languages it is served in."""

    best: dict[tuple[str, str], tuple[str, str]] = {}

    for url, source in bucket:
        host, path, _depth = _variant_key(url)
        key = (host, path)

        current = best.get(key)

        if current is None:
            best[key] = (url, source)
            continue

        _c_host, _c_path, c_depth = _variant_key(current[0])

        # Prefer the canonical host, and never lose a source that
        # actually found the link.
        if _depth < c_depth or (
            _depth == c_depth and url < current[0]
        ):
            best[key] = (url, source)

    kept = set(entry[0] for entry in best.values())

    return [entry for entry in bucket if entry[0] in kept]


def _is_furniture(url: str) -> bool:
    """True for a link that is site navigation rather than a profile."""

    parsed = urlsplit(url)
    host = (parsed.hostname or "").lower()

    if host in _INTERSTITIAL_HOSTS:
        return True

    path = parsed.path.strip("/")

    # A per-performer mirror site puts the profile at the bare root, so
    # "no path means navigation" does not hold for these.
    if not path:
        return not any(
            host == parent or host.endswith(f".{parent}")
            for parent in _PROFILE_PARENTS
        )

    segments = [part for part in path.split("/") if part]
    lowered = [part.lower() for part in segments]
    first = lowered[0]

    if first in _FURNITURE:
        return True

    # x.com/i/jf/onboarding/web?mode=login is a login wall.
    if tuple(lowered[:2]) in _ONBOARDING_PREFIXES:
        return True

    # The tabs of a profile are not other profiles, and they sit
    # either side of the handle: x.com/her/photo and x.com/compose/post
    # are both the account already listed.
    for segment in lowered[:2]:
        if segment in _PROFILE_TABS:
            return True

    # The same words appear as subdomains, and a rendered platform page
    # is mostly links to its own help centre: support.x.com/articles,
    # help.x.com/resources, business.x.com/en/help. Checking the path
    # alone lets all of them through, and the tool then reports its own
    # nav as discovered accounts. Only labels before the host's own
    # name are considered, so www is never mistaken for furniture.
    labels = host.split(".")

    for label in labels[:-1]:
        if label == "www":
            continue

        if label in _FURNITURE or label in _FURNITURE_SUBDOMAINS:
            return True

    return False


@dataclass(slots=True)
class SourceResult:
    """What one page yielded."""

    url: str
    status: str
    platform: str
    found: tuple[tuple[str, str], ...] = ()
    """(platform, url) pairs read off the page."""
    error: str = ""
    note: str = ""


@dataclass(slots=True)
class SocialExpander:
    """Walks a set of pages and harvests the accounts they link to."""

    fetcher: Any
    policy: ExpandPolicy = field(default_factory=ExpandPolicy)

    async def _one(
        self,
        url: str,
        known: set[str],
        semaphore: asyncio.Semaphore,
    ) -> SourceResult:
        platform = classify(url)

        async with semaphore:
            try:
                page = await self.fetcher.fetch(url)
            except Exception as exc:  # noqa: BLE001
                # One unreadable page must not lose the rest.
                return SourceResult(
                    url=url,
                    status="failed",
                    platform=platform,
                    error=str(exc).splitlines()[0][:160],
                )

        found: list[tuple[str, str]] = []
        seen: set[str] = set()
        source_host = urlsplit(url).netloc.lower()

        for link in list(page.links or ())[: self.policy.per_page]:
            text = str(link).strip()

            if not text.startswith(("http://", "https://")):
                continue

            host = urlsplit(text).netloc.lower()

            # Same-host navigation is not a discovery.
            if not host or host == source_host:
                continue

            key = text.lower().removeprefix("https://").removeprefix("http://")
            key = key.removeprefix("www.").rstrip("/")

            if key in seen or key in known:
                continue

            # Site furniture is not an account.
            if _is_furniture(text):
                continue

            label = classify(text)

            if label == "other" and not _is_aggregator(text):
                continue

            seen.add(key)
            found.append((label, text))

        if found and _is_aggregator(url):
            token = _brand_token(source_host)

            if token:
                # A directory's own social accounts are the most
                # misleading thing this tool can report: real links,
                # confidently attributed to a performer, from a page
                # that merely looked like hers. They are recognisable
                # because the handle echoes the site's own name --
                # iafd.com links twitter.com/iafdcom. A performer's own
                # handle on that platform would not.
                found = [
                    (label, text)
                    for label, text in found
                    if not (
                        label != "other"
                        and _mentions(text, token)
                    )
                ]

        note = ""

        note = ""

        if not found and platform not in ("other", ""):
            note = (
                "page returned no links to other accounts; likely "
                "rendered by script or behind a login"
            )

        return SourceResult(
            url=url,
            status="read",
            platform=platform,
            found=tuple(found),
            note=note,
        )

    async def walk(
        self,
        urls: list[str],
        *,
        known_urls: list[str] | None = None,
    ) -> list[SourceResult]:
        """Read the sources, then the directories they point at."""

        sources: list[str] = []
        seen: set[str] = set()

        for url in urls:
            text = url.strip()

            if not text.startswith(("http://", "https://")):
                # A bare host is still worth trying.
                if "." in text and not text.startswith("http"):
                    text = f"https://{text}"

            if not text.startswith(("http://", "https://")):
                continue

            key = text.lower().removeprefix("https://").rstrip("/")

            if key in seen:
                continue

            seen.add(key)
            sources.append(text)

        if not sources:
            raise SocialExpandError("no usable http links to read")

        sources = sources[: self.policy.max_sources]

        known = {
            entry.strip()
            .lower()
            .removeprefix("https://")
            .removeprefix("http://")
            .removeprefix("www.")
            .rstrip("/")
            for entry in (known_urls or [])
        }

        semaphore = asyncio.Semaphore(self.policy.concurrency)

        results = list(
            await asyncio.gather(
                *(
                    self._one(url, known, semaphore)
                    for url in sources
                )
            )
        )

        if self.policy.hops < 2:
            return results

        # Only directories, and only a few: a profile page that links
        # another profile is not a directory, and following those is
        # how a read becomes a crawl.
        directories: list[str] = []
        seen_dirs: set[str] = set()

        for result in results:
            for label, url in result.found:
                if label != "other" or not _is_aggregator(url):
                    continue

                key = url.lower().removeprefix("https://").rstrip("/")

                if key in seen_dirs:
                    continue

                seen_dirs.add(key)
                directories.append(url)

        if not directories:
            return results

        directories = directories[: self.policy.max_directories]

        results.extend(
            await asyncio.gather(
                *(
                    self._one(url, known, semaphore)
                    for url in directories
                )
            )
        )

        return results


def format_expand(
    results: list[SourceResult],
    name: str,
    platforms: list[str] | None = None,
) -> str:
    """Group what was found by platform, and say what was not read."""

    wanted = {
        item.strip().lower()
        for item in (platforms or [])
        if item.strip()
    }

    by_platform: dict[str, list[tuple[str, str]]] = {}

    for result in results:
        for label, url in result.found:
            if wanted and label.lower() not in wanted:
                continue

            # Membership against a list of (url, source) pairs is
            # always False: the element is a tuple, the needle a
            # string. So the same link found on two pages was reported
            # twice and the count above overstated it.
            bucket = by_platform.setdefault(label, [])

            if url not in {entry[0] for entry in bucket}:
                bucket.append((url, result.url))

    # One account in thirteen languages is one account. Without this,
    # a single pornhub page reports de/fr/it/pt/es/... as a dozen
    # discoveries and buries the one link that was actually new.
    for label, bucket in by_platform.items():
        by_platform[label] = _dedupe_variants(bucket)

    by_platform = {
        label: bucket
        for label, bucket in by_platform.items()
        if bucket
    }

    dropped = len(
        [1 for r in results for _l, _u in r.found]
    ) - sum(len(v) for v in by_platform.values())

    read = [r for r in results if r.status == "read"]
    failed = [r for r in results if r.status == "failed"]
    total = sum(len(v) for v in by_platform.values())

    lines: list[str] = []

    who = f" for {name!r}" if name else ""

    lines.append(
        f"Read {len(read)} of {len(results)} source pages{who}, "
        f"found {total} account links."
    )

    if total == 0 and not failed:
        lines.append("")
        lines.append(
            "No account links on any page. That usually means every "
            "source renders its links with script or sits behind a "
            "login. The pages that list accounts rather than showing "
            "one profile are the productive ones here."
        )

    if dropped > 0:
        lines.append("")
        lines.append(
            f"{dropped} further link(s) were the same account in "
            "another language or on a page of the profile already "
            "listed, and are not repeated here."
        )

    for label in sorted(by_platform):
        lines.append("")
        lines.append(f"{label}:")

        for url, source in by_platform[label]:
            lines.append(f"  {url}")
            lines.append(f"    found on {source}")

    if failed:
        lines.append("")
        lines.append(
            "NOT READ, so nothing is known from them: "
            f"{len(failed)} pages"
        )

        for result in failed:
            lines.append(f"  {result.url}: {result.error}")

    return "\n".join(lines)


def create_web_social_expand_tool(fetcher: Any) -> Tool:
    """Read accounts off pages that are already on file.

    No search engine is involved, so this costs nothing against the
    per-address quota that the sweep competes for, and it works for
    performers whose accounts are listed on an aggregator page rather
    than indexed anywhere searchable.
    """

    async def handler(
        arguments: ExpandInput,
        context: ToolContext,
    ) -> str:
        del context

        expander = SocialExpander(
            fetcher,
            ExpandPolicy(
                per_page=max(1, arguments.per_page),
                hops=max(1, arguments.hops),
            ),
        )

        results = await expander.walk(
            arguments.urls,
            known_urls=arguments.urls,
        )

        report = format_expand(
            results,
            arguments.name,
            arguments.platforms,
        )

        # A page that was refused and then read anyway is not a problem, and
        # saying so is worse than silence: it reads as though something
        # was missed when nothing was.
        failures = (
            fetcher.take_failures(arguments.urls)
            if hasattr(fetcher, "take_failures")
            else list(getattr(fetcher, "render_failures", {}).items())
        )

        if failures:
            report += (
                "\n\nNOT READ at all, so nothing is known from them: "
                f"{len(failures)} page(s).\n"
            )

            for url, why in failures[:8]:
                report += f"  {url}: {why}\n"

            report += (
                "\nThese pages answered neither the plain request nor "
                "the browser."
            )

        return report

    return Tool(
        name="web_social_expand",
        description=(
            "Open pages that are already on file for someone and read "
            "the other accounts linked from them. Use it before "
            "searching: a link-in-bio page lists every profile in one "
            "place, and an aggregator page built for this purpose "
            "does the same thing. That is one page fetch per source "
            "instead of one search per platform, and it does not "
            "touch a search engine at all, so it does not compete "
            "with anything else for this address's quota.\n"
            "\n"
            "Pass the urls verbatim from the record, with name for the "
            "heading. Optional platforms narrows the report; leave it "
            "empty to keep every recognised platform.\n"
            "\n"
            "A hit is a link on a page about her, not proof the "
            "account is hers. Check a returned handle against the "
            "record before reporting it, and say what you matched on. "
            "A page that renders its links with script, or sits behind "
            "a login, yields nothing and is reported as having no "
            "links rather than as having none."
        ),
        input_type=ExpandInput,
        handler=handler,
        policy=ToolPolicy(
            permissions=frozenset({
                "web.fetch",
            }),
            timeout=180.0,
            max_output_size=60_000,
        ),
    )


__all__: list[Any] = [
    "ExpandInput",
    "ExpandPolicy",
    "SocialExpandError",
    "SocialExpander",
    "SourceResult",
    "classify",
    "create_web_social_expand_tool",
    "format_expand",
]
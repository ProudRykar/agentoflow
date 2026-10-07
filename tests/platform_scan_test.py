"""A platform list is mechanical work, and a small model will fumble it.

Asked to sweep forty sites by hand, the model loses count, skips
platforms, and then narrates a search it never ran -- inventing a
Linktree that no tool mentioned and declaring the task finished having
answered a fraction of it.

So the sweep happens here, and every platform ends up with a verdict.
The verdict is the point: "found", "no results", "incomplete" and
"failed" have to stay separate, because the first is evidence and the
last three are not, and reporting an unanswered search as an absent
account is a confident falsehood.
"""

from __future__ import annotations

import asyncio

import pytest

from agent_workflow.core.entities.models.platform_scan import (
    BUILTIN_PLATFORMS,
    PlatformResult,
    PlatformScanner,
    ScanPolicy,
    create_web_platform_scan_tool,
    format_scan,
    resolve,
)


class FakeSearch:
    """Answers per host, and can pretend to break."""

    def __init__(
        self,
        answers: dict[str, list[tuple[str, str]]] | None = None,
        *,
        raise_for: set[str] | None = None,
        degraded_for: set[str] | None = None,
    ) -> None:
        self.answers = answers or {}
        self.raise_for = raise_for or set()
        self.degraded_for = degraded_for or set()
        self.queries: list[str] = []
        self.policy = type("P", (), {"timeout": 5.0})()

    async def search_detailed(self, query: str, limit: int = 2):
        self.queries.append(query)

        for host in self.raise_for:
            if host in query:
                raise RuntimeError(f"{host} is down")

        degraded = ("brave", "duckduckgo") if self.degraded_for else ()

        rows = []

        for host, pairs in self.answers.items():
            if host not in query:
                continue

            rows.extend(
                type("R", (), {"url": url, "title": title, "snippet": ""})()
                for url, title in pairs[:limit]
            )

        return rows, degraded

    async def search(self, query: str, limit: int = 2):
        results, _degraded = await self.search_detailed(query, limit)

        return results


async def test_each_platform_gets_a_verdict() -> None:
    search = FakeSearch({
        "reddit.com": [("https://reddit.com/u/x", "u/x")],
        "youtube.com": [("https://youtube.com/@x", "@x")],
    })

    scanner = PlatformScanner(search, ScanPolicy(concurrency=2))

    results, checked = await scanner.scan("Sweetie Fox", sites=["Reddit", "YouTube"])

    assert checked == 2
    assert {r.label: r.status for r in results} == {
        "Reddit": "found",
        "YouTube": "found",
    }


async def test_a_platform_with_no_hits_says_so() -> None:
    scanner = PlatformScanner(FakeSearch(), ScanPolicy())

    results, _checked = await scanner.scan("Nobody Here", sites=["Reddit"])

    assert results[0].status == "none"
    assert results[0].urls == ()


async def test_an_empty_answer_from_broken_engines_is_not_absence() -> None:
    """The dangerous case.

    An instance whose engines are all timing out returns an empty
    result set, indistinguishable from a real 'nothing matched'. Read
    as absence it becomes "this platform has no account", stated as
    evidence, and it happens most often right after a restart.
    """

    search = FakeSearch(degraded_for={"reddit.com"})

    scanner = PlatformScanner(search, ScanPolicy())

    results, _checked = await scanner.scan("Sweetie Fox", sites=["Reddit"])

    assert results[0].status == "incomplete"
    assert results[0].urls == ()
    assert "brave" in results[0].degraded


async def test_a_broken_engine_does_not_hide_a_hit() -> None:
    """Results beat degraded engines; a hit is a hit."""

    search = FakeSearch(
        {"reddit.com": [("https://reddit.com/u/x", "u/x")]},
        degraded_for={"reddit.com"},
    )

    scanner = PlatformScanner(search, ScanPolicy())

    results, _checked = await scanner.scan("Sweetie Fox", sites=["Reddit"])

    assert results[0].status == "found"


async def test_one_failing_platform_does_not_lose_the_others() -> None:
    search = FakeSearch(
        {"youtube.com": [("https://youtube.com/@x", "@x")]},
        raise_for={"reddit.com"},
    )

    scanner = PlatformScanner(search, ScanPolicy())

    results, _checked = await scanner.scan(
        "Sweetie Fox",
        sites=["Reddit", "YouTube"],
    )

    by_label = {r.label: r.status for r in results}

    assert by_label["Reddit"] == "error"
    assert by_label["YouTube"] == "found"


async def test_aliases_are_tried_when_the_name_matches_nothing() -> None:
    class AliasOnly(FakeSearch):
        """Answers only for the alias, so the name genuinely misses."""

        async def search_detailed(self, query: str, limit: int = 2):
            if "RightAlias" not in query:
                return [], ()

            return await super().search_detailed(query, limit)

    search = AliasOnly({"reddit.com": [("https://reddit.com/u/alias", "alias")]})

    scanner = PlatformScanner(search, ScanPolicy())

    results, _checked = await scanner.scan(
        "Wrong Name",
        sites=["Reddit"],
        aliases=["RightAlias"],
    )

    assert results[0].status == "found"
    # Reported so the caller knows the match came from an alias.
    assert results[0].matched_with == "RightAlias"


async def test_aliases_are_capped_so_a_long_list_does_not_stall() -> None:
    search = FakeSearch()

    scanner = PlatformScanner(search, ScanPolicy())

    await scanner.scan(
        "Name",
        sites=["Reddit"],
        aliases=[f"alias{index}" for index in range(50)],
    )

    # The name plus at most two aliases.
    assert len(search.queries) <= 3


async def test_the_name_is_quoted_in_every_query() -> None:
    """An unquoted multi-word name is a pile of keywords.

    That is how a sweep for one person returns a forum thread about
    something else entirely.
    """

    search = FakeSearch()

    scanner = PlatformScanner(search, ScanPolicy())

    await scanner.scan("Sweetie Fox", sites=["Reddit"])

    assert search.queries == ['"Sweetie Fox" site:reddit.com']


async def test_an_empty_name_is_refused_before_any_query() -> None:
    search = FakeSearch()

    with pytest.raises(Exception, match="name to search for"):
        await PlatformScanner(search, ScanPolicy()).scan("   ")

    assert search.queries == []


async def test_known_platforms_expand_to_every_relevant_host() -> None:
    """X lives on two hosts; checking one would miss half of it."""

    targets = resolve(["X", "Bluesky"])

    hosts = {target.site for target in targets}

    assert "twitter.com" in hosts
    assert "x.com" in hosts
    assert "bsky.app" in hosts


async def test_an_unknown_platform_is_searched_rather_than_dropped() -> None:
    """Silently ignoring one line of the user's list is a silent failure."""

    targets = resolve(["SomeNewPlatform"])

    assert targets[0].site == "SomeNewPlatform"


async def test_a_bare_host_in_the_list_works() -> None:
    targets = resolve(["https://example.com/"])

    assert targets[0].site == "example.com"


def test_no_duplicate_platform_requests() -> None:
    targets = resolve(["Reddit", "reddit", " REDDIT "])

    assert len(targets) == 1


def test_the_builtin_list_covers_the_usual_suspects() -> None:
    labels = {target.label for target in BUILTIN_PLATFORMS}

    for expected in (
        "X", "Reddit", "Bluesky", "Telegram", "Mastodon", "Tumblr",
        "FetLife", "Snapchat", "Instagram", "Facebook", "TikTok",
        "YouTube", "Threads", "Pinterest", "VK", "OK", "Twitch",
        "Kick", "LinkedIn", "Patreon", "Fansly", "OnlyFans",
        "ManyVids", "LoyalFans", "Fanvue", "Ko-fi", "RedGIFs",
        "Chaturbate", "Stripchat", "Cam4",
    ):
        assert expected in labels, expected


def test_walled_platforms_carry_a_note() -> None:
    """A blank for a login-walled site is expected and should say so."""

    target = next(t for t in BUILTIN_PLATFORMS if t.label == "OnlyFans")

    assert target.note


def test_policy_rejects_nonsense() -> None:
    for kwargs in (
        {"concurrency": 0},
        {"per_site": 0},
        {"max_sites": 0},
        {"timeout": 0},
    ):
        with pytest.raises(ValueError):
            ScanPolicy(**kwargs)


def test_rendering_keeps_the_verdicts_apart() -> None:
    rendered = format_scan(
        [
            PlatformResult("Reddit", "reddit.com", "", ("https://reddit.com/u/x",)),
            PlatformResult("FetLife", "fetlife.com", "login-walled"),
            PlatformResult(
                "Mastodon",
                "mastodon.social",
                "",
                degraded=("brave",),
            ),
            PlatformResult("X", "x.com", "", error="instance down"),
        ],
        checked=4,
        name="Sweetie Fox",
    )

    assert "FOUND:" in rendered
    assert "NO RESULTS:" in rendered
    assert "INCOMPLETE" in rendered
    assert "SEARCH FAILED:" in rendered
    assert "no account on this platform" not in rendered
    assert "Treat them as unknown, not as absent" in rendered


def test_rendering_an_all_clean_sweep() -> None:
    rendered = format_scan(
        [PlatformResult("Reddit", "reddit.com", "", ("https://r/u",))],
        checked=1,
        name="X",
    )

    assert "found 1, no results 0" in rendered
    assert "SEARCH FAILED" not in rendered


async def test_the_tool_asks_for_search_permission() -> None:
    tool = create_web_platform_scan_tool(FakeSearch())

    assert tool.name == "web_platform_scan"
    assert tool.policy.permissions == frozenset({"web.search"})
    assert tool.policy.timeout >= 120


async def test_the_tool_renders_its_scan() -> None:
    search = FakeSearch({"reddit.com": [("https://reddit.com/u/x", "u/x")]})

    class Context:
        pass

    tool = create_web_platform_scan_tool(search)

    output = await tool.handler(
        tool.input_type(name="Sweetie Fox", sites=["Reddit"]),
        Context(),  # type: ignore[arg-type]
    )

    assert "reddit.com/u/x" in output
    assert "found 1" in output


async def test_concurrency_is_capped_so_the_instance_is_not_flooded() -> None:
    """The instance runs without a limiter, on purpose.

    That makes politeness this tool's responsibility rather than the
    instance's.
    """

    live = 0
    peak = 0

    class Counting(FakeSearch):
        async def search_detailed(self, query: str, limit: int = 2):
            nonlocal live, peak

            live += 1
            peak = max(peak, live)

            await asyncio.sleep(0.01)

            live -= 1

            return await super().search_detailed(query, limit)

    scanner = PlatformScanner(Counting(), ScanPolicy(concurrency=3))

    await scanner.scan("Name", sites=[f"host{index}.com" for index in range(12)])

    assert peak <= 3

class _UrlSearch(FakeSearch):
    """Answers per host with full URLs."""

    def __init__(self, answers):
        super().__init__()
        self.answers = answers


async def test_a_full_profile_url_is_reduced_to_its_host() -> None:
    """Guessing the path is the one thing this tool exists to avoid.

    A caller handing over reddit.com/r/someone must get a whole-site
    search, not a filter for a profile they invented. Keeping the path
    silently produces a query that matches nothing and reads as a
    clean miss.
    """

    search = _UrlSearch(
        {
            "reddit.com": [("https://reddit.com/u/real", "u/real")],
        }
    )

    scanner = PlatformScanner(search, ScanPolicy())

    results, _checked = await scanner.scan(
        "Sweetie Fox",
        sites=["https://www.reddit.com/r/sweetiefox"],
    )

    assert search.queries == ['"Sweetie Fox" site:reddit.com']
    assert results[0].status == "found"


async def test_a_path_is_dropped_from_every_shape_of_entry() -> None:
    for entry, expected in (
        ("https://bsky.app/a/sweetiefox", "bsky.app"),
        ("t.me/sweetiefox", "t.me"),
        ("mastodon.social/@sweetiefox", "mastodon.social"),
        ("reddit.com/u/x", "reddit.com"),
        ("www.instagram.com/swfx_real", "instagram.com"),
    ):
        assert resolve([entry])[0].site == expected, entry


async def test_known_urls_are_separated_from_new_ones() -> None:
    """The usual ask is what is not on file yet.

    Doing that comparison from memory drops links silently, so it is
    done here instead.
    """

    search = _UrlSearch(
        {
            "instagram.com": [("https://instagram.com/swfx_real/", "known")],
            "reddit.com": [("https://reddit.com/u/new", "new")],
        }
    )

    scanner = PlatformScanner(search, ScanPolicy())

    results, _checked = await scanner.scan(
        "Sweetie Fox",
        sites=["Instagram", "Reddit"],
        known_urls=["https://instagram.com/swfx_real"],
    )

    by_label = {r.label: r for r in results}

    assert by_label["Instagram"].new_urls == ()
    assert by_label["Instagram"].known == ("https://instagram.com/swfx_real/",)
    assert by_label["Reddit"].new_urls == ("https://reddit.com/u/new",)


async def test_known_urls_ignore_scheme_www_and_trailing_slash() -> None:
    search = _UrlSearch(
        {"instagram.com": [("https://instagram.com/swfx_real", "x")]},
    )

    scanner = PlatformScanner(search, ScanPolicy())

    results, _checked = await scanner.scan(
        "Sweetie Fox",
        sites=["Instagram"],
        known_urls=["https://www.instagram.com/swfx_real/"],
    )

    # The same link, so it is not a discovery.
    assert results[0].new_urls == ()


async def test_the_new_section_leads_when_filtering() -> None:
    search = _UrlSearch(
        {
            "instagram.com": [("https://instagram.com/known", "known")],
            "reddit.com": [("https://reddit.com/u/new", "new")],
        }
    )

    class Context:
        pass

    from agent_workflow.core.entities.models.searxng_search import SearxngPolicy
    from agent_workflow.core.entities.models.searxng_search import SearxngSearch

    tool = create_web_platform_scan_tool(SearxngSearch(SearxngPolicy()))
    tool = create_web_platform_scan_tool(search)

    output = await tool.handler(
        tool.input_type(
            name="Sweetie Fox",
            sites=["Instagram", "Reddit"],
            known_urls=["https://instagram.com/known"],
        ),
        Context(),  # type: ignore[arg-type]
    )

    assert "NEW, not in the list supplied:" in output
    assert "https://reddit.com/u/new" in output
    assert "(already known)" in output
    assert "1 new links across 1 platforms" in output


async def test_a_run_where_nothing_was_asked_says_so_first() -> None:
    """A wall of non-answers must not read as a sweep that found nothing."""

    search = FakeSearch(degraded_for={"reddit.com"})

    scanner = PlatformScanner(search, ScanPolicy())

    results, checked = await scanner.scan(
        "Sweetie Fox",
        sites=["Reddit", "Reddit"],
    )

    rendered = format_scan(results, checked, "Sweetie Fox")

    assert "NOTHING WAS ACTUALLY ASKED" in rendered
    # Said before the list, not after it.
    assert rendered.index("NOTHING WAS ACTUALLY ASKED") < rendered.index(
        "INCOMPLETE"
    )


def test_the_description_forbids_guessed_urls() -> None:
    from agent_workflow.core.entities.models.searxng_search import SearxngPolicy
    from agent_workflow.core.entities.models.searxng_search import SearxngSearch

    tool = create_web_platform_scan_tool(SearxngSearch(SearxngPolicy()))

    assert "never URLs" in tool.description
    assert "six to ten" in tool.description
    assert "known_urls" in tool.description


def test_no_links_found_does_not_claim_a_comparison_happened() -> None:
    """A misleading dedup line is worse than no line.

    "Every link found was already in the list" reads as a finished
    comparison, and is stated when in fact nothing was found.
    """

    rendered = format_scan(
        [
            PlatformResult(
                "Reddit",
                "reddit.com",
                "",
                degraded=("brave",),
            )
        ],
        checked=1,
        name="Sweetie Fox",
        filtered=True,
    )

    assert "cannot say" in rendered
    assert "already in the list" not in rendered

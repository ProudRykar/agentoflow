"""Reading accounts off pages you already have beats searching for them.

A link-in-bio page lists every profile its owner collected, and an
aggregator page built for this purpose does the same thing. That is one
page fetch instead of one search per platform, and it touches no search
engine, so it never competes for the per-address quota that the sweep
burns through.

The tests are mostly about what must *not* be reported, because that is
where this is easy to get wrong: a profile page links its own login and
legal sections, a page can link a CDN, and "no links found" and "could
not read the page" are different answers.
"""

from __future__ import annotations

import pytest

from agent_workflow.core.entities.models.builtin.tools import (
    create_builtin_tools,
)
from agent_workflow.core.entities.models.builtin.web_crawl import (
    WebCrawler,
)
from agent_workflow.core.entities.models.social_expand import (
    ExpandInput,
    ExpandPolicy,
    SocialExpandError,
    SocialExpander,
    SourceResult,
    classify,
    create_web_social_expand_tool,
    format_expand,
)


class FakePage:
    def __init__(self, links: list[str], status: int = 200) -> None:
        self.links = tuple(links)
        self.status_code = status
        self.content = ""
        self.content_bytes = 1024
        self.content_type = "text/html"
        self.url = ""


class FakeFetcher:
    """Serves a page per host, and can fail or hang on chosen ones."""

    def __init__(
        self,
        pages: dict[str, list[str]] | None = None,
        *,
        fail: set[str] | None = None,
    ) -> None:
        self.pages = pages or {}
        self.fail = fail or set()
        self.seen: list[str] = []

    async def fetch(self, url: str, **kwargs) -> FakePage:
        self.seen.append(url)

        for host in self.fail:
            if host in url:
                raise RuntimeError(f"HTTP 403 for {url}")

        for host, links in self.pages.items():
            if host in url:
                return FakePage(links)

        return FakePage([])


async def test_it_reads_accounts_off_a_source_page() -> None:
    fetcher = FakeFetcher(
        {
            "linktr.ee": [
                "https://instagram.com/her",
                "https://x.com/her",
                "https://twitter.com/her_twin",
            ]
        }
    )

    expander = SocialExpander(fetcher, ExpandPolicy(hops=1))

    results = await expander.walk(
        ["https://linktr.ee/her"],
        known_urls=["https://linktr.ee/her"],
    )

    assert results[0].status == "read"
    # Both x.com and twitter.com are labelled X, so a dict would
    # collapse them; compare the pairs.
    assert set(results[0].found) == {
        ("Instagram", "https://instagram.com/her"),
        ("X", "https://x.com/her"),
        ("X", "https://twitter.com/her_twin"),
    }


async def test_site_furniture_is_not_an_account() -> None:
    """A profile page links its own login and legal sections.

    Reporting those is not a small inaccuracy: it fills the answer with
    rows that are obviously not a person, and it buries the real finds.
    """

    fetcher = FakeFetcher(
        {
            "instagram.com": [
                "https://www.instagram.com/accounts/password/reset/?next=x",
                "https://www.instagram.com/legal/privacy/",
                "https://www.instagram.com/explore/locations/",
                "https://www.instagram.com/web/lite/",
                "https://www.instagram.com/intent/tweet?text=hi",
                "https://instagram.com/",
                "https://t.me/her",
            ]
        }
    )

    expander = SocialExpander(fetcher, ExpandPolicy(hops=1))

    results = await expander.walk(
        ["https://instagram.com/her"],
        known_urls=["https://instagram.com/her"],
    )

    assert [url for _, url in results[0].found] == ["https://t.me/her"]


async def test_a_cdn_is_not_a_directory_to_follow() -> None:
    """A subdomain of the host is not the host.

    Matching aggregators on any subdomain pulled in a CDN, whose links
    are images.
    """

    fetcher = FakeFetcher(
        {
            "socialmediapornstars.com": [
                "https://cdn.socialmediapornstars.com/tweet-abc.jpg",
                "https://pics-x.com/pornstar/1/her",
            ]
        }
    )

    expander = SocialExpander(fetcher, ExpandPolicy(hops=2))

    results = await expander.walk(
        ["https://socialmediapornstars.com/pornstar-her.html"],
        known_urls=[],
    )

    # The jpg is not reported as a find...
    assert all(
        not url.endswith(".jpg")
        for result in results
        for _, url in result.found
    )
    # ...and the one real directory is still followed.
    assert "https://pics-x.com/pornstar/1/her" in fetcher.seen


async def test_the_second_hop_follows_directories_only() -> None:
    """The record points at a profile; the profile points at a directory."""

    fetcher = FakeFetcher(
        {
            "instagram.com": ["https://pics-x.com/pornstar/1/her"],
            "pics-x.com": ["https://x.com/her", "https://ko-fi.com/her"],
        }
    )

    expander = SocialExpander(fetcher, ExpandPolicy(hops=2))

    results = await expander.walk(
        ["https://instagram.com/her"],
        known_urls=["https://instagram.com/her"],
    )

    assert "https://pics-x.com/pornstar/1/her" in fetcher.seen

    found = {
        url
        for result in results
        for _, url in result.found
    }

    assert "https://x.com/her" in found
    assert "https://ko-fi.com/her" in found


async def test_one_hop_skips_the_directory() -> None:
    fetcher = FakeFetcher(
        {"instagram.com": ["https://pics-x.com/pornstar/1/her"]}
    )

    expander = SocialExpander(fetcher, ExpandPolicy(hops=1))

    await expander.walk(["https://instagram.com/her"], known_urls=[])

    assert fetcher.seen == ["https://instagram.com/her"]


async def test_the_directory_hop_is_capped() -> None:
    """Following every directory is how a read becomes a crawl."""

    links = [f"https://pics-x.com/pornstar/{i}/her" for i in range(20)]

    fetcher = FakeFetcher(
        {"instagram.com": links, "pics-x.com": ["https://x.com/her"]}
    )

    expander = SocialExpander(
        fetcher,
        ExpandPolicy(hops=2, max_directories=3),
    )

    await expander.walk(["https://instagram.com/her"], known_urls=[])

    followed = [url for url in fetcher.seen if "pics-x" in url]

    assert len(followed) == 3


async def test_known_links_are_not_reported_back() -> None:
    fetcher = FakeFetcher(
        {"linktr.ee": ["https://x.com/her", "https://x.com/her_twin"]}
    )

    expander = SocialExpander(fetcher, ExpandPolicy(hops=1))

    results = await expander.walk(
        ["https://linktr.ee/her"],
        known_urls=["https://linktr.ee/her", "https://x.com/her"],
    )

    assert [url for _, url in results[0].found] == ["https://x.com/her_twin"]


async def test_a_page_that_cannot_be_read_is_reported_separately() -> None:
    """"No links" and "not read" are different answers.

    The first is a conclusion, the second is an absence of one, and
    reporting them the same way is how a page that blocks crawlers
    becomes "she has no account there".
    """

    fetcher = FakeFetcher({"x.com": ["https://t.me/her"]}, fail={"instagram.com"})

    expander = SocialExpander(fetcher, ExpandPolicy(hops=1))

    results = await expander.walk(
        ["https://instagram.com/her", "https://x.com/her"],
        known_urls=[],
    )

    by_url = {result.url: result for result in results}

    assert by_url["https://instagram.com/her"].status == "failed"
    assert by_url["https://x.com/her"].status == "read"

    rendered = format_expand(results, "Her")

    assert "NOT READ" in rendered
    assert "403" in rendered


async def test_a_script_rendered_page_says_so() -> None:
    fetcher = FakeFetcher({"instagram.com": []})

    expander = SocialExpander(fetcher, ExpandPolicy(hops=1))

    results = await expander.walk(["https://instagram.com/her"], known_urls=[])

    assert "rendered by script" in results[0].note


async def test_an_empty_walk_explains_the_likely_reason() -> None:
    fetcher = FakeFetcher({"instagram.com": [], "x.com": []})

    expander = SocialExpander(fetcher, ExpandPolicy(hops=1))

    rendered = format_expand(
        await expander.walk(
            ["https://instagram.com/her", "https://x.com/her"],
            known_urls=[],
        ),
        "Her",
    )

    assert "No account links on any page" in rendered
    assert "behind a" in rendered


def test_platforms_are_labelled_from_the_sweep_table() -> None:
    assert classify("https://instagram.com/her") == "Instagram"
    assert classify("https://t.me/her") == "Telegram"
    assert classify("https://ko-fi.com/her") == "Ko-fi"
    assert classify("https://onlyfans.com/her") == "OnlyFans"
    assert classify("https://example.com/her") == "other"


async def test_duplicates_across_pages_are_reported_once() -> None:
    fetcher = FakeFetcher(
        {
            "linktr.ee": ["https://x.com/her"],
            "allmylinks.com": ["https://x.com/her", "https://x.com/her"],
        }
    )

    expander = SocialExpander(fetcher, ExpandPolicy(hops=1))

    results = await expander.walk(
        ["https://linktr.ee/her", "https://allmylinks.com/her"],
        known_urls=[],
    )

    # Deduping is the report's job; two pages may each see the link.
    rendered = format_expand(results, "Her")

    assert rendered.count("https://x.com/her") == 1


async def test_a_bare_host_is_still_usable() -> None:
    expander = SocialExpander(FakeFetcher(), ExpandPolicy(hops=1))

    results = await expander.walk(["onlyfans.com/her"], known_urls=[])

    assert results[0].url.startswith("https://")


async def test_a_walk_with_nothing_to_read_says_so() -> None:
    expander = SocialExpander(FakeFetcher(), ExpandPolicy())

    with pytest.raises(SocialExpandError, match="no usable http links"):
        await expander.walk(["", "   "], known_urls=[])


def test_policy_rejects_nonsense() -> None:
    for kwargs in (
        {"concurrency": 0},
        {"max_sources": 0},
        {"per_page": 0},
        {"timeout": 0},
        {"hops": 0},
    ):
        with pytest.raises(ValueError):
            ExpandPolicy(**kwargs)


def test_builtin_wiring_hands_expand_a_thing_that_can_fetch() -> None:
    """The expander must be handed something with fetch().

    Both WebCrawler and a fetcher read as "the web thing" and both were
    wired in here at once, but only the latter has fetch(); WebCrawler
    exposes crawl() and queues a whole site. Handing over the crawler
    raised AttributeError on every URL, and the per-page handler
    reported each one as an unreadable page -- a working tool that
    silently found nothing, on every single performer.

    The fakes in this file all implement fetch(), so no test using them
    could ever have caught this. Read what production actually passes.
    """

    tool = {
        t.name: t for t in create_builtin_tools()
    }["web_social_expand"]

    # The tool factory closes over the fetcher it was given.
    captured = [
        cell.cell_contents
        for cell in tool.handler.__closure__ or ()
        if hasattr(cell.cell_contents, "fetch")
    ]

    assert len(captured) == 1, "expected exactly one fetcher in the closure"

    fetcher = captured[0]

    assert callable(fetcher.fetch), "expander calls fetcher.fetch(url)"

    # The exact bug: the crawler is closured in instead of a fetcher.
    assert not isinstance(fetcher, WebCrawler), (
        "web_social_expand was handed a WebCrawler, which has crawl(), "
        "not fetch()"
    )


def test_the_report_leads_with_a_count() -> None:
    rendered = format_expand(
        [
            SourceResult(
                "https://linktr.ee/her",
                "read",
                "other",
                found=(("X", "https://x.com/her"),),
            )
        ],
        "Her",
    )

    assert "found 1 account links" in rendered
    assert "found on" in rendered


def test_the_report_can_be_narrowed_to_platforms() -> None:
    rendered = format_expand(
        [
            SourceResult(
                "https://linktr.ee/her",
                "read",
                "other",
                found=(
                    ("X", "https://x.com/her"),
                    ("Telegram", "https://t.me/her"),
                ),
            )
        ],
        "Her",
        ["Telegram"],
    )

    assert "Telegram:" in rendered
    assert "X:" not in rendered


async def test_the_tool_is_registered_and_asks_for_fetch_permission() -> None:
    names = {tool.name for tool in create_builtin_tools()}

    assert "web_social_expand" in names

    tool = create_web_social_expand_tool(FakeFetcher())

    assert tool.policy.permissions == frozenset({"web.fetch"})


async def test_the_tool_passes_its_arguments_through() -> None:
    fetcher = FakeFetcher({"linktr.ee": ["https://x.com/her"]})

    class Context:
        pass

    tool = create_web_social_expand_tool(fetcher)

    output = await tool.handler(
        tool.input_type(urls=["https://linktr.ee/her"], name="Her"),
        Context(),  # type: ignore[arg-type]
    )

    assert "x.com/her" in output
    assert "Her" in output


def test_input_defaults_read_the_source_list() -> None:
    arguments = ExpandInput(urls=["https://a.dev"], name="Her")

    assert arguments.hops == 2
    assert arguments.platforms is None

@pytest.mark.parametrize(
    "url",
    [
        # These are what a *rendered* x.com page is mostly made of.
        "https://support.x.com/articles/18311",
        "https://help.x.com/resources/accessibility",
        "https://business.x.com/en/help/twitter-ads-work.html",
        "https://careers.instagram.com/jobs/1",
        "https://legal.tiktok.com/policy",
    ],
)
def test_help_centre_links_are_not_accounts(url: str) -> None:
    """A rendered platform page is mostly links to its own help centre.

    Without this the tool reported four X support articles as
    discovered accounts, which looks like a result and is not one.
    """

    from agent_workflow.core.entities.models.social_expand import (
        _is_furniture,
    )

    assert _is_furniture(url)


@pytest.mark.parametrize(
    "url",
    [
        "https://x.com/Angie_Elif",
        "https://www.instagram.com/sunnyzayka",
        # Short subdomain labels are still profiles, not navigation.
        "https://i.instagram.com/sunnyzayka",
        "https://m.facebook.com/her",
    ],
)
def test_real_profiles_survive_the_subdomain_filter(url: str) -> None:
    from agent_workflow.core.entities.models.social_expand import (
        _is_furniture,
    )

    assert not _is_furniture(url)


def test_a_profile_at_a_bare_root_is_not_navigation() -> None:
    """Per-performer mirror sites put the profile at the domain root."""

    from agent_workflow.core.entities.models.social_expand import (
        _is_furniture,
        classify,
    )

    url = "https://angieelif.bezoxo.com/"

    assert not _is_furniture(url)
    assert classify(url) == "bezoxo.com"


def test_a_bare_root_is_still_furniture_elsewhere() -> None:
    """The exemption is for mirror sites, not a general loosening."""

    from agent_workflow.core.entities.models.social_expand import (
        _is_furniture,
    )

    assert _is_furniture("https://selfymodels.com/")


@pytest.mark.parametrize(
    ("url", "label"),
    [
        ("https://selfymodels.com/her", "selfymodels.com"),
        ("https://stripschat.com/her", "stripschat.com"),
    ],
)
def test_a_profile_host_is_not_folded_into_other(
    url: str,
    label: str,
) -> None:
    """A link-in-bio page is mostly made of hosts the table lacks.

    Without these, the page renders, gets read, and contributes nothing,
    which is the same as not having rendered it at all.
    """

    from agent_workflow.core.entities.models.social_expand import classify

    assert classify(url) == label


@pytest.mark.parametrize(
    ("url", "label"),
    [
        # Already in the sweep's table, so the platform name wins over
        # the bare host name.
        ("https://onlyfans.com/her", "OnlyFans"),
        ("https://fansly.com/her", "Fansly"),
    ],
)
def test_a_known_platform_label_beats_the_host_name(
    url: str,
    label: str,
) -> None:
    from agent_workflow.core.entities.models.social_expand import classify

    assert classify(url) == label


async def test_a_directorys_own_accounts_are_not_the_performers() -> None:
    """The most misleading failure this tool can have.

    Every page of iafd.com carries iafd's own Facebook and Twitter in
    its footer. Reading them off a performer's page and reporting them
    as her accounts is worse than finding nothing, because the links
    are real and the output looks like a result.
    """

    fetcher = FakeFetcher(
        {
            "iafd.com": [
                "https://twitter.com/iafdcom",
                "https://instagram.com/iafdcom",
                "https://instagram.com/sunnyzayka",
                "https://pics-x.com/pornstar/1/her",
            ]
        }
    )

    expander = SocialExpander(fetcher, ExpandPolicy(hops=1))

    results = await expander.walk(
        ["https://www.iafd.com/person.rme/id=1/gender=f"],
        known_urls=[],
    )

    found = {url for _label, url in results[0].found}

    assert "https://instagram.com/sunnyzayka" in found, "hers, not the site's"
    assert "https://twitter.com/iafdcom" not in found
    assert "https://instagram.com/iafdcom" not in found


def test_a_brand_token_needs_to_be_meaningful() -> None:
    """A two-letter token would match half the web."""

    from agent_workflow.core.entities.models.social_expand import (
        _brand_token,
    )

    assert _brand_token("www.iafd.com") == "iafd"
    assert _brand_token("babepedia.com") == "babepedia"
    assert _brand_token("bb.co") == ""
    assert _brand_token("") == ""


def test_a_short_name_is_never_a_brand_token() -> None:
    from agent_workflow.core.entities.models.social_expand import (
        _brand_token,
    )

    # "ab.com" is all generic, so there is nothing to match on.
    assert _brand_token("com") == ""


@pytest.mark.parametrize(
    "url",
    [
        "https://x.com/LikaBusy/photo",
        "https://x.com/LikaBusy/media",
        "https://x.com/LikaBusy/following",
        "https://x.com/LikaBusy/verified_followers",
        "https://x.com/LikaBusy/with_replies",
        "https://x.com/LikaBusy/header_photo",
        "https://www.instagram.com/likabusyofficial/followers/",
    ],
)
def test_a_profiles_own_tabs_are_not_other_accounts(url: str) -> None:
    """The tabs sit after the handle, so the first segment misses them.

    Eleven of these once turned a single X profile into eleven
    "discoveries" and buried the one link that was new.
    """

    from agent_workflow.core.entities.models.social_expand import (
        _is_furniture,
    )

    assert _is_furniture(url)


@pytest.mark.parametrize(
    "url",
    [
        "https://x.com/LikaBusy",
        "https://www.instagram.com/likabusyofficial",
        "https://www.youtube.com/@LikaBusy",
    ],
)
def test_the_profile_itself_is_not_furniture(url: str) -> None:
    from agent_workflow.core.entities.models.social_expand import (
        _is_furniture,
    )

    assert not _is_furniture(url)


@pytest.mark.parametrize(
    "url",
    [
        "https://consent.youtube.com/dl?continue=https://youtube.com/@x",
        "https://accounts.google.com/ServiceLogin",
    ],
)
def test_a_consent_interstitial_is_not_an_account(url: str) -> None:
    from agent_workflow.core.entities.models.social_expand import (
        _is_furniture,
    )

    assert _is_furniture(url)


def test_one_account_in_thirteen_languages_is_one_account() -> None:
    """de/fr/it/... pornhub are the same profile in another language."""

    results = [
        SourceResult(
            url="https://www.pornhub.com/model/x/about",
            status="read",
            platform="Pornhub",
            found=(
                ("Pornhub", "https://www.pornhub.com/model/x/about"),
            ),
        )
    ]

    for locale in ("de", "fr", "it", "pt", "es", "nl", "pl"):
        url = f"https://{locale}.pornhub.com/model/x/about"
        results.append(
            SourceResult(
                url=url,
                status="read",
                platform="Pornhub",
                found=(("Pornhub", url),),
            )
        )

    results.append(
        SourceResult(
            url="https://x.com/x",
            status="read",
            platform="X",
            found=(("Telegram", "https://t.me/x"),),
        )
    )

    rendered = format_expand(results, "X")

    assert "de.pornhub.com" not in rendered
    assert "https://www.pornhub.com/model/x/about" in rendered
    assert "t.me/x" in rendered
    assert "another language" in rendered


def test_the_rule_needs_no_list_of_language_codes() -> None:
    """de, rt, cn, and whatever the next site invents.

    Three codes were missing from a hand-written list in one run, so
    the rule now compares hosts instead of guessing at a set.
    """

    from agent_workflow.core.entities.models.social_expand import (
        _dedupe_variants,
    )

    canonical = "https://www.pornhub.com/model/x/about"
    bucket = [(canonical, canonical)] + [
        (
            f"https://{prefix}.pornhub.com/model/x/about",
            canonical,
        )
        for prefix in ("de", "fr", "it", "rt", "cn", "fil", "zz")
    ]

    kept = [url for url, _source in _dedupe_variants(bucket)]

    assert kept == [canonical], kept

    # Different paths are different pages, even on related hosts.
    assert len(
        _dedupe_variants(
            [
                ("https://de.pornhub.com/models/other", "s"),
                ("https://www.pornhub.com/model/x/about", "s"),
            ]
        )
    ) == 2

    # Same path on a prefixed host is the same page, whichever host
    # the bucket happens to contain.
    assert len(
        _dedupe_variants(
            [
                ("https://de.pornhub.com/models/other", "s"),
                ("https://www.pornhub.com/models/other", "s"),
            ]
        )
    ) == 1


def test_separate_mirrors_at_a_bare_root_all_survive() -> None:
    """angieelif.bezoxo.com is not a translation of clips4sale.bezoxo.com.

    Dropping these would lose real sites, and they are the ones this
    tool exists to find.
    """

    from agent_workflow.core.entities.models.social_expand import (
        _dedupe_variants,
    )

    mirrors = [
        ("https://angieelif.bezoxo.com/", "src"),
        ("https://angieelifdate.bezoxo.com/", "src"),
        ("https://clips4sale.bezoxo.com/", "src"),
        ("https://skyprivate.bezoxo.com/", "src"),
    ]

    assert _dedupe_variants(mirrors) == mirrors

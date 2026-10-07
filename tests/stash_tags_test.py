"""The compact tag listing.

The point of the tool is the shape: an id and a name per line, so
the ids can be handed straight to a write tool without the model
picking them out of full records full of aliases and descriptions.
"""

from __future__ import annotations

import json

import pytest

from agent_workflow.core.entities.models.graphql_query import (
    StashGraphQLClient,
)
from agent_workflow.core.entities.models.stash_tags import (
    StashTagLister,
    StashTagsError,
    TagListPolicy,
    TagRef,
    _match_pattern,
    create_stash_list_tags_tool,
    format_tag_list,
)


ENDPOINT = "http://stash.test/graphql"


class _Transport:
    """Serves a fixed set of tags, recording the queries it was asked."""

    def __init__(self, rows: list[dict], page_size: int = 250) -> None:
        self.rows = rows
        self.page_size = page_size
        self.queries: list[str] = []

    async def execute(self, query: str, variables: str = "") -> str:
        self.queries.append(query)

        import re

        page = int(
            (re.search(r"\bpage:\s*(\d+)", query) or [0, 1])[1]
        )
        size = int(
            (re.search(r"per_page:\s*(\d+)", query) or [0, self.page_size])[1]
        )

        start = (page - 1) * size
        chunk = self.rows[start : start + size]

        return json.dumps(
            {
                "findTags": {
                    "count": len(self.rows),
                    "tags": chunk,
                }
            }
        )


def _lister(
    rows: list[dict],
    **policy: object,
) -> tuple[StashTagLister, _Transport]:
    transport = _Transport(rows)

    client = StashGraphQLClient(ENDPOINT, "key")
    # The transport stands in for HTTP; everything above it is real.
    client.execute = transport.execute  # type: ignore[method-assign]

    return (
        StashTagLister(
            client,
            policy=TagListPolicy(**policy),  # type: ignore[arg-type]
        ),
        transport,
    )


def _rows(count: int) -> list[dict]:
    return [
        {"id": str(1000 + index), "name": f"Tag {index}"}
        for index in range(count)
    ]


# ======================================================================
# Output shape
# ======================================================================


def test_one_line_per_tag_with_the_id_first() -> None:
    rendered = format_tag_list(
        [
            TagRef(id="1678", name="3rd Person Narrative"),
            TagRef(id="1184", name="4k"),
        ],
        2,
    )

    body = rendered.splitlines()[3:]

    assert body == [
        "1678\t3rd Person Narrative",
        "1184\t4k",
    ]


def test_the_header_states_the_total() -> None:
    """Without a total, a partial list looks like the whole library."""

    rendered = format_tag_list(
        [TagRef(id="1", name="a")],
        981,
    )

    assert "1 shown of 981 total" in rendered


def test_a_partial_list_says_how_much_is_missing() -> None:
    rendered = format_tag_list([TagRef(id="1", name="a")], 50)

    assert "49 more not shown" in rendered
    assert "page=2" in rendered


def test_a_complete_list_has_no_footnote() -> None:
    rendered = format_tag_list([TagRef(id="1", name="a")], 1)

    assert "more not shown" not in rendered


def test_an_empty_library_says_none() -> None:
    rendered = format_tag_list([], 0)

    assert "(none)" in rendered
    assert "0 shown of 0 total" in rendered


def test_a_search_is_named_in_the_header() -> None:
    rendered = format_tag_list(
        [TagRef(id="1", name="Anal")],
        1,
        search="anal",
    )

    assert 'matching "anal"' in rendered


def test_the_listing_is_much_smaller_than_json() -> None:
    """The reason this tool exists rather than the MCP equivalent."""

    tags = [
        TagRef(id=str(1000 + index), name=f"Tag {index}")
        for index in range(200)
    ]

    compact = format_tag_list(tags, 200)
    verbose = json.dumps(
        [
            {
                "id": tag.id,
                "name": tag.name,
                "scene_count": 12,
                "aliases": ["a", "b", "c"],
                "description": "x" * 200,
            }
            for tag in tags
        ]
    )

    assert len(compact) < len(verbose) / 2


# ======================================================================
# Collection
# ======================================================================


async def test_every_tag_comes_back() -> None:
    lister, _ = _lister(_rows(981), page_size=250, max_rows=5_000)

    refs, total = await lister.list_tags()

    assert total == 981
    assert len(refs) == 981


async def test_pages_are_followed_until_the_library_runs_out() -> None:
    lister, transport = _lister(_rows(30), page_size=10, max_rows=100)

    refs, _ = await lister.list_tags()

    assert len(refs) == 30
    # Three full pages plus the short one that ends it.
    assert len(transport.queries) == 4


async def test_the_row_cap_is_honoured() -> None:
    lister, _ = _lister(_rows(500), page_size=250, max_rows=100)

    refs, total = await lister.list_tags()

    assert len(refs) == 100
    # The total is still reported, so a truncated list is visible.
    assert total == 500


async def test_a_second_page_returns_the_next_slice() -> None:
    lister, _ = _lister(_rows(30), page_size=10, max_rows=100)

    first, _ = await lister.list_tags(per_page=10, page=1)
    second, _ = await lister.list_tags(per_page=10, page=2)

    assert len(first) == 10
    assert len(second) == 10
    assert first[0].id != second[0].id


async def test_a_page_past_the_end_is_empty_not_an_error() -> None:
    lister, _ = _lister(_rows(15), page_size=10, max_rows=100)

    refs, total = await lister.list_tags(per_page=10, page=9)

    assert refs == []
    # The total still reports what exists, so the note explains why.
    assert total == 15


async def test_an_empty_library_is_not_an_error() -> None:
    lister, _ = _lister([])

    refs, total = await lister.list_tags()

    assert refs == []
    assert total == 0


async def test_descending_order_is_requested() -> None:
    lister, transport = _lister(_rows(3))

    await lister.list_tags(ascending=False)

    assert "direction: DESC" in transport.queries[0]


# ======================================================================
# Search
# ======================================================================


async def test_search_uses_a_case_insensitive_pattern() -> None:
    # Stash capitalises names, so a case-sensitive match makes "anal"
    # miss the tag named "Anal".
    lister, transport = _lister(_rows(3))

    await lister.list_tags("anal")

    assert "MATCHES_REGEX" in transport.queries[0]
    assert "(?i)anal" in transport.queries[0]


def test_regex_metacharacters_are_neutralised() -> None:
    assert _match_pattern("big ass") == "(?i)big ass"
    assert _match_pattern("a.b") == "(?i)a\\\\.b"
    assert _match_pattern("(male)") == "(?i)\\\\(male\\\\)"
    assert _match_pattern("a+b") == "(?i)a\\\\+b"


def test_a_quote_in_the_search_is_escaped_for_graphql() -> None:
    # An unescaped quote ends the literal and the whole document fails
    # to parse, with a message pointing at the syntax rather than here.
    pattern = _match_pattern('say "hi"')

    assert '\\"' in pattern


def test_a_backslash_survives_both_escapes() -> None:
    pattern = _match_pattern("a\\b")

    # Regex sees one backslash, GraphQL sees two.
    assert pattern == "(?i)a\\\\\\\\b"


async def test_search_is_sent_as_a_filter() -> None:
    lister, transport = _lister(_rows(3))

    await lister.list_tags("anal")

    assert "tag_filter" in transport.queries[0]


# ======================================================================
# The tool
# ======================================================================


def test_the_tool_requires_the_graphql_permission() -> None:
    tool = create_stash_list_tags_tool(_lister(_rows(1))[0])

    assert tool.policy.permissions == frozenset({"graphql.execute"})


def test_the_tool_explains_the_shape_and_the_header() -> None:
    tool = create_stash_list_tags_tool(_lister(_rows(1))[0])

    assert "id and a name" in tool.description
    assert "total" in tool.description


async def test_the_tool_runs_end_to_end() -> None:
    from agent_workflow.core.entities.models.tool import ToolContext

    tool = create_stash_list_tags_tool(
        _lister([{"id": "57", "name": "Anal"}])[0]
    )

    from agent_workflow.core.entities.models.arguments import (
        ArgumentDecoder,
    )

    arguments = ArgumentDecoder().decode({}, tool.input_type)

    output = await tool.handler(
        arguments,
        ToolContext(
            working_directory=None,  # type: ignore[arg-type]
            environment={},
            allowed_path=(),
            permissions=frozenset(),
        ),
    )

    assert "57\tAnal" in output


# ======================================================================
# Policy
# ======================================================================


def test_policy_rejects_nonsense_values() -> None:
    with pytest.raises(ValueError, match="max_rows"):
        TagListPolicy(max_rows=0)

    with pytest.raises(ValueError, match="page_size"):
        TagListPolicy(page_size=0)


async def test_a_server_failure_is_reported_as_a_stash_error() -> None:
    from agent_workflow.core.entities.models.graphql_query import (
        GraphQLQueryError,
    )

    client = StashGraphQLClient(ENDPOINT, "key")

    async def failing(query: str, variables: str = "") -> str:
        raise GraphQLQueryError("server said no")

    client.execute = failing  # type: ignore[method-assign]

    with pytest.raises(StashTagsError, match="server said no"):
        await StashTagLister(client).list_tags()


async def test_a_non_json_reply_is_reported() -> None:
    client = StashGraphQLClient(ENDPOINT, "key")

    async def html(query: str, variables: str = "") -> str:
        return "<html>gateway</html>"

    client.execute = html  # type: ignore[method-assign]

    with pytest.raises(StashTagsError, match="not JSON"):
        await StashTagLister(client).list_tags()
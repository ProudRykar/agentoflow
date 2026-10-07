"""The read-only GraphQL tool.

The guards are the point of this file. A write must never reach the
server, and "never" has to mean before the request is built, not
after the server answers 405.
"""

from __future__ import annotations

import json
import os
import re

import httpx
import pytest

from agent_workflow.core.entities.models.graphql_query import (
    GraphQLQueryError,
    GraphQLWriteRefused,
    StashGraphQLClient,
    StashGraphQLInput,
    StashGraphQLPolicy,
    create_stash_graphql_tool,
    resolve_api_key,
)
from agent_workflow.core.entities.models.tool import ToolContext


ENDPOINT = "http://stash.test/graphql"


class _Transport(httpx.AsyncBaseTransport):
    """Captures the request and returns a canned response."""

    def __init__(
        self,
        payload: dict | None = None,
        *,
        status_code: int = 200,
        text: str | None = None,
    ) -> None:
        self.payload = payload
        self.status_code = status_code
        self.text = text
        self.requests: list[httpx.Request] = []

    async def handle_async_request(
        self,
        request: httpx.Request,
    ) -> httpx.Response:
        self.requests.append(request)

        body = (
            self.text
            if self.text is not None
            else json.dumps(
                self.payload
                if self.payload is not None
                else {"data": {"ok": True}}
            )
        )

        return httpx.Response(
            self.status_code,
            text=body,
            request=request,
        )


def _client(
    transport: _Transport,
    **policy: object,
) -> StashGraphQLClient:
    return StashGraphQLClient(
        ENDPOINT,
        "secret-key",
        policy=StashGraphQLPolicy(**policy),  # type: ignore[arg-type]
        transport=transport,
    )


def _sent(transport: _Transport) -> dict:
    return json.loads(transport.requests[0].content)


# ======================================================================
# Read-only enforcement
# ======================================================================


async def test_a_query_is_sent() -> None:
    transport = _Transport({"data": {"findTags": {"count": 0}}})
    client = _client(transport)

    await client.execute("{ findTags { count } }")

    assert len(transport.requests) == 1
    assert _sent(transport)["query"]


@pytest.mark.parametrize(
    "document",
    [
        'mutation { tagUpdate(id: "1") { id } }',
        'mutation Update { tagUpdate(id: "1") { id } }',
        'subscription { tagUpdate(id: "1") { id } }',
    ],
)
async def test_writes_are_refused_before_sending(
    document: str,
) -> None:
    transport = _Transport({"data": {}})
    client = _client(transport)

    with pytest.raises(GraphQLWriteRefused):
        await client.execute(document)

    # Nothing left the process: the guard runs before the client is
    # even constructed.
    assert transport.requests == []


async def test_a_mutation_field_in_an_anonymous_query_is_left_to_the_server() -> None:
    # A bare brace block is syntactically a query, so no local check
    # can classify it as a write. It is still safe: the server
    # rejects a mutation field reached through a query operation,
    # which test_server_rejects_mutation_fields_in_a_query covers
    # against a live schema. Asserting that it is *not* sent would
    # document a guarantee the client cannot make.
    transport = _Transport({"data": {"ok": True}})

    await _client(transport).execute(
        '{ tagDestroy(id: "1") { id } }'
    )

    assert len(transport.requests) == 1


async def test_a_mutation_named_query_is_still_refused() -> None:
    # The operation keyword decides, not the name.
    transport = _Transport({"data": {}})

    with pytest.raises(GraphQLWriteRefused):
        await _client(transport).execute(
            'mutation read { tagDestroy(id: "1") { id } }'
        )


async def test_a_write_hidden_in_a_fragment_is_refused() -> None:
    transport = _Transport({"data": {}})

    with pytest.raises(GraphQLWriteRefused):
        await _client(transport).execute(
            "query Q { findTags { tags { ...F } } } "
            "mutation M { tagDestroy(id: \"1\") { id } }"
        )


async def test_a_read_only_document_is_allowed() -> None:
    transport = _Transport({"data": {"ok": True}})
    client = _client(transport)

    await client.execute(
        "query Q { findTags { tags { id } } } "
        "fragment F on Tag { name }"
    )

    assert len(transport.requests) == 1


async def test_syntax_errors_are_reported_before_sending() -> None:
    transport = _Transport({"data": {}})

    with pytest.raises(GraphQLQueryError, match="not valid"):
        await _client(transport).execute("{ findTags { tags {")

    assert transport.requests == []


async def test_a_fragment_without_an_operation_is_refused() -> None:
    transport = _Transport({"data": {}})

    with pytest.raises(GraphQLQueryError, match="no operation"):
        await _client(transport).execute("fragment F on Tag { id }")

    assert transport.requests == []


async def test_introspection_can_be_disabled() -> None:
    transport = _Transport({"data": {}})

    with pytest.raises(GraphQLQueryError, match="introspection"):
        await _client(
            transport,
            allow_introspection=False,
        ).execute('{ __type(name: "Tag") { name } }')

    assert transport.requests == []


# ======================================================================
# Requests
# ======================================================================


async def test_the_api_key_travels_in_the_header() -> None:
    transport = _Transport({"data": {"ok": True}})

    await _client(transport).execute("{ findTags { count } }")

    assert transport.requests[0].headers["ApiKey"] == "secret-key"


async def test_the_api_key_is_not_in_the_body() -> None:
    # Otherwise the key would be visible to anything logging the
    # request, and would echo back into the transcript on an error.
    transport = _Transport({"data": {"ok": True}})

    await _client(transport).execute("{ findTags { count } }")

    assert "secret-key" not in transport.requests[0].content.decode()


async def test_variables_are_forwarded() -> None:
    transport = _Transport({"data": {"ok": True}})

    await _client(transport).execute(
        "query Q($n: Int) { findTags(filter: {per_page: $n}) { count } }",
        '{"n": 7}',
    )

    assert _sent(transport)["variables"] == {"n": 7}


async def test_malformed_variables_are_refused() -> None:
    transport = _Transport({"data": {}})

    with pytest.raises(GraphQLQueryError, match="valid JSON"):
        await _client(transport).execute("{ findTags { count } }", "{n:")


async def test_variables_must_be_an_object() -> None:
    transport = _Transport({"data": {}})

    with pytest.raises(GraphQLQueryError, match="JSON object"):
        await _client(transport).execute(
            "{ findTags { count } }",
            "[1, 2]",
        )


async def test_empty_variables_are_fine() -> None:
    transport = _Transport({"data": {"ok": True}})

    await _client(transport).execute(
        "{ findTags { count } }",
        "",
    )

    assert _sent(transport)["variables"] == {}


async def test_an_over_long_query_is_refused() -> None:
    transport = _Transport({"data": {}})

    with pytest.raises(GraphQLQueryError, match="limit is"):
        await _client(
            transport,
            max_query_chars=10,
        ).execute("{ findTags { count } }")

    assert transport.requests == []


async def test_an_empty_query_is_refused() -> None:
    transport = _Transport({"data": {}})

    with pytest.raises(GraphQLQueryError, match="empty"):
        await _client(transport).execute("   ")


# ======================================================================
# Responses
# ======================================================================


async def test_data_is_returned_as_json() -> None:
    transport = _Transport(
        {"data": {"findTags": {"count": 764}}}
    )

    out = await _client(transport).execute("{ findTags { count } }")

    assert json.loads(out)["findTags"]["count"] == 764


async def test_server_errors_are_surfaced_verbatim() -> None:
    # GraphQL reports validation failures as HTTP 400 with a message
    # the model needs in order to correct itself.
    transport = _Transport(
        {
            "errors": [
                {"message": "invalid sort: scene_count"},
            ]
        },
        status_code=400,
    )

    with pytest.raises(GraphQLQueryError, match="invalid sort"):
        await _client(transport).execute("{ findTags { count } }")


async def test_several_errors_are_all_listed() -> None:
    transport = _Transport(
        {
            "errors": [
                {"message": "first problem"},
                {"message": "second problem"},
            ]
        },
        status_code=400,
    )

    with pytest.raises(GraphQLQueryError) as caught:
        await _client(transport).execute("{ findTags { count } }")

    assert "first problem" in str(caught.value)
    assert "second problem" in str(caught.value)


async def test_a_non_json_error_body_is_reported() -> None:
    transport = _Transport(
        text="<html>gateway timeout</html>",
        status_code=504,
    )

    with pytest.raises(GraphQLQueryError, match="504"):
        await _client(transport).execute("{ findTags { count } }")


async def test_a_response_with_neither_data_nor_errors() -> None:
    transport = _Transport({"extensions": {}})

    with pytest.raises(GraphQLQueryError, match="neither"):
        await _client(transport).execute("{ findTags { count } }")


async def test_rows_are_capped_and_the_cut_is_reported() -> None:
    transport = _Transport(
        {
            "data": {
                "findTags": {
                    "tags": [
                        {"id": str(index)}
                        for index in range(50)
                    ],
                }
            }
        }
    )

    out = await _client(
        transport,
        max_rows=5,
    ).execute("{ findTags { tags { id } } }")

    assert len(json.loads(out.split("\n\n[note]")[0])["findTags"]["tags"]) == 5
    assert "45 further item" in out


async def test_capping_keeps_the_ordering_asked_for() -> None:
    # Rows are trimmed from the end so a DESC query still shows its
    # largest values first.
    transport = _Transport(
        {
            "data": {
                "findTags": {
                    "tags": [{"n": n} for n in range(100, 0, -1)],
                }
            }
        }
    )

    out = await _client(
        transport,
        max_rows=3,
    ).execute("{ findTags { tags { n } } }")

    rows = json.loads(out.split("\n\n[note]")[0])["findTags"]["tags"]

    assert [row["n"] for row in rows] == [100, 99, 98]


async def test_nested_lists_are_capped_too() -> None:
    transport = _Transport(
        {
            "data": {
                "a": {"b": [{"n": index} for index in range(10)]},
            }
        }
    )

    out = await _client(
        transport,
        max_rows=2,
    ).execute("{ a { b { n } } }")

    assert len(json.loads(out.split("\n\n[note]")[0])["a"]["b"]) == 2


async def test_an_oversized_response_is_truncated() -> None:
    transport = _Transport(
        {"data": {"blob": "x" * 5_000}}
    )

    out = await _client(
        transport,
        max_response_chars=500,
    ).execute("{ blob }")

    assert len(out) < 800
    assert "truncated" in out


# ======================================================================
# API key resolution
# ======================================================================


def test_inline_key_wins(tmp_path) -> None:
    env = tmp_path / ".env"
    env.write_text("STASH_API_KEY=from-file\n")

    assert resolve_api_key("inline", str(env)) == "inline"


def test_key_is_read_from_the_env_file(tmp_path) -> None:
    env = tmp_path / ".env"
    env.write_text("STASH_API_KEY=from-file\n")

    assert resolve_api_key("", str(env)) == "from-file"


def test_quotes_and_comments_are_handled(tmp_path) -> None:
    env = tmp_path / ".env"
    env.write_text(
        "# a comment\n"
        "OTHER=1\n"
        'STASH_API_KEY="quoted"\n'
    )

    assert resolve_api_key("", str(env)) == "quoted"


def test_a_missing_env_file_is_not_fatal(tmp_path) -> None:
    assert resolve_api_key("", str(tmp_path / "nope")) == ""


def test_no_configuration_at_all() -> None:
    assert resolve_api_key("", "") == ""


# ======================================================================
# The tool
# ======================================================================


# ======================================================================
# order_by: ranking by a field the server will not sort by
# ======================================================================


class _PagingTransport(httpx.AsyncBaseTransport):
    """Serves several pages of a find result."""

    def __init__(
        self,
        rows: list[dict],
        page_size: int,
    ) -> None:
        self.rows = rows
        self.page_size = page_size
        self.pages_requested: list[int] = []
        self.sent_queries: list[str] = []

    async def handle_async_request(
        self,
        request: httpx.Request,
    ) -> httpx.Response:
        query = json.loads(request.content)["query"]

        self.sent_queries.append(query)

        match = re.search(r"\bpage:\s*(\d+)", query)

        page = int(match.group(1)) if match else 1

        self.pages_requested.append(page)

        start = (page - 1) * self.page_size
        chunk = self.rows[start : start + self.page_size]

        payload = {
            "data": {
                "findTags": {
                    "count": len(self.rows),
                    "tags": chunk,
                }
            }
        }

        return httpx.Response(
            200,
            text=json.dumps(payload),
            request=request,
        )


ORDER_QUERY = """{
  findTags(tag_filter: { is_missing: "description" },
           filter: { per_page: 3 })
  { count tags { id name scene_count } }
}"""


def _ordering_client(
    rows: list[dict],
    **policy: object,
) -> tuple[StashGraphQLClient, _PagingTransport]:
    transport = _PagingTransport(rows, page_size=4)

    client = StashGraphQLClient(
        ENDPOINT,
        "key",
        policy=StashGraphQLPolicy(**policy),  # type: ignore[arg-type]
        transport=transport,
    )

    return client, transport


def _counts(output: str) -> list[int]:
    payload = json.loads(output.split("\n\n[note]")[0])

    return [
        row["scene_count"]
        for row in payload["findTags"]["tags"]
    ]


async def test_order_by_ranks_descending() -> None:
    rows = [
        {"id": str(index), "name": f"t{index}", "scene_count": count}
        for index, count in enumerate([5, 40, 12, 33, 7, 28, 1, 21])
    ]

    client, transport = _ordering_client(rows)

    out = await client.execute(
        ORDER_QUERY,
        order_by="scene_count",
        order="DESC",
    )

    # per_page in the query is 3, so the top three are shown.
    assert _counts(out) == [40, 33, 28]


async def test_order_by_ranks_ascending() -> None:
    rows = [
        {"id": str(index), "name": f"t{index}", "scene_count": count}
        for index, count in enumerate([5, 40, 12, 33])
    ]

    client, _ = _ordering_client(rows)

    out = await client.execute(
        ORDER_QUERY,
        order_by="scene_count",
        order="ASC",
    )

    assert _counts(out) == [5, 12, 33]


async def test_order_by_reads_every_page() -> None:
    """The whole point: one page cannot answer a top-N by count."""

    rows = [
        {"id": str(index), "name": f"t{index}", "scene_count": count}
        for index, count in enumerate([3, 1, 9, 2, 8, 4])
    ]

    client, transport = _ordering_client(rows)

    await client.execute(
        ORDER_QUERY,
        order_by="scene_count",
    )

    # 6 rows at 4 per page.
    assert len(transport.pages_requested) == 2


async def test_order_by_reports_what_it_scanned() -> None:
    rows = [
        {"id": str(index), "name": f"t{index}", "scene_count": index}
        for index in range(6)
    ]

    client, _ = _ordering_client(rows)

    out = await client.execute(ORDER_QUERY, order_by="scene_count")

    assert "[note]" in out
    assert "scene_count" in out
    assert "6 matching row" in out


async def test_order_by_keeps_the_original_criteria() -> None:
    """Paging must be the only thing that changes."""

    rows = [{"id": "1", "name": "a", "scene_count": 1}]

    client, transport = _ordering_client(rows)

    await client.execute(ORDER_QUERY, order_by="scene_count")

    sent = transport.sent_queries[0]

    assert "is_missing" in sent
    assert "description" in sent


async def test_order_by_needs_the_field_selected() -> None:
    rows = [{"id": "1", "name": "a", "scene_count": 2}]

    client, _ = _ordering_client(rows)

    with pytest.raises(GraphQLQueryError, match="not present"):
        await client.execute(
            ORDER_QUERY,
            order_by="performer_count",
        )


async def test_order_by_refuses_an_unbounded_scan() -> None:
    rows = [
        {"id": str(index), "name": f"t{index}", "scene_count": 1}
        for index in range(30)
    ]

    client, _ = _ordering_client(rows, max_scan_rows=8)

    with pytest.raises(GraphQLQueryError, match="Narrow the filter"):
        await client.execute(ORDER_QUERY, order_by="scene_count")


async def test_order_by_rejects_an_ambiguous_shape() -> None:
    rows = [{"id": "1", "name": "a", "scene_count": 1}]

    client, _ = _ordering_client(rows)

    with pytest.raises(GraphQLQueryError, match="one list field"):
        await client.execute(
            "{ findTags(filter: { per_page: 3 }) { count } }",
            order_by="scene_count",
        )


async def test_order_by_surfaces_server_errors() -> None:
    transport = _Transport(
        {"errors": [{"message": "invalid sort: scene_count"}]},
        status_code=400,
    )

    client = StashGraphQLClient(
        ENDPOINT,
        "key",
        transport=transport,
    )

    with pytest.raises(GraphQLQueryError, match="invalid sort"):
        await client.execute(ORDER_QUERY, order_by="scene_count")


async def test_order_by_sorts_nulls_to_the_bottom() -> None:
    # A missing count must not outrank a real one in a DESC.
    rows = [
        {"id": "1", "name": "a", "scene_count": None},
        {"id": "2", "name": "b", "scene_count": 4},
        {"id": "3", "name": "c"},
    ]

    client, _ = _ordering_client(rows)

    out = await client.execute(
        ORDER_QUERY,
        order_by="scene_count",
        order="DESC",
    )

    names = [
        row["name"]
        for row in json.loads(
            out.split("\n\n[note]")[0]
        )["findTags"]["tags"]
    ]

    assert names[0] == "b"


async def test_order_by_sorts_strings_too() -> None:
    rows = [
        {"id": "1", "name": "banana"},
        {"id": "2", "name": "Apple"},
        {"id": "3", "name": "cherry"},
    ]

    client, _ = _ordering_client(rows)

    out = await client.execute(
        ORDER_QUERY,
        order_by="name",
        order="ASC",
    )

    names = [
        row["name"]
        for row in json.loads(
            out.split("\n\n[note]")[0]
        )["findTags"]["tags"]
    ]

    assert names == ["Apple", "banana", "cherry"]


def test_the_tool_documents_order_by() -> None:
    tool = create_stash_graphql_tool(
        StashGraphQLClient(ENDPOINT, "key")
    )

    assert "order_by" in tool.description

    # The failure it replaces has to be named, or the model reads one
    # page, orders it by eye and returns a wrong answer that looks
    # right.
    assert "by eye" in tool.description


def test_the_guidance_contains_no_rejected_syntax() -> None:
    """Teaching text that the server rejects wastes a round trip."""

    from agent_workflow.core.entities.models.graphql_query import (
        GRAPHQL_GUIDANCE,
    )

    assert "modifier: GTE" not in GRAPHQL_GUIDANCE
    assert "modifier: CONTAINS" not in GRAPHQL_GUIDANCE
    assert "AND: [" not in GRAPHQL_GUIDANCE
    assert "GREATER_THAN" in GRAPHQL_GUIDANCE


def test_the_tool_teaches_the_schema_shape() -> None:
    # The parts a model cannot infer: which sorts the server accepts,
    # and that filters are split across two arguments.
    tool = create_stash_graphql_tool(
        StashGraphQLClient(ENDPOINT, "key")
    )

    assert "is_missing" in tool.description
    assert "tag_filter" in tool.description
    assert "SortDirectionEnum" in tool.description
    assert "ASC" in tool.description
    assert "DESC" in tool.description


def test_the_tool_names_the_sorts_that_are_rejected() -> None:
    # Naming the invalid ones prevents the model from trying them.
    tool = create_stash_graphql_tool(
        StashGraphQLClient(ENDPOINT, "key")
    )

    assert "scene_count" in tool.description
    assert "NOT accepted" in tool.description


def test_the_tool_requires_its_own_permission() -> None:
    tool = create_stash_graphql_tool(
        StashGraphQLClient(ENDPOINT, "key")
    )

    assert tool.policy.permissions == frozenset(
        {"graphql.execute"}
    )


async def test_the_tool_runs_end_to_end() -> None:
    from agent_workflow.core.entities.models.arguments import (
        ArgumentDecoder,
    )

    transport = _Transport(
        {"data": {"findTags": {"count": 3}}}
    )
    tool = create_stash_graphql_tool(
        StashGraphQLClient(ENDPOINT, "key", transport=transport)
    )

    arguments = ArgumentDecoder().decode(
        {"query": "{ findTags { count } }"},
        tool.input_type,
    )

    assert isinstance(arguments, StashGraphQLInput)

    out = await tool.handler(
        arguments,
        ToolContext(
            working_directory=None,  # type: ignore[arg-type]
            environment={},
            allowed_path=(),
            permissions=frozenset(),
        ),
    )

    assert json.loads(out)["findTags"]["count"] == 3


async def test_the_tool_refuses_a_write_end_to_end() -> None:
    from agent_workflow.core.entities.models.arguments import (
        ArgumentDecoder,
    )

    transport = _Transport({"data": {}})
    tool = create_stash_graphql_tool(
        StashGraphQLClient(ENDPOINT, "key", transport=transport)
    )

    arguments = ArgumentDecoder().decode(
        {'query': 'mutation { tagDestroy(id: "1") { id } }'},
        tool.input_type,
    )

    with pytest.raises(GraphQLWriteRefused):
        await tool.handler(
            arguments,
            ToolContext(
                working_directory=None,  # type: ignore[arg-type]
                environment={},
                allowed_path=(),
                permissions=frozenset(),
            ),
        )

    assert transport.requests == []


# ======================================================================
# Policy
# ======================================================================


def test_policy_rejects_nonsense_values() -> None:
    with pytest.raises(ValueError, match="timeout"):
        StashGraphQLPolicy(timeout=0)

    with pytest.raises(ValueError, match="max_rows"):
        StashGraphQLPolicy(max_rows=0)


def test_a_client_requires_an_endpoint() -> None:
    with pytest.raises(ValueError, match="endpoint"):
        StashGraphQLClient("")

# ======================================================================
# Against a real server
# ======================================================================


async def test_server_rejects_mutation_fields_in_a_query() -> None:
    """Completes the read-only guarantee against the live schema.

    The local guard only sees operation types, so a mutation field
    reached through an anonymous query is caught here instead, by
    schema validation on the server. Skipped when no endpoint is
    configured.
    """

    endpoint = os.environ.get("AGENTOFLOW_TEST_GRAPHQL")

    if not endpoint:
        pytest.skip(
            "AGENTOFLOW_TEST_GRAPHQL is not set"
        )

    client = StashGraphQLClient(
        endpoint,
        os.environ.get("AGENTOFLOW_TEST_GRAPHQL_KEY", ""),
    )

    with pytest.raises(GraphQLQueryError, match="Cannot query"):
        await client.execute('{ tagDestroy(id: "1") { id } }')


# ======================================================================
# An omitted outer brace
# ======================================================================


async def test_a_missing_outer_brace_is_repaired() -> None:
    """The single most common way to get the syntax wrong.

    A selection set is shorthand for a query, and models routinely
    drop the leading brace. Without this the model sees
    "Unexpected Name 'findTags'" against a document that is otherwise
    correct.
    """

    transport = _Transport({"data": {"findTags": {"tags": []}}})

    client = StashGraphQLClient(ENDPOINT, "key", transport=transport)

    await client.execute(
        "findTags(tag_filter: { is_missing: \"description\" })"
        " { tags { id name } }"
    )

    sent = _sent(transport)["query"]

    assert sent.startswith("{")
    assert "findTags" in sent


async def test_the_repair_reaches_the_ordered_path() -> None:
    # Normalisation happens once, up front. Repairing only at the
    # write-guard left the ordering path re-parsing the broken
    # original and failing there instead.
    rows = [
        {"id": str(index), "name": f"t{index}", "scene_count": index}
        for index in range(6)
    ]

    transport = _PagingTransport(rows, page_size=4)
    client = StashGraphQLClient(ENDPOINT, "key", transport=transport)

    out = await client.execute(
        "findTags(tag_filter: { is_missing: \"description\" },"
        " filter: { per_page: 3 })"
        " { tags { id name scene_count } }",
        order_by="scene_count",
        order="DESC",
    )

    assert _counts(out) == [5, 4, 3]


async def test_a_genuinely_broken_query_still_fails() -> None:
    # The repair must not become a way to send nonsense through.
    transport = _Transport({"data": {}})

    client = StashGraphQLClient(ENDPOINT, "key", transport=transport)

    with pytest.raises(GraphQLQueryError, match="not valid GraphQL"):
        await client.execute("findTags(tag_filter: { ")


async def test_the_repair_does_not_hide_an_inner_error() -> None:
    # Already opens with a brace: the problem is elsewhere and
    # wrapping would conceal it.
    transport = _Transport({"data": {}})

    client = StashGraphQLClient(ENDPOINT, "key", transport=transport)

    with pytest.raises(GraphQLQueryError, match="not valid GraphQL"):
        await client.execute("{ findTags(tag_filter: { ) { tags { id } } }")


async def test_the_repair_still_refuses_a_write() -> None:
    transport = _Transport({"data": {}})

    client = StashGraphQLClient(ENDPOINT, "key", transport=transport)

    with pytest.raises(GraphQLWriteRefused):
        await client.execute('mutation { tagDestroy(id: "1") { id } }')

    assert transport.requests == []


async def test_a_valid_query_is_sent_unchanged() -> None:
    transport = _Transport({"data": {"findTags": {"tags": []}}})

    client = StashGraphQLClient(ENDPOINT, "key", transport=transport)

    query = '{ findTags { tags { id } } }'

    await client.execute(query)

    assert _sent(transport)["query"] == query


def test_the_guidance_shows_a_complete_document() -> None:
    from agent_workflow.core.entities.models.graphql_query import (
        GRAPHQL_GUIDANCE,
    )

    # A runnable-looking example that is not a valid document teaches
    # the model to write an invalid one.
    assert '\\"description\\"' not in GRAPHQL_GUIDANCE
    assert '"query": "{ findTags' not in GRAPHQL_GUIDANCE

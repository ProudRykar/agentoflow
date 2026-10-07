from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import httpx
from graphql import (
    DocumentNode,
    FieldNode,
    IntValueNode,
    NameNode,
    ObjectFieldNode,
    ObjectValueNode,
    OperationDefinitionNode,
    OperationType,
    SelectionSetNode,
    parse,
    print_ast,
)

from agent_workflow.core.entities.models.tool import (
    Tool,
    ToolContext,
    ToolPolicy,
)


class GraphQLQueryError(RuntimeError):
    """The query was refused, malformed, or rejected by the server."""


class GraphQLWriteRefused(GraphQLQueryError):
    """The document contains an operation that writes.

    Raised before anything is sent. The distinction matters: the
    caller needs to know the request never left the process.
    """


@dataclass(slots=True, frozen=True)
class StashGraphQLInput:
    query: str
    """A GraphQL document. Queries only; writes are refused."""

    variables: str = ""
    """Optional JSON object, as a string.

    A string rather than a nested object because the model writes
    JSON far more reliably than it nests a dict inside a tool call.
    """

    order_by: str = ""
    """Order the rows by this field, descending.

    Stash sorts only by name, created_at, updated_at and id. It
    refuses scene_count, description and favorite with "invalid sort",
    and there is no query shape that gets around that. This sorts the
    result here instead, which means paging through every matching row
    first: a top-N by count cannot be taken from one page, because
    that page is ordered by name and holds whatever the alphabet
    happens to put there.

    Leave empty when the server can do the sorting.
    """

    order: str = "DESC"
    """ASC or DESC, used with order_by."""



@dataclass(slots=True, frozen=True)
class StashGraphQLPolicy:
    """Trusted runtime limits.

    The tool can read the whole library, so the runtime decides how
    much one call may cost.
    """

    timeout: float = 30.0

    max_query_chars: int = 8_000

    max_response_chars: int = 60_000

    # Stash ignores per_page above a few hundred, but a query can ask
    # for a lot and the response is truncated for the model anyway.
    max_rows: int = 250

    # Ceiling on how many rows client-side ordering may pull before it
    # gives up. Ordering by a count has to read every match, so this is
    # what stops a wide filter from becoming an unbounded crawl.
    max_scan_rows: int = 5_000

    scan_page_size: int = 250

    allow_introspection: bool = True

    def __post_init__(self) -> None:
        if self.timeout <= 0:
            raise ValueError("timeout must be > 0")

        if self.max_query_chars < 1:
            raise ValueError("max_query_chars must be >= 1")

        if self.max_response_chars < 1:
            raise ValueError("max_response_chars must be >= 1")

        if self.max_rows < 1:
            raise ValueError("max_rows must be >= 1")

        if self.max_scan_rows < 1:
            raise ValueError("max_scan_rows must be >= 1")

        if self.scan_page_size < 1:
            raise ValueError("scan_page_size must be >= 1")


class StashGraphQLClient:
    """Runs read-only GraphQL against a Stash server.

    The API key is attached here from the configuration and is never
    part of the arguments, so it cannot reach the model's context or
    the transcript.
    """

    def __init__(
        self,
        endpoint: str,
        api_key: str = "",
        *,
        policy: StashGraphQLPolicy | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        if not endpoint.strip():
            raise ValueError(
                "A GraphQL endpoint is required"
            )

        self._endpoint = endpoint.strip()
        self._api_key = api_key.strip()
        self._policy = policy or StashGraphQLPolicy()
        # Injectable so the guards can be tested without a server.
        self._transport = transport

    @property
    def policy(self) -> StashGraphQLPolicy:
        return self._policy

    @property
    def configured(self) -> bool:
        return bool(self._endpoint)

    async def execute(
        self,
        query: str,
        variables: str = "",
        *,
        order_by: str = "",
        order: str = "DESC",
    ) -> str:
        document = query.strip()

        if not document:
            raise GraphQLQueryError("The query is empty")

        if len(document) > self._policy.max_query_chars:
            raise GraphQLQueryError(
                f"Query is {len(document)} characters, "
                f"limit is {self._policy.max_query_chars}"
            )

        # Normalised once, up front: every later parse in this class
        # works from the same text, so a repaired document is not
        # re-parsed in its broken original form further down.
        document = _normalize_document(document)

        self._refuse_writes(document)

        payload_variables = self._decode_variables(variables)

        if order_by.strip():
            return await self._execute_ordered(
                document,
                payload_variables,
                order_by.strip(),
                order,
            )

        response_text, status = await self._post(
            document,
            payload_variables,
        )

        return self._render_text(response_text, status)

    async def _post(
        self,
        query: str,
        variables: dict[str, object],
    ) -> tuple[str, int]:
        """Send one document, returning its body and HTTP status.

        Shared with the ordered path so a failure surfaces the same way
        whichever route produced the query. The status comes back
        because a gateway error is HTML, and "HTTP 504" says more than
        a parse failure does.
        """

        headers = {
            "Content-Type": "application/json",
        }

        if self._api_key:
            headers["ApiKey"] = self._api_key

        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(
                    self._policy.timeout,
                    connect=min(
                        self._policy.timeout,
                        10.0,
                    ),
                ),
                trust_env=False,
                transport=self._transport,
            ) as client:
                response = await client.post(
                    self._endpoint,
                    json={
                        "query": query,
                        "variables": variables,
                    },
                    headers=headers,
                )
        except httpx.HTTPError as exc:
            raise GraphQLQueryError(
                f"Request to the GraphQL endpoint failed: {exc}"
            ) from exc

        return response.text, response.status_code

    # ------------------------------------------------------------------
    # Client-side ordering
    # ------------------------------------------------------------------

    async def _execute_ordered(
        self,
        document: str,
        variables: dict[str, object],
        order_by: str,
        order: str,
    ) -> str:
        """Order rows by a field the server will not sort by.

        A top-N by count cannot be read off one page: the page is
        ordered by name, so it holds whichever rows the alphabet put
        there. Every match has to be read before the largest ten are
        known, which is what this does.
        """

        descending = order.upper() != "ASC"

        shape = _scan_shape(document)

        page = 1
        per_page = self._policy.scan_page_size

        collected: list[dict[str, object]] = []
        total = 0

        while True:
            scan = _rebuild_scan_query(
                document,
                page=page,
                per_page=per_page,
            )

            raw, status = await self._post(scan, variables)

            try:
                payload = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise GraphQLQueryError(
                    f"HTTP {status} from the GraphQL endpoint: "
                    f"{raw[:300]}"
                ) from exc

            errors = payload.get("errors")

            if errors:
                raise GraphQLQueryError(
                    "The server rejected the query:\n"
                    + "\n".join(
                        f"- {item.get('message', item)}"
                        for item in errors
                    )
                )

            block = (payload.get("data") or {}).get(
                shape.root_field
            ) or {}

            if "count" in block:
                total = int(block.get("count") or 0)

            rows = block.get(shape.rows_field) or []

            if isinstance(rows, list):
                collected.extend(
                    row for row in rows if isinstance(row, dict)
                )

            if not rows or len(collected) >= total:
                break

            if len(collected) >= self._policy.max_scan_rows:
                raise GraphQLQueryError(
                    f"Ordering by '{order_by}' needs every match, and "
                    f"this filter has more than "
                    f"{self._policy.max_scan_rows} of them "
                    f"(count reports {total}). Narrow the filter, or "
                    "raise graphql.max_scan_rows."
                )

            page += 1

        # Only a field absent from every row is an error. A row that
        # simply lacks it carries a null, and refusing to order those
        # would make a sparse selection unusable.
        if collected and not any(
            order_by in row for row in collected
        ):
            raise GraphQLQueryError(
                f"order_by '{order_by}' is not present in the "
                "results, so there is nothing to sort on. Add it to "
                "the selection set, for example "
                f"{{ {shape.rows_field} {{ {order_by} }} }}."
            )

        collected.sort(
            key=lambda row: _sort_key(row.get(order_by)),
            reverse=descending,
        )

        # The caller asked for a page in their own query; honour it
        # after ordering, so per_page and page mean what they say.
        requested_page, requested_per_page = _requested_window(
            document
        )

        start = (requested_page - 1) * requested_per_page
        window = collected[start : start + requested_per_page]

        payload = {
            shape.root_field: {
                "count": total or len(collected),
                shape.rows_field: window,
            }
        }

        body = json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        )

        note = (
            f"[note] Ordered by {order_by} "
            f"{'DESC' if descending else 'ASC'} in the tool, over "
            f"{len(collected)} matching row(s) read across "
            f"{page} request(s); Stash itself sorts only by name, "
            "created_at, updated_at and id."
        )

        if len(collected) > len(window):
            note += (
                f" Showing {len(window)} of them; raise per_page "
                "for more."
            )

        if len(body) > self._policy.max_response_chars:
            body = (
                body[: self._policy.max_response_chars]
                + "\n... [truncated by the runtime limit]"
            )

        return f"{body}\n\n{note}"

    # ------------------------------------------------------------------
    # Guards
    # ------------------------------------------------------------------

    def _refuse_writes(self, document: str) -> None:
        """Reject any operation that is not a query.

        Checked by parsing the document rather than by matching the
        word "mutation": an operation's *name* is not its type, and a
        document can carry several operations.

        This catches every declared mutation. It cannot catch a
        mutation field hidden in an anonymous query, because a bare
        brace block is syntactically a query and only the schema says
        otherwise. That second case is covered by the server, which
        refuses to query a mutation field ("Cannot query field
        \"tagDestroy\" on type \"Query\""), so the two layers
        together leave no way to write.
        """

        parsed = parse(document)

        operations = [
            node
            for node in parsed.definitions
            if isinstance(
                node, OperationDefinitionNode
            )
        ]

        if not operations:
            raise GraphQLQueryError(
                "The document contains no operation. Send a "
                "query, not just fragments."
            )

        writes = [
            node
            for node in operations
            if node.operation is not OperationType.QUERY
        ]

        if writes:
            kinds = ", ".join(
                sorted(
                    {
                        node.operation.value
                        for node in writes
                    }
                )
            )

            raise GraphQLWriteRefused(
                f"This tool is read-only and the document "
                f"contains a {kinds} operation. Nothing was sent. "
                "Use the dedicated Stash MCP tools "
                "(update_tag_description and the rest) for writes."
            )

        if not self._policy.allow_introspection and (
            "__schema" in document or "__type" in document
        ):
            raise GraphQLQueryError(
                "Schema introspection is disabled in the "
                "configuration."
            )

    def _decode_variables(
        self,
        variables: str,
    ) -> dict[str, object]:
        raw = variables.strip()

        if not raw:
            return {}

        try:
            decoded = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise GraphQLQueryError(
                f"variables is not valid JSON: {exc}"
            ) from exc

        if not isinstance(decoded, dict):
            raise GraphQLQueryError(
                "variables must be a JSON object, got "
                f"{type(decoded).__name__}"
            )

        return decoded

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render_text(
        self,
        text: str,
        status_code: int = 200,
    ) -> str:

        # Parsed before anything else on purpose: GraphQL reports
        # validation failures with HTTP 400 and a useful errors array,
        # and the model needs that message to correct itself.
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            if status_code >= 400:
                raise GraphQLQueryError(
                    f"HTTP {status_code} from the GraphQL "
                    f"endpoint: {text[:600]}"
                ) from exc

            raise GraphQLQueryError(
                "The endpoint did not return JSON: "
                f"{text[:300]}"
            ) from exc

        errors = payload.get("errors")

        if errors:
            # Field and argument mistakes are the common case and the
            # message is what lets the model correct itself, so it is
            # surfaced verbatim.
            rendered = "\n".join(
                f"- {error.get('message', error)}"
                for error in errors
            )

            raise GraphQLQueryError(
                f"The server rejected the query:\n{rendered}"
            )



        data = payload.get("data")

        if data is None:
            raise GraphQLQueryError(
                "The response contained neither data nor "
                "errors."
            )

        pruned, dropped = _prune_rows(
            data,
            self._policy.max_rows,
        )

        body = json.dumps(
            pruned,
            ensure_ascii=False,
            indent=2,
        )

        if len(body) > self._policy.max_response_chars:
            body = (
                body[: self._policy.max_response_chars]
                + "\n... [truncated by the runtime limit]"
            )

        if dropped:
            body += (
                f"\n\n[note] {dropped} further item(s) "
                "were omitted by the max_rows limit. Narrow the "
                "query or raise graphql.max_rows."
            )

        return body


def _prune_rows(
    value: object,
    budget: int,
    _dropped: list[int] | None = None,
) -> tuple[object, int]:
    """Cap every list in the response.

    A single ``findTags`` can carry thousands of rows, and the model
    cannot use them all: the token cost is real and the extra rows
    crowd out the conversation. Lists are trimmed from the end so
    the ordering the query asked for is preserved.
    """

    if _dropped is None:
        _dropped = [0]

    if isinstance(value, dict):
        pruned: dict[str, object] = {}

        for key, item in value.items():
            pruned[key], _ = _prune_rows(
                item,
                budget,
                _dropped,
            )

        return pruned, _dropped[0]

    if isinstance(value, list):
        kept = value[:budget]

        _dropped[0] += max(0, len(value) - len(kept))

        pruned_list: list[object] = []

        for item in kept:
            child, _ = _prune_rows(
                item,
                budget,
                _dropped,
            )
            pruned_list.append(child)

        return pruned_list, _dropped[0]

    return value, _dropped[0]


# ======================================================================
# Ordering the server refuses to do
# ======================================================================


# Fields that carry a total rather than a row of results.
_COUNT_FIELDS = frozenset({"count"})


class _ScanShape(NamedTuple):
    """What a find query looks like once it is taken apart."""

    root_field: str
    rows_field: str
    document: str


def _scan_shape(document: str) -> _ScanShape:
    """Locate the find field and its list of rows.

    Built by inspecting the AST rather than by matching text, so a
    query with variables, comments or odd whitespace still works.
    """

    parsed = parse(document)

    operations = [
        node
        for node in parsed.definitions
        if isinstance(node, OperationDefinitionNode)
    ]

    if len(operations) != 1:
        raise GraphQLQueryError(
            "order_by needs exactly one operation in the "
            "document."
        )

    selections = list(operations[0].selection_set.selections)

    roots = [
        node
        for node in selections
        if isinstance(node, FieldNode)
    ]

    if len(roots) != 1:
        raise GraphQLQueryError(
            "order_by needs exactly one top-level field, so the "
            "rows to order can be identified."
        )

    root = roots[0]

    if root.selection_set is None:
        raise GraphQLQueryError(
            "order_by needs a selection set on "
            f"'{root.name.value}'."
        )

    rows = [
        node
        for node in root.selection_set.selections
        if isinstance(node, FieldNode)
        and node.name.value not in _COUNT_FIELDS
    ]

    if len(rows) != 1:
        raise GraphQLQueryError(
            "order_by needs exactly one list field under "
            f"'{root.name.value}', such as 'tags' or 'scenes'; "
            f"found {len(rows)}."
        )

    return _ScanShape(
        root_field=root.name.value,
        rows_field=rows[0].name.value,
        document=document,
    )


def _normalize_document(document: str) -> str:
    """Return a parseable document, repairing an omitted outer brace.

    Raises with graphql-core's own message when the document is broken
    in a way wrapping cannot fix, because that message names the line
    and is what lets the model correct itself.
    """

    candidate = document.strip()

    try:
        parse(candidate)

        return candidate
    except Exception as first:
        repaired = _repair_document(candidate)

        if repaired is None:
            raise GraphQLQueryError(
                f"The query is not valid GraphQL: {first}"
            ) from first

        try:
            parse(repaired)
        except Exception:
            raise GraphQLQueryError(
                f"The query is not valid GraphQL: {first}"
            ) from first

        return repaired


def _repair_document(document: str) -> str | None:
    """Restore an omitted outer brace, and nothing else.

    A selection set is the shorthand for a query, and models
    routinely write ``findTags(...) { ... }`` without the leading
    brace. That single character is the whole difference between a
    working query and "Syntax Error: Unexpected Name 'findTags'".

    Narrow on purpose: it only rescues a document that fails to parse
    on its own and becomes valid when wrapped, so it cannot change
    the meaning of anything that was already accepted.
    """

    candidate = document.strip()

    if not candidate:
        return None

    if candidate.startswith("{"):
        # A document that already opens with a brace has a real
        # syntax error somewhere else; wrapping it would hide it.
        return None

    return "{ " + candidate + " }"


def _sort_key(value: object) -> tuple[int, object]:
    """Make mixed and missing values comparable.

    Tags in a real library hold strings, numbers and nulls together.
    Sorting them directly raises, and a missing count must land at the
    bottom of a DESC rather than at the top, so nulls are a separate
    rank that never competes with a real value.
    """

    if value is None:
        return (0, 0)

    if isinstance(value, bool):
        return (1, int(value))

    if isinstance(value, (int, float)):
        return (1, value)

    return (2, str(value).casefold())


def _rebuild_scan_query(
    document: str,
    *,
    page: int,
    per_page: int,
) -> str:
    """Reissue the caller's find query one page at a time.

    The caller's own filter is preserved -- including any criteria and
    the sort it asked for -- because only the paging changes. Text is
    rewritten through the printer rather than by string surgery, so
    variables and odd formatting survive.
    """

    parsed = parse(document)

    operation = next(
        node
        for node in parsed.definitions
        if isinstance(node, OperationDefinitionNode)
    )

    root = next(
        node
        for node in operation.selection_set.selections
        if isinstance(node, FieldNode)
    )

    arguments = []

    for argument in root.arguments or ():
        if argument.name.value != "filter":
            arguments.append(argument)
            continue

        entries = list(argument.value.fields)

        kept = [
            field
            for field in entries
            if field.name.value
            not in ("page", "per_page")
        ]

        kept.append(
            _field_node("per_page", _int_value(per_page))
        )
        kept.append(_field_node("page", _int_value(page)))

        arguments.append(
            argument.__class__(
                name=argument.name,
                value=ObjectValueNode(fields=tuple(kept)),
            )
        )

    # A find with no filter argument at all still needs paging added.
    if not any(
        argument.name.value == "filter"
        for argument in arguments
    ):
        arguments.append(
            _filter_argument(page, per_page)
        )

    rebuilt = FieldNode(
        alias=root.alias,
        name=root.name,
        arguments=tuple(arguments),
        directives=root.directives,
        selection_set=root.selection_set,
    )

    new_operation = OperationDefinitionNode(
        operation=operation.operation,
        name=operation.name,
        variable_definitions=operation.variable_definitions,
        directives=operation.directives,
        selection_set=SelectionSetNode(
            selections=(rebuilt,)
        ),
    )

    return print_ast(
        DocumentNode(definitions=(new_operation,))
    )


def _field_node(name: str, value: object) -> ObjectFieldNode:
    return ObjectFieldNode(
        name=NameNode(value=name),
        value=value,
    )


def _filter_argument(
    page: int,
    per_page: int,
) -> ObjectFieldNode:
    return ObjectFieldNode(
        name=NameNode(value="filter"),
        value=ObjectValueNode(
            fields=(
                _field_node("per_page", _int_value(per_page)),
                _field_node("page", _int_value(page)),
            )
        ),
    )


def _int_value(value: int) -> IntValueNode:
    return IntValueNode(value=str(value))


def _requested_window(
    document: str,
) -> tuple[int, int]:
    """The page and size the caller's own query asked for."""

    parsed = parse(document)

    operation = next(
        node
        for node in parsed.definitions
        if isinstance(node, OperationDefinitionNode)
    )

    root = next(
        node
        for node in operation.selection_set.selections
        if isinstance(node, FieldNode)
    )

    page = 1
    per_page = 100

    for argument in root.arguments or ():
        if argument.name.value != "filter":
            continue

        if not isinstance(argument.value, ObjectValueNode):
            continue

        for field in argument.value.fields:
            raw = getattr(field.value, "value", None)

            if field.name.value == "page" and raw:
                page = int(raw)

            if field.name.value == "per_page" and raw:
                per_page = int(raw)

    return max(1, page), max(1, per_page)


def resolve_api_key(
    api_key: str,
    env_file: str,
) -> str:
    """Find the API key without storing a second copy of it.

    Credentials belong in an env file rather than in config.toml,
    which is readable by anything that can read the home directory.
    The MCP server config already keeps one, so pointing at it avoids
    duplicating the secret.
    """

    if api_key.strip():
        return api_key.strip()

    path = env_file.strip()

    if not path:
        return ""

    candidates = (
        "STASH_API_KEY",
        "GRAPHQL_API_KEY",
        "API_KEY",
    )

    try:
        lines = Path(path).read_text(
            encoding="utf-8"
        ).splitlines()
    except OSError:
        return ""

    for line in lines:
        stripped = line.strip()

        if not stripped or stripped.startswith("#"):
            continue

        if "=" not in stripped:
            continue

        name, _, value = stripped.partition("=")

        if name.strip().upper() in candidates:
            candidate = value.strip().strip('"').strip("'")

            if candidate:
                return candidate

    return ""


# ======================================================================
# Teaching material
# ======================================================================

# What the model actually needs to know about this schema, including
# the parts that cannot be discovered by reading the schema: which
# sorts are accepted, and that the filter arguments are split in two.
GRAPHQL_GUIDANCE = """\
Stash GraphQL, verified against this server.

Shape of a find query. Entity criteria and paging are separate \
arguments, and putting one where the other belongs is rejected. Every \
entity argument is named after the entity, singular, plus _filter:

  findTags(tag_filter: TagFilterType, filter: FindFilterType, ids: [ID])
  findPerformers(performer_filter: PerformerFilterType, filter: FindFilterType, ...)
  findScenes(scene_filter: SceneFilterType, filter: FindFilterType, ...)
  findImages(image_filter: ImageFilterType, filter: FindFilterType, ...)
  findGalleries(gallery_filter: GalleryFilterType, filter: FindFilterType, ...)
  findStudios(studio_filter: StudioFilterType, filter: FindFilterType, ...)
  findGroups(group_filter: GroupFilterType, filter: FindFilterType, ...)

  <entity>_filter -> the entity's own fields, plus AND/OR/NOT
  filter          -> q, page, per_page, sort, direction. Nothing else.

READ THE ERROR WHEN ONE COMES BACK. A rejection that says "not defined \
by type FindFilterType" means the criteria were put in filter instead \
of the entity argument. The criterion named in the error is almost \
always real; the argument it was written in is wrong. For example, \
"Field name is not defined by type FindFilterType" on a performer \
search means findPerformers(performer_filter: {name: ...}), not \
findPerformers(filter: {name: ...}).

Finding a performer by name, both ways:

  findPerformers(performer_filter: { name: { value: "X", modifier: EQUALS } })
  findPerformers(filter: { q: "X" })     # also matches aliases

Comparisons on entity fields use value and modifier, same as below. \
Filtering a scene by tag is scene_filter: { tags: [...] }, and \
AND/OR/NOT belong inside the entity argument, never in filter.

Absent fields. is_missing is a plain string, not a filter object, and \
lives in tag_filter. This finds tags that have no description at all:

  { findTags(tag_filter: { is_missing: "description" },
              filter: { per_page: 10 }) { count tags { id name } } }

Combine with AND, OR and NOT inside tag_filter. They take ONE nested \
object, not a list. Comparisons use value and modifier, and value is \
required even for IS_NULL:

  tag_filter: { description: { value: "", modifier: NOT_NULL } }
  tag_filter: { scene_count: { value: 5, value2: 20, modifier: BETWEEN } }
  tag_filter: { AND: { favorite: true,
                      scene_count: { value: 3, modifier: GREATER_THAN } } }

Modifier names are exact: GREATER_THAN and LESS_THAN, not GTE or \
LTE, and there is no CONTAINS -- use MATCHES_REGEX for a substring. \
Full list: EQUALS, NOT_EQUALS, GREATER_THAN, LESS_THAN, IS_NULL, \
NOT_NULL, INCLUDES_ALL, INCLUDES, EXCLUDES, MATCHES_REGEX, \
NOT_MATCHES_REGEX, BETWEEN, NOT_BETWEEN.

Sorting. direction is the SortDirectionEnum, ASC or DESC. sort is a \
plain string and the server rejects unknown values at runtime.

  filter: { sort: "name", direction: DESC }

Valid sort values for findTags: name, created_at, updated_at, id. \
scene_count, description and favorite are NOT accepted and will fail \
with "invalid sort", and no query shape gets around that.

To rank by one of those fields anyway, use the order_by argument \
instead of sort. The tool then reads every matching row and sorts \
here. The document itself is:

  { findTags(tag_filter: { is_missing: "description" },
              filter: { per_page: 10 })
    { count tags { id name scene_count } } }

sent as the query argument, with the two ordering arguments beside it:

  query    = the document above, verbatim and complete
  order_by = "scene_count"
  order    = "DESC"

The document must start with the opening brace. If it does not, the \
tool reports a syntax error and will repair it once -- but write it \
correctly first.

Read this before sorting by eye. filter.per_page returns one page \
ordered by name, so a page of results is not a page of the highest \
counts -- it is whichever rows the alphabet happens to put there. \
Ordering fifty rows inside a reasoning trace drops ties and returns \
something that looks right and is not. order_by reads every match \
before it ranks, which is why it can be trusted.

order_by needs the field present in your selection set, so select it. \
Leave order_by empty when the server can already do the sorting.

Discovering anything not listed here:

  { __type(name: "TagFilterType") { inputFields { name } } }
  { __schema { queryType { fields { name args { name } } } } }

Read-only: mutations are refused before the request is sent. For \
writes use the Stash MCP tools instead.\
"""


def create_stash_graphql_tool(
    client: StashGraphQLClient,
) -> Tool:
    async def handler(
        arguments: StashGraphQLInput,
        context: ToolContext,
    ) -> str:
        del context

        return await client.execute(
            arguments.query,
            arguments.variables,
            order_by=arguments.order_by,
            order=arguments.order,
        )

    return Tool(
        name="stash_graphql",
        description=(
            "Run a read-only GraphQL query against the Stash "
            "database. Use it when the fixed Stash tools cannot "
            "express what you need: filtering on absent fields, "
            "combining criteria with AND/OR/NOT, or sorting. "
            "Returns raw JSON data. Writes are refused.\\n\\n"
            + GRAPHQL_GUIDANCE
        ),
        input_type=StashGraphQLInput,
        handler=handler,
        policy=ToolPolicy(
            permissions=frozenset({
                "graphql.execute",
            }),
            timeout=client.policy.timeout + 10.0,
            max_output_size=120_000,
        ),
    )
from __future__ import annotations

import json
from dataclasses import dataclass

from agent_workflow.core.entities.models.graphql_query import (
    GraphQLQueryError,
    StashGraphQLClient,
)
from agent_workflow.core.entities.models.tool import (
    Tool,
    ToolContext,
    ToolPolicy,
)


class StashTagsError(RuntimeError):
    """The tag list could not be produced."""


@dataclass(slots=True, frozen=True)
class ListTagsInput:
    search: str = ""
    """Case-insensitive substring of the tag name. Empty lists all."""

    page: int = 1
    """1-based, and only meaningful together with per_page."""

    per_page: int = 0
    """0 means "everything that fits", in which case page is unused."""

    ascending: bool = True
    """Names sort A-Z by default; False gives Z-A."""


@dataclass(slots=True, frozen=True)
class TagListPolicy:
    """Trusted runtime limits.

    The whole library has to fit in one tool result, so the ceiling
    is what protects the context rather than the page size.
    """

    max_rows: int = 5_000

    page_size: int = 250

    def __post_init__(self) -> None:
        if self.max_rows < 1:
            raise ValueError("max_rows must be >= 1")

        if self.page_size < 1:
            raise ValueError("page_size must be >= 1")


@dataclass(slots=True, frozen=True)
class TagRef:
    """One tag, identified the way the write tools need it."""

    id: str
    name: str

    def render(self) -> str:
        # Tab separated: one line per tag, no quoting to undo, and
        # roughly half the tokens of the JSON equivalent.
        return f"{self.id}\t{self.name}"


class StashTagLister:
    """Every tag in the library as an id and a name.

    The other tag tools return full records: aliases, descriptions,
    scene counts, parents and children. That is the wrong shape for
    the common bulk task, where all you need is the id to hand to
    `mcp_stash_update_tag_description` and the name to decide which
    ones to act on.
    """

    def __init__(
        self,
        client: StashGraphQLClient,
        *,
        policy: TagListPolicy | None = None,
    ) -> None:
        self._client = client
        self._policy = policy or TagListPolicy()

    @property
    def policy(self) -> TagListPolicy:
        return self._policy

    @property
    def client(self) -> StashGraphQLClient:
        return self._client

    async def list_tags(
        self,
        search: str = "",
        *,
        page: int = 1,
        per_page: int = 0,
        ascending: bool = True,
    ) -> tuple[list[TagRef], int]:
        """Return the requested slice and the total number of matches."""

        needle = search.strip()
        direction = "ASC" if ascending else "DESC"

        limit = self._policy.max_rows

        if per_page > 0:
            limit = min(per_page, limit)

        # Enough rows to reach the requested page, not just one
        # page's worth: slicing page 2 out of the first page yields
        # nothing at all.
        needed = min(limit * max(1, page), self._policy.max_rows)

        refs, total = await self._collect(
            needle=needle,
            direction=direction,
            limit=needed,
        )

        if page > 1:
            start = (page - 1) * limit

            refs = refs[start : start + limit]

        return refs, total

    async def _collect(
        self,
        *,
        needle: str,
        direction: str,
        limit: int,
    ) -> tuple[list[TagRef], int]:
        rows: list[dict] = []
        total = 0

        page_size = min(self._policy.page_size, self._policy.max_rows)
        page = 1

        while len(rows) < limit:
            payload = await self._fetch_page(
                needle=needle,
                direction=direction,
                page=page,
                per_page=page_size,
            )

            block = payload.get("findTags") or {}

            if page == 1:
                total = int(block.get("count") or 0)

            batch = block.get("tags") or []

            if not batch:
                break

            rows.extend(batch)

            if len(batch) < page_size:
                break

            page += 1

        refs = [
            TagRef(
                id=str(row.get("id", "")),
                name=str(row.get("name", "")),
            )
            for row in rows[:limit]
            if row.get("id") is not None
        ]

        return refs, total

    async def _fetch_page(
        self,
        *,
        needle: str,
        direction: str,
        page: int,
        per_page: int,
    ) -> dict:
        criteria = ""

        if needle:
            criteria = (
                'tag_filter: { name: { value: "'
                + _match_pattern(needle)
                + '", modifier: MATCHES_REGEX } }, '
            )

        query = (
            "{ findTags("
            + criteria
            + "filter: { per_page: "
            + str(per_page)
            + ", page: "
            + str(page)
            + ', sort: "name", direction: '
            + direction
            + " }) { count tags { id name } } }"
        )

        try:
            raw = await self._client.execute(query)
        except GraphQLQueryError as exc:
            raise StashTagsError(str(exc)) from exc

        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise StashTagsError(
                "Stash returned something that is not JSON"
            ) from exc

        # execute() already unwraps the GraphQL "data" envelope, so
        # the usual shape is {findTags: ...}. Tolerating the wrapped
        # form too costs one lookup and removes a silent empty list
        # if that ever changes.
        if "data" in payload and isinstance(payload["data"], dict):
            payload = payload["data"]

        return payload


# Regex metacharacters. The search is a plain substring, but the
# server only offers MATCHES_REGEX for one, so a name containing a
# dot or a bracket must not turn into a pattern that matches more
# than the user asked for.
_REGEX_SPECIALS = set("\\^$.|?*+()[]{}")


def _graphql_string(value: str) -> str:
    """Escape for a GraphQL string literal.

    Only these escapes are legal inside one, so a regex escape has to
    arrive doubled: ``\\(`` in the document is ``\\\\(`` in the
    pattern. Getting this wrong produces "Invalid character escape"
    rather than anything pointing at the cause.
    """

    return (
        value.replace("\\", "\\\\").replace('"', '\\"')
    )


def _match_pattern(needle: str) -> str:
    """A case-insensitive literal pattern for MATCHES_REGEX.

    Stash stores names capitalised, so a case-sensitive match would
    make "anal" miss the tag named "Anal" -- the search would look
    broken for exactly the input people try first.
    """

    escaped = "".join(
        ("\\" + char) if char in _REGEX_SPECIALS else char
        for char in needle
    )

    return _graphql_string(f"(?i){escaped}")


def format_tag_list(
    refs: list[TagRef],
    total: int,
    *,
    search: str = "",
) -> str:
    """Render the list with a header the model can act on.

    The header matters more than it looks: without a total, "50 tags"
    and "all 981" look identical, and the model concludes it has seen
    everything.
    """

    scope = f'matching "{search}"' if search else "in the library"

    lines = [
        f"Stash tags {scope}: {len(refs)} shown of {total} total.",
        "One tag per line, id then name.",
        "",
    ]

    if not refs:
        lines.append("(none)")
        return "\n".join(lines)

    lines.extend(ref.render() for ref in refs)

    if len(refs) < total:
        remaining = total - len(refs)

        lines.extend(
            [
                "",
                f"[note] {remaining} more not shown. Call again with "
                "page=2, or narrow with search.",
            ]
        )

    return "\n".join(lines)


def create_stash_list_tags_tool(
    lister: StashTagLister,
) -> Tool:
    async def handler(
        arguments: ListTagsInput,
        context: ToolContext,
    ) -> str:
        del context

        refs, total = await lister.list_tags(
            arguments.search,
            page=arguments.page,
            per_page=arguments.per_page,
            ascending=arguments.ascending,
        )

        return format_tag_list(
            refs,
            total,
            search=arguments.search,
        )

    return Tool(
        name="stash_list_tags",
        description=(
            "List the tags in the Stash library as an id and a name, "
            "one per line. Use it when you need the tag ids to pass "
            "to another tool, such as mcp_stash_update_tag_description "
            "or mcp_stash_get_tag, or when you need to see which tags "
            "exist without reading their descriptions and aliases. "
            "The header states how many tags matched in total, so an "
            "empty-looking result can be told apart from a truncated "
            "one. Narrow with search, or page, when the library is "
            "large."
        ),
        input_type=ListTagsInput,
        handler=handler,
        policy=ToolPolicy(
            permissions=frozenset({
                "graphql.execute",
            }),
            timeout=lister.client.policy.timeout
            * max(
                1,
                lister.policy.max_rows // lister.policy.page_size,
            )
            + 15.0,
            max_output_size=200_000,
        ),
    )
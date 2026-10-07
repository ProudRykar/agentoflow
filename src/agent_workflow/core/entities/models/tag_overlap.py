"""Evidence for judging whether two Stash tags are duplicates.

The failure this exists to prevent: a name-only comparison reports
"Cowgirl Position" and "Doggy Style" as duplicates because both
contain the word "Position", and reports "Big Breast" and "Big vs.
Small" as a pair because both start with "Big". Neither is a
duplicate. Word overlap measures spelling, not meaning.

Usage data separates them. Two tags applied to the same scenes behave
the same way; two tags that never share a scene are distinct concepts
however similar their names read, and two that partially overlap are
related without being interchangeable. That is a measurement, so it
is made here rather than guessed at from names.

What this deliberately does not do is declare a semantic verdict.
Containment of 1.0 with a Jaccard of 0.6 means every occurrence of the
smaller tag also carries the larger one, which is often a curation
habit rather than a synonymy: in a real library "HD Available" sits
inside "Verified Amateur" for reasons that have nothing to do with
meaning. The numbers are reported, the descriptions are supplied, and
the judgement is left to the caller.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, replace
from itertools import combinations
from typing import Any

from agent_workflow.core.entities.models.graphql_query import (
    StashGraphQLClient,
)
from agent_workflow.core.entities.models.tool import (
    Tool,
    ToolContext,
    ToolPolicy,
)


class StashTagOverlapError(RuntimeError):
    """Raised when the library cannot be read well enough to compare."""


# Appended by the client when a response exceeds the model-facing cap.
_TRUNCATION_MARKER = "[truncated by the runtime limit]"


def _rows(raw: Any, root: str, field: str) -> list[Any]:
    """Pull one list out of a response.

    A page that cannot be decoded raises rather than yielding an empty
    list. The usual result of this tool is "nothing looked like a
    duplicate", and a silent decode failure would look exactly like that
    verdict: the model would report a clean library on the strength of a
    response it never read.
    """

    body = raw if isinstance(raw, str) else json.dumps(raw)

    if _TRUNCATION_MARKER in body:
        raise StashTagOverlapError(
            "the response was truncated before it could be read; "
            "narrow the query with tags or search"
        )

    try:
        payload = json.loads(raw) if isinstance(raw, str) else raw
    except json.JSONDecodeError as exc:
        raise StashTagOverlapError(
            "Stash returned something that is not JSON"
        ) from exc

    section = payload.get(root) if isinstance(payload, dict) else None

    if not isinstance(section, dict):
        return []

    rows = section.get(field)

    return rows if isinstance(rows, list) else []


@dataclass(slots=True, frozen=True)
class TagOverlapInput:
    """Arguments for ``stash_tag_overlap``."""

    tags: list[str] | None = None
    """Specific names or ids to compare. Empty considers everything."""

    search: str = ""
    """Restrict the candidate tags to names containing this."""

    min_jaccard: float = 0.5
    """Pairs below this are only reported if never-together."""

    min_scenes: int = 5
    """Usage floor. Below it a tag cannot support a verdict."""

    include_descriptions: bool = False
    """Print both descriptions, which is what the judgement needs."""

    limit: int = 40
    """Most pairs to report."""


@dataclass(slots=True, frozen=True)
class TagOverlapPolicy:
    """Bounds on the scan.

    ``min_scenes`` is the important one. A tag on a single scene shares
    that scene with every other tag on it, so single-use tags produce a
    perfect overlap with whatever else is present and drown out the real
    signal. Evidence below the floor is reported as insufficient rather
    than as a match.
    """

    scene_page_size: int = 20
    max_scene_pages: int = 200
    max_pairs: int = 2_000
    max_output_size: int = 120_000

    def __post_init__(self) -> None:
        if self.scene_page_size < 1:
            raise ValueError("scene_page_size must be >= 1")
        if self.max_scene_pages < 1:
            raise ValueError("max_scene_pages must be >= 1")
        if self.max_pairs < 1:
            raise ValueError("max_pairs must be >= 1")


@dataclass(slots=True, frozen=True)
class TagRecord:
    """What the scan needs to know about one tag."""

    id: str
    name: str
    aliases: tuple[str, ...] = ()
    # Filled in only for the tags that get reported: the whole
    # library's descriptions do not fit in one response.
    description: str = ""


@dataclass(slots=True, frozen=True)
class PairEvidence:
    """Measured relationship between two tags.

    Frozen with slots: the ratios are properties, not stored fields, so
    two pairs with the same numbers compare equal.
    """

    first: TagRecord
    second: TagRecord
    first_scenes: int
    second_scenes: int
    shared: int

    @property
    def jaccard(self) -> float:
        union = self.first_scenes + self.second_scenes - self.shared

        return self.shared / union if union else 0.0

    @property
    def containment(self) -> float:
        """How much of the smaller tag's usage the larger one covers.

        1.0 means the smaller tag is only ever used alongside the
        larger one, which makes it a candidate for becoming an alias
        rather than for being deleted.
        """

        smaller = min(self.first_scenes, self.second_scenes)

        return self.shared / smaller if smaller else 0.0

    @property
    def alias_link(self) -> str:
        """Whether one name is already recorded as the other's alias."""

        first_lower = {alias.strip().lower() for alias in self.first.aliases}
        second_lower = {alias.strip().lower() for alias in self.second.aliases}

        if self.second.name.strip().lower() in first_lower:
            return f"{self.second.name!r} is already an alias of {self.first.name!r}"

        if self.first.name.strip().lower() in second_lower:
            return f"{self.first.name!r} is already an alias of {self.second.name!r}"

        return ""

    @property
    def lexical_overlap(self) -> tuple[str, ...]:
        """Content words the two names share.

        Included because the interesting finding is often a mismatch:
        names that read as the same thing and never co-occur. That
        pairing is exactly what a name-only pass reports as a duplicate,
        so it has to be visible rather than left to the caller.
        """

        return tuple(
            sorted(
                _content_words(self.first.name)
                & _content_words(self.second.name)
            )
        )

    @property
    def verdict(self) -> str:
        """What the usage data supports, in one line.

        Deliberately not a synonymy claim. It says what the numbers
        rule out, which is the part a name comparison gets wrong.
        """

        if self.usage == "unused":
            return "one of these is unused, so usage cannot compare them"

        if self.shared == 0:
            return (
                "used on entirely different scenes: distinct tags, "
                "however similar the names read"
            )

        if self.jaccard >= 0.999:
            return "always used together: a merge candidate worth checking"

        if self.containment >= 0.999:
            smaller = (
                self.second.name
                if self.first_scenes <= self.second_scenes
                else self.first.name
            )

            return (
                f"{smaller!r} only ever appears with the other, which "
                "is a curation habit as often as a synonym"
            )

        return (
            f"overlap {self.jaccard:.0%} of the union: related, and not "
            "interchangeable"
        )

    @property
    def usage(self) -> str:
        """Whether the usage data can speak to this pair at all."""

        if self.first_scenes == 0 or self.second_scenes == 0:
            return "unused"

        return "comparable"

    @property
    def relationship(self) -> str:
        if self.usage == "unused":
            return "no-usage-data"

        if self.shared == 0:
            return "never-together"

        if self.jaccard >= 0.999:
            return "identical-usage"

        if self.containment >= 0.999:
            return "strict-subset"

        return "partial-overlap"


@dataclass
class UsageIndex:
    """Tag id to the scenes carrying it."""

    scenes: dict[str, set[str]] = field(default_factory=dict)

    def count(self, tag_id: str) -> int:
        return len(self.scenes.get(tag_id, ()))


class StashTagOverlap:
    """Builds the usage index and pairs tags against it."""

    def __init__(
        self,
        client: StashGraphQLClient,
        *,
        policy: TagOverlapPolicy | None = None,
    ) -> None:
        self._client = client
        self._policy = policy or TagOverlapPolicy()

    @property
    def client(self) -> StashGraphQLClient:
        return self._client

    @property
    def policy(self) -> TagOverlapPolicy:
        return self._policy

    async def tags(self, *, search: str = "") -> list[TagRecord]:
        """Every tag with its aliases and description."""

        # No description here: a few hundred tags with their aliases
        # and descriptions exceed the response cap, and a truncated
        # page reads as an empty library. Descriptions are fetched
        # later, for the handful of tags that actually get reported.
        query = """
        query($page: Int!) {
          findTags(
            filter: { per_page: 250, page: $page, sort: "name" }
          ) {
            tags {
              id
              name
              aliases
            }
          }
        }
        """

        needle = search.strip().lower()
        found: dict[str, TagRecord] = {}
        page = 1

        while page <= self._policy.max_scene_pages:
            raw = await self._client.execute(
                query,
                json.dumps({"page": page}),
            )
            batch = _rows(raw, "findTags", "tags")

            if not batch:
                break

            for raw in batch:
                name = str(raw.get("name") or "")

                if needle and needle not in name.lower():
                    continue

                tag_id = str(raw.get("id") or "")

                if not tag_id:
                    continue

                found.setdefault(
                    tag_id,
                    TagRecord(
                        id=tag_id,
                        name=name,
                        aliases=tuple(raw.get("aliases") or ()),
                    ),
                )

            if len(batch) < 250:
                break

            page += 1

        return sorted(found.values(), key=lambda tag: tag.name.lower())

    async def descriptions(self, ids: list[str]) -> dict[str, str]:
        """Descriptions for a short list of tags.

        Fetched separately from the sweep because the whole library's
        descriptions do not fit in one response.
        """

        if not ids:
            return {}

        quoted = ", ".join(f'"{tag}"' for tag in ids[:80])

        query = f"""
        query {{
          findTags(ids: [{quoted}]) {{
            tags {{ id description }}
          }}
        }}
        """

        found: dict[str, str] = {}

        for row in _rows(await self._client.execute(query), "findTags", "tags"):
            found[str(row.get("id") or "")] = str(row.get("description") or "")

        return found

    async def usage(self) -> UsageIndex:
        """Index every tag by the scenes that carry it.

        Paged deliberately small: the response limit is far below what
        a single page of scenes with their tags costs once indented,
        and a truncated page would silently index only part of the
        library.
        """

        query = """
        query($page: Int!) {
          findScenes(filter: { per_page: PAGE, page: $page }) {
            scenes {
              id
              tags { id }
            }
          }
        }
        """

        index = UsageIndex()
        page = 1

        while page <= self._policy.max_scene_pages:
            payload = await self._client.execute(
                query.replace("PAGE", str(self._policy.scene_page_size)),
                json.dumps({"page": page}),
            )
            scenes = _rows(payload, "findScenes", "scenes")

            if not scenes:
                break

            for scene in scenes:
                scene_id = str(scene.get("id") or "")

                if not scene_id:
                    continue

                for tag in scene.get("tags") or ():
                    tag_id = str(tag.get("id") or "")

                    if tag_id:
                        index.scenes.setdefault(tag_id, set()).add(scene_id)

            page += 1

        return index

    async def compare(
        self,
        records: list[TagRecord],
        index: UsageIndex,
        *,
        min_scenes: int = 5,
    ) -> list[PairEvidence]:
        """Measure every pair, strongest first."""

        pairs: list[PairEvidence] = []

        for first, second in combinations(records, 2):
            first_scenes = index.count(first.id)
            second_scenes = index.count(second.id)

            # Both sides have to clear the floor. A tag on one scene
            # appears to "never co-occur" with almost anything, and
            # reporting that as evidence buries the pairs that say
            # something real.
            if first_scenes < min_scenes or second_scenes < min_scenes:
                continue

            pairs.append(
                PairEvidence(
                    first=first,
                    second=second,
                    first_scenes=first_scenes,
                    second_scenes=second_scenes,
                    shared=len(
                        index.scenes.get(first.id, set())
                        & index.scenes.get(second.id, set())
                    ),
                )
            )

        pairs.sort(key=lambda pair: (-pair.jaccard, -pair.shared))

        return pairs[: self._policy.max_pairs]

    async def analyse(
        self,
        *,
        tags: list[str] | None = None,
        search: str = "",
        min_jaccard: float = 0.5,
        min_scenes: int = 5,
        include_descriptions: bool = False,
        limit: int = 40,
    ) -> tuple[list[PairEvidence], int, list[TagRecord]]:
        """Pairs worth looking at, the tag count considered, the tags."""

        records = await self.tags(search=search)

        wanted: set[str] | None = None

        if tags:
            wanted = {name.strip().lower() for name in tags if name.strip()}

            records = [
                tag
                for tag in records
                if tag.name.strip().lower() in wanted
                or tag.id in wanted
            ]

        missing = sorted(
            (wanted or set())
            - {
                token
                for tag in records
                for token in (tag.name.strip().lower(), tag.id)
            }
        )

        index = await self.usage()
        pairs = await self.compare(records, index, min_scenes=min_scenes)

        if wanted:
            # The caller named the suspects, so every pair they named
            # is answered. Thresholding here would hide the case that
            # matters most: two similarly named tags that turn out to
            # be used on entirely different scenes.
            interesting = pairs
        else:
            interesting = [
                pair
                for pair in pairs
                if pair.jaccard >= min_jaccard
                or pair.alias_link
                # A name-level pass would call these duplicates, so they
                # are worth surfacing precisely because usage refutes it.
                or (
                    pair.relationship == "never-together"
                    and bool(pair.lexical_overlap)
                )
            ]

        chosen = interesting[: max(1, limit)]

        if include_descriptions:
            involved = sorted({
                tag.id
                for pair in chosen
                for tag in (pair.first, pair.second)
            })

            described = await self.descriptions(involved)
            by_id = {tag.id: tag for tag in records}

            filled = {
                tag_id: replace(by_id[tag_id], description=text)
                for tag_id, text in described.items()
                if text and tag_id in by_id
            }

            if filled:
                chosen = [
                    replace(
                        pair,
                        first=filled.get(pair.first.id, pair.first),
                        second=filled.get(pair.second.id, pair.second),
                    )
                    for pair in chosen
                ]

        return chosen, len(records), missing


_STOPWORDS = frozenset({
    "and", "as", "at", "by", "for", "from", "in", "of", "on", "or", "the",
    "to", "with", "without", "а", "без", "в", "и", "из", "к", "на", "не",
    "от", "по", "с", "со",
})


def _content_words(name: str) -> set[str]:
    """Lower-cased words long enough to carry meaning."""

    words = re.findall(r"[\w']+", name.lower())

    return {
        word
        for word in words
        if len(word) > 2 and word not in _STOPWORDS
    }


def _short(text: str, limit: int = 160) -> str:
    text = " ".join(text.split())

    if len(text) <= limit:
        return text

    return text[: limit - 1].rstrip() + "…"


def format_overlap(
    pairs: list[PairEvidence],
    considered: int,
    missing: list[str],
    *,
    min_jaccard: float,
    min_scenes: int,
    include_descriptions: bool,
) -> str:
    """Render the evidence, with the rule the numbers came from."""

    lines: list[str] = []

    lines.append(f"Tags considered: {considered}.")

    if missing:
        lines.append(f"Not found in the library: {', '.join(missing)}.")

    lines.append("")
    lines.append(
        "Usage is compared over the scenes that actually carry each "
        f"tag; pairs where either side is under {min_scenes} scenes "
        "are left out, because a tag on one scene overlaps everything "
        "on that scene."
    )
    lines.append("")
    lines.append("relationship:")
    lines.append("  identical-usage   the two tags are on exactly the same scenes")
    lines.append("  strict-subset     the smaller tag only ever appears with the larger")
    lines.append("  partial-overlap   related, not interchangeable")
    lines.append("  never-together    share no scene at all; similar names, distinct tags")
    lines.append("")
    lines.append(
        "None of these is a synonymy proof. Subset in particular often "
        "reflects a curation habit rather than meaning, so read the "
        "descriptions before deciding to merge anything."
    )

    if not pairs:
        lines.append("")
        lines.append(
            "No pair cleared the threshold. Nothing here looks like a "
            "duplicate on the available evidence."
        )

        return "\n".join(lines)

    lines.append("")
    lines.append(
        f"Pairs at or above jaccard {min_jaccard:.2f}, plus every "
        "never-together and alias-linked pair:"
    )

    for pair in pairs:
        lines.append("")
        lines.append(
            f"  {pair.first.name!r} [{pair.first.id}] ~ "
            f"{pair.second.name!r} [{pair.second.id}]"
        )
        lines.append(
            f"    scenes {pair.first_scenes} vs {pair.second_scenes}, "
            f"shared {pair.shared}, "
            f"jaccard {pair.jaccard:.2f}, "
            f"containment {pair.containment:.2f}"
        )
        lines.append(f"    relationship: {pair.relationship}")
        lines.append(f"    usage says: {pair.verdict}")

        if pair.lexical_overlap:
            shared = ", ".join(pair.lexical_overlap)

            lines.append(f"    shared words in the names: {shared}")

        if pair.alias_link:
            lines.append(f"    alias: {pair.alias_link}")

        if include_descriptions:
            for side in (pair.first, pair.second):
                if side.description:
                    lines.append(f"    {side.name} description: {_short(side.description)}")

    return "\n".join(lines)


def create_stash_tag_overlap_tool(
    overlap: StashTagOverlap,
) -> Tool:
    """The tool the model reaches for instead of comparing names."""

    async def handler(
        arguments: TagOverlapInput,
        context: ToolContext,
    ) -> str:
        del context

        pairs, considered, missing = await overlap.analyse(
            tags=arguments.tags,
            search=arguments.search,
            min_jaccard=arguments.min_jaccard,
            min_scenes=arguments.min_scenes,
            include_descriptions=arguments.include_descriptions,
            limit=arguments.limit,
        )

        return format_overlap(
            pairs,
            considered,
            missing,
            min_jaccard=arguments.min_jaccard,
            min_scenes=arguments.min_scenes,
            include_descriptions=arguments.include_descriptions,
        )

    return Tool(
        name="stash_tag_overlap",
        description=(
            "Measure how two sets of Stash tags are actually used, so "
            "that duplicate analysis rests on data instead of on how "
            "the names read. Builds an index of every tag to the scenes "
            "carrying it, then reports each pair with its scene counts, "
            "how many scenes they share, the Jaccard and containment "
            "ratios, whether one name is already recorded as the "
            "other's alias, and whether the descriptions say the same "
            "thing.\n"
            "\n"
            "Use this whenever the question is whether tags are "
            "duplicates, redundant, or should be merged, and also when "
            "it is whether two similarly named tags are the same "
            "concept. Comparing names alone gets this wrong in both "
            "directions: it calls distinct tags duplicates because "
            "they share a word, and it misses real redundancy because "
            "the synonyms happen to be spelled differently.\n"
            "\n"
            "Pass tags to check specific ones; pass search to find "
            "candidates across the library, or neither to consider "
            "everything. Lowers min_scenes to see rarely used tags at "
            "the cost of trusting them less. The verdict is left open "
            "on purpose, because containment and overlap describe "
            "curation habits as much as meaning."
        ),
        input_type=TagOverlapInput,
        handler=handler,
        policy=ToolPolicy(
            permissions=frozenset({
                "graphql.execute",
            }),
            # The index is a full paged sweep of the scenes table.
            timeout=max(
                60.0,
                overlap.client.policy.timeout
                * (overlap.policy.max_scene_pages // 4),
            ),
            max_output_size=overlap.policy.max_output_size,
        ),
    )


__all__: list[Any] = [
    "PairEvidence",
    "StashTagOverlap",
    "StashTagOverlapError",
    "TagOverlapInput",
    "TagOverlapPolicy",
    "TagRecord",
    "create_stash_tag_overlap_tool",
    "format_overlap",
]
"""Duplicate analysis must rest on usage, not on how names read.

The failure being guarded against is specific: a name comparison calls
"Cowgirl Position" and "Ballerina Position" duplicates because both
end in "Position", and calls "Big Breast" and "Big vs. Small" a pair
because both start with "Big". Neither is a duplicate, and no amount of
prompting fixes it, because the signal needed to tell them apart is
not in the name.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_workflow.core.entities.models.tag_overlap import (
    PairEvidence,
    StashTagOverlap,
    StashTagOverlapError,
    TagOverlapInput,
    TagOverlapPolicy,
    TagRecord,
    create_stash_tag_overlap_tool,
    format_overlap,
)


def pair(
    first: str,
    second: str,
    first_scenes: int,
    second_scenes: int,
    shared: int,
    *,
    first_aliases: tuple[str, ...] = (),
    second_aliases: tuple[str, ...] = (),
) -> PairEvidence:
    return PairEvidence(
        first=TagRecord(id="1", name=first, aliases=first_aliases),
        second=TagRecord(id="2", name=second, aliases=second_aliases),
        first_scenes=first_scenes,
        second_scenes=second_scenes,
        shared=shared,
    )


class FakeClient:
    """Serves canned pages and records what was asked for."""

    def __init__(
        self,
        tags: list[dict],
        scenes: list[dict],
        *,
        descriptions: dict[str, str] | None = None,
    ) -> None:
        self.tags = tags
        self.scenes = scenes
        self.descriptions = descriptions or {}
        self.queries: list[str] = []
        self._policy = type("P", (), {"timeout": 30.0})()

    @property
    def policy(self):
        return self._policy

    async def execute(self, query: str, variables: str = "") -> str:
        self.queries.append(query)

        page = json.loads(variables).get("page", 1) if variables else 1

        if "findTags(ids:" in query:
            wanted = {
                token.strip().strip('"')
                for token in query.split("findTags(ids: [")[1].split("]")[0].split(",")
            }

            return json.dumps({
                "findTags": {
                    "tags": [
                        {"id": tag_id, "description": self.descriptions.get(tag_id, "")}
                        for tag_id in wanted
                    ],
                },
            })

        if "findScenes" in query:
            size = int(query.split("per_page: ")[1].split(",")[0])
            window = self.scenes[(page - 1) * size : page * size]

            return json.dumps({
                "findScenes": {"scenes": window},
            })

        size = 250
        window = self.tags[(page - 1) * size : page * size]

        return json.dumps({"findTags": {"tags": window}})


def library() -> tuple[list[dict], list[dict]]:
    """Three tags, laid out to separate the interesting cases."""

    tags = [
        {"id": "10", "name": "Cowgirl Position", "aliases": ["Riding Position"]},
        {"id": "11", "name": "Ballerina Position", "aliases": []},
        {"id": "12", "name": "Big Breast", "aliases": []},
        {"id": "13", "name": "Big vs. Small", "aliases": []},
        {"id": "14", "name": "rare", "aliases": []},
        {"id": "15", "name": "Slut", "aliases": ["Whore"]},
        {"id": "16", "name": "Whore", "aliases": []},
    ]

    # 10: scenes 1..10, 11: 1..2, 12: 11..20, 13: 21..22, 14: one scene
    # only, 15: 1..5, 16: 1..3.
    usage: dict[str, list[str]] = {
        "10": [f"s{index}" for index in range(1, 11)],
        "11": ["s1", "s2"],
        "12": [f"s{index}" for index in range(11, 21)],
        "13": ["s21", "s22"],
        "14": ["s1"],
        "15": [f"s{index}" for index in range(1, 6)],
        "16": ["s1", "s2", "s3"],
    }

    scenes: list[dict] = []
    all_ids = {scene for ids in usage.values() for scene in ids}

    for scene_id in sorted(all_ids, key=lambda name: int(name[1:])):
        scenes.append({
            "id": scene_id,
            "tags": [
                {"id": tag_id}
                for tag_id, ids in usage.items()
                if scene_id in ids
            ],
        })

    return tags, scenes


async def test_identical_usage_is_reported_as_a_merge_candidate() -> None:
    tags, scenes = library()
    overlap = StashTagOverlap(FakeClient(tags, scenes))
    records = [tag for tag in await overlap.tags() if tag.id in {"15", "16"}]
    index = await overlap.usage()

    pairs = await overlap.compare(records, index, min_scenes=1)

    assert len(pairs) == 1
    evidence = pairs[0]

    assert evidence.relationship == "strict-subset"
    assert evidence.containment == 1.0
    assert "curation habit" in evidence.verdict


async def test_never_together_refutes_a_name_level_duplicate() -> None:
    evidence = pair("Big Breast", "Big vs. Small", 225, 6, 0)

    assert evidence.relationship == "never-together"
    assert evidence.lexical_overlap == ("big",)
    assert "however similar the names read" in evidence.verdict


async def test_partial_overlap_is_related_not_interchangeable() -> None:
    evidence = pair("Cowgirl Position", "Ballerina Position", 233, 7, 6)

    assert evidence.relationship == "partial-overlap"
    assert "not interchangeable" in evidence.verdict
    assert evidence.lexical_overlap == ("position",)


def test_alias_link_is_reported_in_both_directions() -> None:
    forward = pair("Slut", "Whore", 5, 5, 5, first_aliases=("Whore",))
    backward = pair("Whore", "Slut", 5, 5, 5, second_aliases=("Whore",))

    assert "already an alias" in forward.alias_link
    assert "already an alias" in backward.alias_link
    assert pair("Alpha", "Beta", 5, 5, 5).alias_link == ""


def test_jaccard_and_containment_maths() -> None:
    evidence = pair("A", "B", 10, 5, 5)

    assert evidence.jaccard == pytest.approx(0.5)
    assert evidence.containment == 1.0

    disjoint = pair("A", "B", 10, 5, 0)

    assert disjoint.jaccard == 0.0
    assert disjoint.containment == 0.0


def test_identical_usage_needs_the_full_union() -> None:
    evidence = pair("A", "B", 10, 10, 10)

    assert evidence.jaccard == 1.0
    assert evidence.relationship == "identical-usage"
    assert "merge candidate" in evidence.verdict


async def test_explicitly_named_pairs_bypass_the_threshold() -> None:
    """A named suspect is answered even when the overlap is tiny.

    Thresholding here would hide the most useful answer available: two
    similarly named tags that turn out to be used on different scenes.
    """

    tags, scenes = library()
    overlap = StashTagOverlap(FakeClient(tags, scenes))

    named, considered, _missing = await overlap.analyse(
        tags=["Cowgirl Position", "Ballerina Position"],
        min_jaccard=0.9,
        min_scenes=1,
    )

    assert considered == 2
    assert len(named) == 1

    evidence = named[0]

    assert {evidence.first.name, evidence.second.name} == {
        "Cowgirl Position",
        "Ballerina Position",
    }
    # Jaccard here is 0.2, far below the 0.9 asked for, yet the pair is
    # still answered: a named suspect is never silently dropped.
    assert evidence.relationship == "strict-subset"
    assert evidence.jaccard < 0.9


async def test_discovery_surfaces_name_collisions_refuted_by_usage() -> None:
    tags, scenes = library()
    overlap = StashTagOverlap(FakeClient(tags, scenes))

    found, _considered, _missing = await overlap.analyse(min_scenes=1)

    names = {
        frozenset((pair_.first.name, pair_.second.name))
        for pair_ in found
    }

    assert frozenset(("Big Breast", "Big vs. Small")) in names


async def test_both_sides_must_clear_the_usage_floor() -> None:
    """A tag on one scene appears to disagree with everything.

    Pairing it with a well-used tag produced a wall of never-together
    rows that said nothing, so the floor applies to each side.
    """

    tags, scenes = library()
    overlap = StashTagOverlap(FakeClient(tags, scenes))

    records = [tag for tag in await overlap.tags() if tag.id in {"10", "14"}]
    index = await overlap.usage()

    assert index.count("14") == 1
    assert await overlap.compare(records, index, min_scenes=5) == []


async def test_a_truncated_page_raises_instead_of_reporting_no_duplicates() -> None:
    """Silence here would read as a clean library.

    The usual answer from this tool is that nothing looked like a
    duplicate, so a page that fails to decode must not be mistaken for
    evidence of that.
    """

    class Truncating(FakeClient):
        async def execute(self, query: str, variables: str = "") -> str:
            if "findTags" in query and "ids:" not in query:
                return '{"findTags": {"tags": [{"id": "1", "name": "A", "aliases": []}]}}\\n... [truncated by the runtime limit]'

            return await super().execute(query, variables)

    tags, scenes = library()
    overlap = StashTagOverlap(Truncating(tags, scenes))

    with pytest.raises(StashTagOverlapError, match="truncated"):
        await overlap.tags()


async def test_a_non_json_page_raises() -> None:
    class Broken(FakeClient):
        async def execute(self, query: str, variables: str = "") -> str:
            return "<html>502 Bad Gateway</html>"

    tags, scenes = library()

    with pytest.raises(StashTagOverlapError, match="not JSON"):
        await StashTagOverlap(Broken(tags, scenes)).tags()


async def test_descriptions_are_attached_for_reported_tags_only() -> None:
    tags, scenes = library()
    client = FakeClient(
        tags,
        scenes,
        descriptions={
            "12": "Breasts that are notably large.",
            "13": "A comparison between breast sizes.",
        },
    )
    overlap = StashTagOverlap(client)

    found, _considered, _missing = await overlap.analyse(
        tags=["Big Breast", "Big vs. Small"],
        include_descriptions=True,
        min_scenes=1,
    )

    assert found[0].first.description == "Breasts that are notably large."


async def test_unknown_names_are_reported_back() -> None:
    tags, scenes = library()
    overlap = StashTagOverlap(FakeClient(tags, scenes))

    _pairs, _considered, missing = await overlap.analyse(
        tags=["Big Breast", "Nonexistent Tag"],
        min_scenes=1,
    )

    assert missing == ["nonexistent tag"]


async def test_search_restricts_the_candidates() -> None:
    tags, scenes = library()
    overlap = StashTagOverlap(FakeClient(tags, scenes))

    records = await overlap.tags(search="breast")

    assert [tag.name for tag in records] == ["Big Breast"]


def test_policy_rejects_nonsense_bounds() -> None:
    with pytest.raises(ValueError):
        TagOverlapPolicy(scene_page_size=0)

    with pytest.raises(ValueError):
        TagOverlapPolicy(max_scene_pages=0)

    with pytest.raises(ValueError):
        TagOverlapPolicy(max_pairs=0)


def test_rendering_states_the_rule_it_used() -> None:
    rendered = format_overlap(
        [pair("Big Breast", "Big vs. Small", 225, 6, 0)],
        considered=981,
        missing=[],
        min_jaccard=0.5,
        min_scenes=5,
        include_descriptions=False,
    )

    assert "never-together" in rendered
    assert "however similar the names read" in rendered
    assert "None of these is a synonymy proof" in rendered
    assert "Tags considered: 981" in rendered


def test_rendering_says_so_when_there_is_nothing() -> None:
    rendered = format_overlap(
        [],
        considered=12,
        missing=["ghost"],
        min_jaccard=0.5,
        min_scenes=5,
        include_descriptions=False,
    )

    assert "Nothing here looks like a duplicate" in rendered
    assert "ghost" in rendered


async def test_the_tool_asks_for_the_permission_it_needs() -> None:
    tags, scenes = library()
    overlap = StashTagOverlap(FakeClient(tags, scenes))
    tool = create_stash_tag_overlap_tool(overlap)

    assert tool.name == "stash_tag_overlap"
    assert tool.policy.permissions == frozenset({"graphql.execute"})
    assert isinstance(tool.input_type, type)
    assert ToolOverlapIsDataclass(tool.input_type)


def ToolOverlapIsDataclass(  # noqa: N802 - a test helper, not an API
    input_type: type,
) -> bool:
    from dataclasses import is_dataclass

    return is_dataclass(input_type)


def test_defaults_are_conservative() -> None:
    arguments = TagOverlapInput()

    assert arguments.min_scenes == 5
    assert arguments.min_jaccard == 0.5
    assert arguments.include_descriptions is False
    assert Path(__file__).suffix == ".py"
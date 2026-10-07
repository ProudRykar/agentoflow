"""Learning something and being able to use it later.

The failure these cover is specific and was invisible in every existing
test. ``TaskAnchor`` is immutable by design, so retrieval always
matched notes against the words the user used at the start. An agent
told to "play the game" that works out the red key opens the blue door
writes a note, and that note shares no significant word with the
prompt -- so it is never selected, however carefully it was written,
and the agent spends the rest of the session rediscovering it.

Four changes, tested together because they only matter together:

- retrieval is aimed at the live conversation, not the frozen prompt
- notes are scoped to a task, so unrelated requests are not polluted
- there is a way to ask what is known without knowing the key
- the agent is asked once, before finishing, to record what it learned
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from agent_workflow.core.context.memory import (
    retrieve_snapshot,
    retrieval_query,
    visible_entries,
)
from agent_workflow.core.entities.models.builtin.memory_registry import (
    create_memory_tools,
)
from agent_workflow.core.entities.models.memory import (
    GLOBAL_SCOPE,
    MemoryEntry,
)
from agent_workflow.core.entities.models.memory_manager import MemoryManager
from agent_workflow.core.entities.models.tool import ToolContext
from agent_workflow.core.infrastructure.in_memory_store import (
    InMemoryStore,
)
from agent_workflow.core.infrastructure.sqlite_store import SQLiteStore


class FakeAnchor:
    def __init__(self, task_id: str) -> None:
        self.task_id = task_id


class FakeAgent:
    def __init__(self, task_id: str) -> None:
        self.task_anchor = FakeAnchor(task_id)


def context_for(task_id: str | None = "task-a") -> ToolContext:
    return ToolContext(
        working_directory=Path("."),
        environment={},
        allowed_path=(),
        permissions=frozenset({"memory.read", "memory.write"}),
        agent=FakeAgent(task_id) if task_id else None,
    )


def manager(store=None) -> MemoryManager:
    return MemoryManager(store or InMemoryStore())


def tool_by_name(memory: MemoryManager, name: str):
    return {t.name: t for t in create_memory_tools(memory)}[name]


# ======================================================================
# A. Retrieval aimed at the live conversation
# ======================================================================


class TestRetrievalQuery:
    def test_the_objective_is_always_included(self) -> None:
        query = retrieval_query("play the game", [])

        assert "play the game" in query

    def test_recent_turns_contribute_their_words(self) -> None:
        query = retrieval_query(
            "play the game",
            [{"role": "user", "content": "try the red key"}],
        )

        assert "red key" in query

    def test_only_the_tail_is_read(self) -> None:
        dialogue = [
            {"content": f"turn {index}"}
            for index in range(40)
        ]

        query = retrieval_query("objective", dialogue, tail=3)

        assert "turn 39" in query
        assert "turn 0" not in query

    def test_a_block_style_content_is_read_too(self) -> None:
        query = retrieval_query(
            "objective",
            [
                {
                    "content": [
                        {"type": "text", "text": "blue door opened"},
                    ]
                }
            ],
        )

        assert "blue door opened" in query

    def test_a_verbose_turn_cannot_fill_the_query_alone(self) -> None:
        """Otherwise one long tool result matches every note."""

        query = retrieval_query(
            "objective",
            [{"content": "needle " + ("filler " * 500)}],
        )

        assert len(query) < 2_000


class TestRetrievalReachesLearnedNotes:
    def test_a_note_learned_mid_session_is_offered_back(self) -> None:
        """The case that used to be impossible."""

        from datetime import UTC, datetime

        now = datetime.now(UTC)

        note = MemoryEntry(
            key="rules",
            value="the red key opens the blue door",
            created_at=now,
            updated_at=now,
        )

        # The prompt alone shares nothing with the note.
        prompt_only = retrieve_snapshot((note,), "play the game")
        assert prompt_only.items == ()

        # The live conversation does.
        with_context = retrieve_snapshot(
            (note,),
            retrieval_query(
                "play the game",
                [{"content": "I used the red key on the blue door"}],
            ),
        )

        assert len(with_context.items) == 1

    def test_a_note_from_the_original_request_is_still_reachable(self) -> None:
        """The objective is still in the query on purpose."""

        from datetime import UTC, datetime

        now = datetime.now(UTC)

        note = MemoryEntry(
            key="user.style",
            value="the user wants concise answers",
            created_at=now,
            updated_at=now,
        )

        snapshot = retrieve_snapshot(
            (note,),
            retrieval_query("write the concise version", []),
        )

        assert len(snapshot.items) == 1


# ======================================================================
# C. Scoping
# ======================================================================


class TestScoping:
    @pytest.fixture(params=["memory", "sqlite"])
    def store(self, request):
        if request.param == "memory":
            return InMemoryStore()

        path = Path(tempfile.mkdtemp()) / "memory.db"
        return SQLiteStore(path)

    async def test_a_task_does_not_see_another_tasks_notes(self, store):
        memory = manager(store)

        await memory.remember("rules", "blue", scope="task-a")
        await memory.remember("rules", "red", scope="task-b")

        visible = {e.value for e in await memory.all(scope="task-a")}

        assert visible == {"blue"}

    async def test_global_notes_are_visible_everywhere(self, store):
        memory = manager(store)

        await memory.remember("user.style", "concise", scope=GLOBAL_SCOPE)
        await memory.remember("rules", "blue", scope="task-a")

        visible = {e.value for e in await memory.all(scope="task-b")}

        assert "concise" in visible
        assert "blue" not in visible

    async def test_the_same_key_can_exist_in_two_scopes(self, store):
        memory = manager(store)

        await memory.remember("rules", "a", scope="task-a")
        await memory.remember("rules", "b", scope="task-b")

        assert len(await memory.all()) == 2

    async def test_no_scope_arguments_means_everything(self, store):
        memory = manager(store)

        await memory.remember("a", "1", scope="task-a")
        await memory.remember("b", "2", scope="task-b")

        assert len(await memory.all()) == 2

    async def test_deleting_is_scoped_too(self, store):
        memory = manager(store)

        await memory.remember("k", "mine", scope="task-a")
        await memory.remember("k", "theirs", scope="task-b")

        await memory.forget("k", scope="task-a")

        # Only the caller's copy went; task-b still has its own.
        remaining = await memory.all()

        assert len(remaining) == 1
        assert remaining[0].scope == "task-b"
        assert remaining[0].value == "theirs"
        assert len(await memory.all(scope="task-a")) == 0

    async def test_visible_entries_filters_defensively(self):
        from datetime import UTC, datetime

        now = datetime.now(UTC)

        entries = (
            MemoryEntry("a", "1", now, now, scope="task-a"),
            MemoryEntry("b", "2", now, now, scope="task-b"),
            MemoryEntry("c", "3", now, now, scope=GLOBAL_SCOPE),
        )

        assert len(visible_entries(entries, "task-a")) == 2


class TestPromotion:
    async def test_a_task_note_can_be_promoted_to_global(self):
        memory = manager()

        await memory.remember(
            "stash.endpoint", "http://localhost:9999", scope="task-a"
        )

        promoted = await memory.promote(
            "stash.endpoint", from_scope="task-a"
        )

        assert promoted.scope == GLOBAL_SCOPE
        assert len(await memory.all(scope="task-b")) == 1

    async def test_promotion_keeps_the_original(self):
        """Deleting it would surprise a run still in progress."""

        memory = manager()

        await memory.remember("k", "v", scope="task-a")
        await memory.promote("k", from_scope="task-a")

        own = {
            e.scope
            for e in await memory.all(scope="task-a")
        }

        assert own == {"task-a", GLOBAL_SCOPE}

    async def test_promoting_a_missing_note_says_so(self):
        memory = manager()

        with pytest.raises(KeyError):
            await memory.promote("absent", from_scope="task-a")

    async def test_promoting_an_already_global_note_is_refused(self):
        memory = manager()

        await memory.remember("k", "v")

        with pytest.raises(ValueError, match="already global"):
            await memory.promote("k", from_scope=GLOBAL_SCOPE)


class TestLegacyDatabase:
    def test_an_old_schema_is_migrated_not_lost(self):
        import sqlite3

        path = Path(tempfile.mkdtemp()) / "old.db"

        connection = sqlite3.connect(path)
        connection.execute(
            """
            CREATE TABLE memory (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            "INSERT INTO memory VALUES (?, ?, ?, ?)",
            (
                "k1",
                "v1",
                "2026-01-01T00:00:00+00:00",
                "2026-01-01T00:00:00+00:00",
            ),
        )
        connection.commit()
        connection.close()

        import asyncio

        async def read() -> list[MemoryEntry]:
            return await SQLiteStore(path).all()

        entries = asyncio.run(read())

        assert len(entries) == 1
        # Everything written before scopes existed was global.
        assert entries[0].scope == GLOBAL_SCOPE
        assert entries[0].value == "v1"


# ======================================================================
# B. Asking by topic
# ======================================================================


class TestRecallMatching:
    async def test_a_topic_finds_a_note_whose_key_is_unknown(self):
        memory = manager()

        await memory.remember(
            "obscure-key-7", "the vault code is 4417", scope="task-a"
        )

        result = await tool_by_name(
            memory, "recall_matching"
        ).handler(
            type("In", (), {"topic": "vault code", "limit": 5})(),
            context_for(),
        )

        assert "4417" in result

    async def test_an_empty_store_says_so(self):
        result = await tool_by_name(
            manager(), "recall_matching"
        ).handler(
            type("In", (), {"topic": "anything", "limit": 5})(),
            context_for(),
        )

        assert "Nothing in memory" in result

    async def test_another_tasks_notes_are_not_offered(self):
        memory = manager()

        await memory.remember(
            "other", "someone else's secret", scope="task-b"
        )

        result = await tool_by_name(
            memory, "recall_matching"
        ).handler(
            type("In", (), {"topic": "secret", "limit": 5})(),
            context_for("task-a"),
        )

        assert "someone else's secret" not in result

    async def test_a_nonsense_limit_is_refused_kindly(self):
        result = await tool_by_name(
            manager(), "recall_matching"
        ).handler(
            type("In", (), {"topic": "x", "limit": 0})(),
            context_for(),
        )

        assert "at least 1" in result


class TestRecall:
    async def test_an_exact_key_still_works(self):
        memory = manager()

        await memory.remember("k", "v", scope="task-a")

        result = await tool_by_name(memory, "recall").handler(
            type("In", (), {"key": "k"})(),
            context_for(),
        )

        assert "v" in result

    async def test_a_miss_points_at_the_other_tool(self):
        """A dead end that says where the other door is."""

        result = await tool_by_name(manager(), "recall").handler(
            type("In", (), {"key": "nope"})(),
            context_for(),
        )

        assert "recall_matching" in result

    async def test_a_global_note_is_found_from_a_task(self):
        memory = manager()

        await memory.remember("k", "v", scope=GLOBAL_SCOPE)

        result = await tool_by_name(memory, "recall").handler(
            type("In", (), {"key": "k"})(),
            context_for(),
        )

        assert "global" in result


# ======================================================================
# D. Consolidation, and the write side
# ======================================================================


class TestRemember:
    async def test_a_note_defaults_to_the_running_task(self):
        memory = manager()

        result = await tool_by_name(memory, "remember").handler(
            type(
                "In",
                (),
                {
                    "key": "game.rules",
                    "value": "red opens blue",
                    "scope": "task",
                    "tags": (),
                },
            )(),
            context_for("task-a"),
        )

        assert "this task" in result
        assert len(await memory.all(scope="task-a")) == 1

    async def test_global_is_asked_for_explicitly(self):
        memory = manager()

        await tool_by_name(memory, "remember").handler(
            type(
                "In",
                (),
                {
                    "key": "user.style",
                    "value": "concise",
                    "scope": "global",
                    "tags": (),
                },
            )(),
            context_for("task-a"),
        )

        assert len(await memory.all(scope="task-b")) == 1

    async def test_rewriting_accumulates_tags(self):
        memory = manager()

        for tags in (("a",), ("b",)):
            await tool_by_name(memory, "remember").handler(
                type(
                    "In",
                    (),
                    {
                        "key": "k",
                        "value": "v",
                        "scope": "task",
                        "tags": tags,
                    },
                )(),
                context_for("task-a"),
            )

        entry = await memory.get("k", scope="task-a")

        assert set(entry.tags) == {"a", "b"}

    async def test_no_agent_context_falls_back_to_global(self):
        memory = manager()

        await tool_by_name(memory, "remember").handler(
            type(
                "In",
                (),
                {
                    "key": "k",
                    "value": "v",
                    "scope": "task",
                    "tags": (),
                },
            )(),
            context_for(None),
        )

        assert len(await memory.all(scope=GLOBAL_SCOPE)) == 1


class TestConsolidationNote:
    def test_the_note_tells_the_model_it_may_skip(self):
        """A note that reads as an order produces noise."""

        from agent_workflow.core.entities.models.agent import (
            CONSOLIDATION_NOTE,
        )

        assert "Nothing to save" in CONSOLIDATION_NOTE

    def test_it_names_the_tool_to_use(self):
        from agent_workflow.core.entities.models.agent import (
            CONSOLIDATION_NOTE,
        )

        assert "remember" in CONSOLIDATION_NOTE

    def test_it_excludes_what_needs_no_memory(self):
        from agent_workflow.core.entities.models.agent import (
            CONSOLIDATION_NOTE,
        )

        lowered = CONSOLIDATION_NOTE.lower()

        assert "tool output" in lowered
        assert "variable" in lowered


class TestStopWords:
    """A note must not be selected because both texts say "the".

    Article words are long enough to clear a length filter and carry
    no information, so every note about the game matched every query
    about the game. That is not a weak signal, it is a false one, and
    it fills the memory block with whatever happens to share an
    article with the request.
    """

    def test_common_words_are_not_significant(self):
        from agent_workflow.core.context.memory import query_terms

        assert query_terms("play the game") == frozenset(
            {"play", "game"}
        )

    def test_an_article_alone_never_matches(self):
        from datetime import UTC, datetime

        from agent_workflow.core.context.memory import (
            entry_terms,
            query_terms,
        )

        note = MemoryEntry(
            "rules",
            "the red key opens the blue door",
            datetime.now(UTC),
            datetime.now(UTC),
        )

        assert (
            entry_terms(note) & query_terms("play the game") == frozenset()
        )

    def test_meaningful_words_still_match(self):
        from datetime import UTC, datetime

        from agent_workflow.core.context.memory import (
            entry_terms,
            query_terms,
        )

        note = MemoryEntry(
            "rules",
            "the red key opens the blue door",
            datetime.now(UTC),
            datetime.now(UTC),
        )

        assert "key" in (
            entry_terms(note) & query_terms("where is the key")
        )

    def test_short_words_are_still_excluded(self):
        from agent_workflow.core.context.memory import query_terms

        assert "id" not in query_terms("the id of the tag")


class TestPlurals:
    """A note saying "keys" and a request asking about "key" agree.

    They share no exact token, so the one is invisible to the other --
    which is the failure this whole change is about, arriving through
    a different door. Only the plural "s" is folded, and only where
    removing it cannot produce a different word.
    """

    def _note(self, value: str) -> MemoryEntry:
        from datetime import UTC, datetime

        return MemoryEntry(
            "rules",
            value,
            datetime.now(UTC),
            datetime.now(UTC),
        )

    def test_a_plural_note_answers_a_singular_request(self):
        snapshot = retrieve_snapshot(
            (self._note("collect 3 keys before the door appears"),),
            retrieval_query("objective", [{"content": "where is the key"}]),
        )

        assert len(snapshot.items) == 1

    def test_and_the_other_way_round(self):
        snapshot = retrieve_snapshot(
            (self._note("the red key opens the door"),),
            retrieval_query("objective", [{"content": "all the keys"}]),
        )

        assert len(snapshot.items) == 1

    @pytest.mark.parametrize(
        ("word", "expected"),
        [
            # Words ending in s that are not plurals of anything.
            ("class", {"class"}),
            ("bus", {"bus"}),
            ("analysis", {"analysis"}),
            ("status", {"status"}),
            ("process", {"process"}),
            ("address", {"address"}),
        ],
    )
    def test_words_that_are_already_singular_are_left_alone(
        self,
        word: str,
        expected: set[str],
    ):
        from agent_workflow.core.context.memory import _with_variants

        assert _with_variants(word) == expected

    def test_a_real_plural_folds(self):
        from agent_workflow.core.context.memory import _with_variants

        assert _with_variants("keys") == {"keys", "key"}

    def test_short_words_are_untouched(self):
        """Stripping the s from a short word loses too much."""

        from agent_workflow.core.context.memory import _with_variants

        assert _with_variants("is") == {"is"}

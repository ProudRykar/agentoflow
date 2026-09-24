from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from core.context.memory import (
    entry_terms,
    memory_to_item,
    query_terms,
    retrieve_snapshot,
)
from core.entities.models.agent import Agent
from core.entities.models.llm import LLMResponse
from core.entities.models.llm_client import LLMClient
from core.entities.models.memory import MemoryEntry
from core.entities.models.memory_manager import MemoryManager
from core.entities.models.tool import ToolContext
from core.entities.models.tool_executor import ToolExecutor
from core.entities.models.tool_registry import ToolRegistry
from core.infrastructure.in_memory_store import InMemoryStore


def _entry(key: str, value: str) -> MemoryEntry:
    now = datetime.now(UTC)

    return MemoryEntry(
        key=key,
        value=value,
        created_at=now,
        updated_at=now,
    )


def test_query_terms_filters_short_tokens() -> None:
    assert query_terms("Analyze Textual!") == {
        "analyze",
        "textual",
    }
    assert query_terms("a be") == set()


def test_score_counts_shared_terms() -> None:
    from core.context.memory import score_entry

    entry = _entry(
        "ui-framework",
        "Textual is the preferred UI framework",
    )

    assert score_entry(entry, frozenset({"textual"})) == 1
    assert (
        score_entry(
            entry, frozenset({"textual", "framework"})
        )
        == 2
    )
    assert score_entry(entry, frozenset({"django"})) == 0
    assert score_entry(entry, frozenset()) == 0


def test_retrieve_snapshot_ranks_and_limits() -> None:
    entries = (
        _entry("other", "nothing relevant here xyz"),
        _entry(
            "ui-framework",
            "Textual is the preferred UI framework",
        ),
        _entry("textual-docs", "read the Textual guide"),
    )

    snapshot = retrieve_snapshot(
        entries,
        "Analyze the Textual framework",
        limit=2,
    )

    assert len(snapshot) == 2
    assert snapshot.items[0].reference == "ui-framework"
    assert snapshot.items[1].reference == "textual-docs"

    for item in snapshot.items:
        assert item.source.value == "memory"


def test_retrieve_snapshot_no_match_is_empty() -> None:
    snapshot = retrieve_snapshot(
        (_entry("other", "nothing relevant here xyz"),),
        "quantum chromodynamics",
    )

    assert len(snapshot) == 0
    assert retrieve_snapshot((), "anything").items == ()


def test_retrieve_snapshot_rejects_bad_limit() -> None:
    with pytest.raises(ValueError):
        retrieve_snapshot((), "anything", limit=0)


def test_memory_to_item_format() -> None:
    item = memory_to_item(_entry("key", "value"))

    assert item.content == "key: value"
    assert item.reference == "key"
    assert "MEMORY" in item.header()


class FinalLLM(LLMClient):
    def __init__(self) -> None:
        self.messages: list[list[dict[str, Any]]] = []

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: tuple[Any, ...] = (),
    ) -> LLMResponse:
        self.messages.append(messages)

        return LLMResponse(
            content="done",
            tool_calls=(),
            raw={},
        )


@pytest.mark.asyncio
async def test_relevant_memory_reaches_llm(
    tmp_path: Path,
) -> None:
    manager = MemoryManager(InMemoryStore())
    await manager.remember(
        "ui-framework",
        "Textual is the preferred UI framework",
    )
    await manager.remember(
        "unrelated",
        "quantum chromodynamics qqq",
    )

    registry = ToolRegistry()
    agent = Agent(
        llm=FinalLLM(),
        registry=registry,
        executor=ToolExecutor(registry),
        memory=manager,
    )

    result = await agent.run(
        prompt="Analyze the Textual framework",
        context=ToolContext(
            working_directory=tmp_path,
            environment={},
            allowed_path=(tmp_path,),
            permissions=frozenset(),
        ),
    )

    assert result == "done"

    llm = agent._llm
    assert isinstance(llm, FinalLLM)

    joined = "\n".join(
        str(message.get("content", ""))
        for message in llm.messages[0]
    )

    assert "[MEMORY]" in joined
    assert "Textual is the preferred UI framework" in joined
    assert "quantum chromodynamics" not in joined


@pytest.mark.asyncio
async def test_no_memory_manager_no_block(
    tmp_path: Path,
) -> None:
    registry = ToolRegistry()
    agent = Agent(
        llm=FinalLLM(),
        registry=registry,
        executor=ToolExecutor(registry),
    )

    result = await agent.run(
        prompt="Analyze the Textual framework",
        context=ToolContext(
            working_directory=tmp_path,
            environment={},
            allowed_path=(tmp_path,),
            permissions=frozenset(),
        ),
    )

    assert result == "done"

    llm = agent._llm
    assert isinstance(llm, FinalLLM)

    joined = "\n".join(
        str(message.get("content", ""))
        for message in llm.messages[0]
    )

    # The provenance legend always names [MEMORY]; a retrieved
    # block would start a line with the bare header instead.
    assert "[MEMORY]\n" not in joined


def test_entry_terms_cover_key_and_value() -> None:
    terms = entry_terms(_entry("ui-framework", "Textual"))

    # "ui" is too short to be a significant token.
    assert {"framework", "textual"} <= terms
    assert "ui" not in terms

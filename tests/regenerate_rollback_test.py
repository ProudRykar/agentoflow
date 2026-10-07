"""A retry must not inherit the attempt it replaces.

The evidence store is append-only by design, which is right within a
run and wrong across a retry. Before this, regenerating an answer left
the discarded attempt's fetched pages counted: a research contract
would read as satisfied by material the user had just thrown away, and
the second attempt would skip the research it had not actually done.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from agent_workflow.core.context.evidence import (
    Evidence,
    EvidenceStore,
)
from agent_workflow.core.entities.models.agent import Agent
from agent_workflow.core.entities.models.llm import LLMResponse
from agent_workflow.core.entities.models.llm_client import LLMClient
from agent_workflow.core.entities.models.research_contract import (
    ResearchPage,
    ResearchResult,
)
from agent_workflow.core.entities.models.tool import ToolContext
from agent_workflow.core.entities.models.tool_executor import ToolExecutor
from agent_workflow.core.entities.models.tool_registry import ToolRegistry


@dataclass
class NoArguments:
    pass


class SpeakingLLM(LLMClient):
    def __init__(self) -> None:
        self.calls = 0

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: tuple[Any, ...] = (),
    ) -> LLMResponse:
        self.calls += 1

        return LLMResponse(
            content=f"answer {self.calls}",
            tool_calls=(),
            raw={},
        )


def _context(tmp_path: Path) -> ToolContext:
    return ToolContext(
        working_directory=tmp_path,
        environment={},
        allowed_path=(tmp_path,),
        permissions=frozenset(),
    )


def _urls(store: EvidenceStore) -> set[str]:
    return {item.title for item in store.all()}


def _agent() -> Agent:
    registry = ToolRegistry()

    return Agent(
        llm=SpeakingLLM(),
        registry=registry,
        executor=ToolExecutor(registry),
    )


def _page(url: str) -> ResearchPage:
    return ResearchPage(
        url=url,
        depth=0,
        title=f"title {url}",
        content="body",
        links=(),
        content_bytes=4,
    )


def _result(url: str) -> ResearchResult:
    return ResearchResult(
        root_url=url,
        pages=(_page(url),),
        discovered_urls=(),
        failed_urls=(),
        max_depth_reached=0,
        total_bytes=4,
        page_limit_reached=False,
        byte_limit_reached=False,
    )


# ======================================================================
# The store primitive
# ======================================================================


def test_retain_only_drops_the_rest() -> None:
    store = EvidenceStore()

    for index in range(3):
        store._items[f"e{index}"] = Evidence(
            evidence_id=f"e{index}",
            kind="page",
            source="web",
            title="t",
            content="x",
        )

    assert store.retain_only(("e0",)) == 2
    assert store.ids == ("e0",)


def test_retain_only_unpins_what_it_drops() -> None:
    """A pin keeps a citation alive; a dropped item has no citation."""

    store = EvidenceStore()

    store._items["e0"] = Evidence(
        evidence_id="e0",
        kind="page",
        source="web",
        title="t",
        content="x",
    )
    store._items["e1"] = Evidence(
        evidence_id="e1",
        kind="page",
        source="web",
        title="t",
        content="x",
    )
    store.pin("e1")

    store.retain_only(("e0",))

    assert store.ids == ("e0",)


def test_retain_only_is_idempotent() -> None:
    store = EvidenceStore()

    store._items["e0"] = Evidence(
        evidence_id="e0",
        kind="page",
        source="web",
        title="t",
        content="x",
    )

    assert store.retain_only(("e0",)) == 0
    assert store.retain_only(("e0",)) == 0
    assert store.ids == ("e0",)


# ======================================================================
# The retry
# ======================================================================


@pytest.mark.asyncio
async def test_a_retry_does_not_keep_the_discarded_evidence(
    tmp_path: Path,
) -> None:
    agent = _agent()

    store = agent.orchestrator.evidence_store

    # Work done before the turn being replaced: it must survive.
    store.append_page(
        _page("https://example.com/earlier"),
        "https://example.com",
    )

    await agent.run(
        prompt="research it",
        context=_context(tmp_path),
    )

    # The attempt being replaced gathers more.
    agent.orchestrator.store_research_evidence(
        _result("https://example.com/discarded")
    )

    discarded = len(store)
    assert discarded > 1

    await agent.regenerate(context=_context(tmp_path))

    # Work from before the replaced turn survives; the replaced
    # attempt's own page does not.
    assert _urls(store) == {"title https://example.com/earlier"}
    assert len(store) < discarded


@pytest.mark.asyncio
async def test_a_retry_does_not_inherit_satisfied_coverage(
    tmp_path: Path,
) -> None:
    """The contract must not read as met by discarded work.

    Coverage is counted per run -- ``reset_for_run`` clears it at the
    start of every run and deliberately keeps the evidence store -- so
    the store was the one place a retry really could inherit the
    discarded attempt. Asserted through coverage anyway: what matters
    is the contract, not the mechanism.
    """

    agent = _agent()

    await agent.run(
        prompt="research it",
        context=_context(tmp_path),
    )

    agent.orchestrator.store_research_evidence(
        _result("https://example.com/discarded")
    )

    coverage = agent.orchestrator.task_progress.research_coverage

    assert "https://example.com/discarded" in coverage.fetched_urls

    await agent.regenerate(context=_context(tmp_path))

    assert "https://example.com/discarded" not in coverage.fetched_urls


@pytest.mark.asyncio
async def test_evidence_from_earlier_turns_survives_a_retry(
    tmp_path: Path,
) -> None:
    """Only the discarded attempt is undone."""

    agent = _agent()

    store = agent.orchestrator.evidence_store

    await agent.run(
        prompt="first",
        context=_context(tmp_path),
    )

    agent.orchestrator.store_research_evidence(
        _result("https://example.com/turn-one")
    )

    await agent.continue_run(
        prompt="second",
        context=_context(tmp_path),
    )

    agent.orchestrator.store_research_evidence(
        _result("https://example.com/turn-two")
    )

    await agent.regenerate(context=_context(tmp_path))

    # Turn one's pages stand; only turn two's attempt is undone.
    assert "title https://example.com/turn-one" in _urls(store)
    assert "title https://example.com/turn-two" not in _urls(store)


@pytest.mark.asyncio
async def test_a_second_retry_rolls_back_only_its_own_attempt(
    tmp_path: Path,
) -> None:
    agent = _agent()

    store = agent.orchestrator.evidence_store

    await agent.run(
        prompt="one",
        context=_context(tmp_path),
    )

    agent.orchestrator.store_research_evidence(
        _result("https://example.com/a")
    )

    await agent.regenerate(context=_context(tmp_path))

    agent.orchestrator.store_research_evidence(
        _result("https://example.com/b")
    )

    assert len(store) == 1

    await agent.regenerate(context=_context(tmp_path))

    assert len(store) == 0


@pytest.mark.asyncio
async def test_a_plain_continuation_rolls_nothing_back(
    tmp_path: Path,
) -> None:
    """Rollback belongs to a retry, not to every new turn."""

    agent = _agent()

    store = agent.orchestrator.evidence_store

    await agent.run(
        prompt="one",
        context=_context(tmp_path),
    )

    agent.orchestrator.store_research_evidence(
        _result("https://example.com/a")
    )

    before = len(store)

    await agent.continue_run(
        prompt="two",
        context=_context(tmp_path),
    )

    assert len(store) == before

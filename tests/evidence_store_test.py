from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from core.context.evidence import (
    Evidence,
    EvidenceStore,
)
from core.context.research import ResearchContext
from core.entities.models.agent import Agent
from core.entities.models.agent_orchestrator import AgentOrchestrator
from core.entities.models.llm import LLMResponse, LLMToolCall
from core.entities.models.llm_client import LLMClient
from core.entities.models.research_contract import (
    ResearchContract,
    ResearchCoverage,
    ResearchPage,
    ResearchResult,
)
from core.entities.models.task_contract import TaskContract
from core.entities.models.tool import Tool, ToolContext, ToolPolicy
from core.entities.models.tool_executor import ToolExecutor
from core.entities.models.tool_registry import ToolRegistry


def _make_page() -> ResearchPage:
    return ResearchPage(
        url="https://example.com/",
        depth=0,
        title="Example",
        content="SECRET_PAGE_BODY",
        links=(),
        content_bytes=100,
    )


def _make_result() -> ResearchResult:
    return ResearchResult(
        root_url="https://example.com/",
        pages=(_make_page(),),
        discovered_urls=(),
        failed_urls=(),
        max_depth_reached=0,
        total_bytes=100,
        page_limit_reached=False,
        byte_limit_reached=False,
    )


def test_evidence_requires_id_and_kind() -> None:
    with pytest.raises(ValueError):
        Evidence(
            evidence_id="",
            kind="research.page",
            source="https://example.com/",
            title="",
            content="",
        )

    with pytest.raises(ValueError):
        Evidence(
            evidence_id="ev-0001",
            kind="",
            source="https://example.com/",
            title="",
            content="",
        )


def test_evidence_serialization_roundtrip() -> None:
    evidence = Evidence.from_research_page(
        _make_page(),
        "ev-0001",
        "https://example.com/",
    )

    assert evidence.kind == "research.page"
    assert evidence.content == "SECRET_PAGE_BODY"

    assert Evidence.from_dict(evidence.to_dict()) == evidence


def test_evidence_store_append_and_lookup() -> None:
    store = EvidenceStore()

    first = store.append_page(
        _make_page(),
        "https://example.com/",
    )
    second = store.append_page(
        _make_page(),
        "https://example.com/",
    )

    assert first.evidence_id == "ev-0001"
    assert second.evidence_id == "ev-0002"
    assert len(store) == 2
    assert store.ids == ("ev-0001", "ev-0002")
    assert store.get("ev-0001") == first
    assert store.get("missing") is None


def test_evidence_store_serialization_roundtrip() -> None:
    store = EvidenceStore()
    store.append_page(
        _make_page(),
        "https://example.com/",
    )

    restored = EvidenceStore.from_dict(store.to_dict())

    assert restored.ids == store.ids
    assert restored.get("ev-0001") == store.get("ev-0001")

    # Counter survives: next id does not collide.
    nxt = restored.append_page(
        _make_page(),
        "https://example.com/",
    )

    assert nxt.evidence_id == "ev-0002"


def test_research_context_snapshot_isolates_coverage() -> None:
    coverage = ResearchCoverage()
    coverage.fetched_urls.add("https://example.com/")

    snapshot = ResearchContext.snapshot(
        contract=None,
        coverage=coverage,
        evidence_ids=("ev-0001",),
    )

    coverage.fetched_urls.add("https://other.example/")

    assert snapshot.coverage.fetched_urls == {
        "https://example.com/"
    }
    assert snapshot.evidence_ids == ("ev-0001",)


def test_store_research_evidence_merges_and_receipts() -> None:
    orchestrator = AgentOrchestrator()
    contract = TaskContract(
        requires_research=True,
        research=ResearchContract(
            root_urls=("https://example.com/",),
            min_pages=1,
            min_depth=0,
            require_all_roots=True,
        ),
    )
    orchestrator.set_task_contract(contract)

    receipt = orchestrator.store_research_evidence(
        _make_result(),
    )

    assert receipt.evidence_ids == ("ev-0001",)
    assert receipt.page_count == 1
    assert receipt.total_bytes == 100
    assert receipt.satisfied is True

    stored = orchestrator.evidence_store.get("ev-0001")

    assert stored is not None
    assert stored.content == "SECRET_PAGE_BODY"

    context = orchestrator.research_context

    assert context.contract == contract.research
    assert context.evidence_ids == ("ev-0001",)
    assert "https://example.com/" in context.coverage.fetched_urls


@dataclass(slots=True, frozen=True)
class FakeResearchInput:
    url: str


class ResearchThenDoneLLM(LLMClient):
    def __init__(self) -> None:
        self.messages: list[list[dict[str, Any]]] = []

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: tuple[Any, ...] = (),
    ) -> LLMResponse:
        self.messages.append(messages)

        tool_calls = [
            message
            for message in messages
            if message.get("role") == "tool"
        ]

        if not tool_calls:
            return LLMResponse(
                content="",
                tool_calls=(
                    LLMToolCall(
                        id="call_1",
                        name="fake_research",
                        arguments={
                            "url": "https://example.com/",
                        },
                    ),
                ),
                raw={},
            )

        return LLMResponse(
            content="done",
            tool_calls=(),
            raw={},
        )


@pytest.mark.asyncio
async def test_agent_writes_receipt_not_page_body(
    tmp_path: Path,
) -> None:
    async def handler(
        arguments: FakeResearchInput,
        context: ToolContext,
    ) -> str:
        return _make_result().to_json()

    registry = ToolRegistry()
    registry.register(
        Tool(
            name="fake_research",
            description="Fake research tool",
            input_type=FakeResearchInput,
            handler=handler,
            policy=ToolPolicy(
                permissions=frozenset({"test.research"}),
                timeout=1.0,
                max_output_size=100_000,
            ),
        )
    )

    llm = ResearchThenDoneLLM()
    orchestrator = AgentOrchestrator()
    agent = Agent(
        llm=llm,
        registry=registry,
        executor=ToolExecutor(registry),
        orchestrator=orchestrator,
    )

    context = ToolContext(
        working_directory=tmp_path,
        environment={},
        allowed_path=(tmp_path,),
        permissions=frozenset({"test.research"}),
    )

    result = await agent.run(
        prompt="Research https://example.com documentation",
        context=context,
    )

    assert result == "done"

    tool_message: dict[str, Any] | None = None

    for messages in llm.messages:
        for message in messages:
            if message.get("role") == "tool":
                tool_message = message

    assert tool_message is not None
    assert tool_message["content"].startswith("[research]")
    assert "ev-0001" in tool_message["content"]
    assert "SECRET_PAGE_BODY" not in tool_message["content"]

    stored = orchestrator.evidence_store.get("ev-0001")

    assert stored is not None
    assert stored.content == "SECRET_PAGE_BODY"

    from core.context.history import HistoryKind

    kinds = [
        item.kind
        for item in agent.controller.history.task_items(
            agent.task_anchor.task_id
        )
    ]

    assert HistoryKind.EVIDENCE_REF in kinds

from typing import Any

import pytest

from core.context.checkpoint import build_checkpoint
from core.context.context_assembler import ContextAssembler
from core.context.context_budget import ContextBudget
from core.context.context_item import (
    ContextItem,
    ContextSource,
)
from core.context.tokens import ApproximateTokenCounter
from core.entities.models.agent_orchestrator import AgentOrchestrator
from core.entities.models.planner import Planner
from core.entities.models.research_contract import (
    ResearchPage,
    ResearchResult,
)
from core.entities.models.task_contract import TaskContract
from core.entities.models.tool_definition import ToolDefinition


def _started_orchestrator(
    prompt: str = "Read test.txt",
) -> AgentOrchestrator:
    orchestrator = AgentOrchestrator()
    orchestrator.prepare_task(
        Planner().plan(prompt),
        TaskContract(),
        prompt=prompt,
    )
    orchestrator.on_agent_started(prompt)

    return orchestrator


def _build(
    orchestrator: AgentOrchestrator,
    conversation: tuple[dict[str, Any], ...] = (),
    assembler: ContextAssembler | None = None,
    **kwargs: Any,
) -> Any:
    assembler = assembler or ContextAssembler()

    return assembler.build(
        anchor=orchestrator.task_anchor,
        task_state=orchestrator.task_state,
        execution=orchestrator.execution_context,
        conversation_recent=conversation,
        evidence_selected=orchestrator.evidence_store.all(),
        execution_plan=orchestrator.plan,
        **kwargs,
    )


def test_fixed_order_and_provenance_headers() -> None:
    orchestrator = _started_orchestrator()

    request = _build(
        orchestrator,
        ({"role": "user", "content": "hello"},),
    )

    texts = [
        str(message.get("content", ""))
        for message in request.messages
        if message.get("role") == "system"
    ]
    joined = "\n".join(texts)

    assert "PROVENANCE RULES" in joined
    assert "[TASK ANCHOR]" in joined
    assert "[TASK STATE]" in joined
    assert "[EXECUTION]" in joined
    assert "[RESEARCH]" in joined

    heads = [
        str(message.get("content", "")).split("\n", 1)[0]
        for message in request.messages
        if message.get("role") == "system"
    ]

    assert heads[1] == "[TASK ANCHOR]"
    assert heads[2] == "[TASK STATE]"
    assert heads[3] == "[EXECUTION]"
    assert heads[4] == "[RESEARCH]"

    # Dialogue follows the system blocks.
    assert request.messages[-1] == {
        "role": "user",
        "content": "hello",
    }


def test_anchor_is_non_evictable_under_tiny_budget() -> None:
    orchestrator = _started_orchestrator()

    assembler = ContextAssembler(
        budget=ContextBudget(
            maximum_tokens=600,
            reserved_system=10,
            reserved_task=10,
            reserved_output=10,
        ),
    )

    conversation = tuple(
        {"role": "user", "content": f"message {index} filler text"}
        for index in range(30)
    )

    request = _build(
        orchestrator,
        conversation,
        assembler=assembler,
    )

    joined = "\n".join(
        str(message.get("content", ""))
        for message in request.messages
    )

    assert "Read test.txt" in joined
    assert "[TASK ANCHOR]" in joined
    # Old dialogue was evicted, newest kept.
    assert "message 29" in joined
    assert "message 0" not in joined


def test_conversation_never_starts_with_orphan_tool() -> None:
    orchestrator = _started_orchestrator()

    conversation = (
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": {},
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": "file body",
        },
    )

    request = _build(orchestrator, conversation)

    tail = [
        message
        for message in request.messages
        if message.get("role") in ("assistant", "tool", "user")
    ]

    assert tail[0].get("role") == "assistant"
    assert tail[-1] == conversation[-1]


def test_evidence_excerpts_enter_context() -> None:
    orchestrator = _started_orchestrator()
    orchestrator.store_research_evidence(
        ResearchResult(
            root_url="https://example.com/",
            pages=(
                ResearchPage(
                    url="https://example.com/",
                    depth=0,
                    title="Example",
                    content="FULL_PAGE_BODY",
                    links=(),
                    content_bytes=14,
                ),
            ),
            discovered_urls=(),
            failed_urls=(),
            max_depth_reached=0,
            total_bytes=14,
            page_limit_reached=False,
            byte_limit_reached=False,
        )
    )

    request = _build(orchestrator, ())

    joined = "\n".join(
        str(message.get("content", ""))
        for message in request.messages
    )

    assert "[EVIDENCE ev-0001]" in joined
    assert "FULL_PAGE_BODY" in joined


def test_completion_instruction_is_last() -> None:
    orchestrator = _started_orchestrator()

    request = _build(
        orchestrator,
        ({"role": "user", "content": "hi"},),
        completion_instruction="Keep going.",
    )

    last = request.messages[-1]

    assert last["role"] == "system"
    assert "[CURRENT INSTRUCTION]" in str(last["content"])
    assert "Keep going." in str(last["content"])


def test_checkpoint_block_included() -> None:
    orchestrator = _started_orchestrator()

    request = _build(
        orchestrator,
        (),
        checkpoint=build_checkpoint(orchestrator),
    )

    joined = "\n".join(
        str(message.get("content", ""))
        for message in request.messages
    )

    assert "[CHECKPOINT]" in joined


def test_memory_and_history_fit_when_space() -> None:
    orchestrator = _started_orchestrator()

    request = _build(
        orchestrator,
        ({"role": "user", "content": "hi"},),
        memory_snapshot=(
            ContextItem(
                source=ContextSource.MEMORY,
                content="dataclasses preferred",
            ),
        ),
        history_selected=(
            ContextItem(
                source=ContextSource.HISTORY,
                content="previous window summary",
                reference="window-01",
            ),
        ),
    )

    joined = "\n".join(
        str(message.get("content", ""))
        for message in request.messages
    )

    assert "[MEMORY]" in joined
    assert "[HISTORY window-01]" in joined


def test_delegation_hints_rendered_on_marked_steps() -> None:
    from core.entities.models.planner import Planner
    from core.entities.models.task_contract import TaskContract

    planner = Planner()
    prompt = "Research https://example.com docs"
    task_plan = planner.plan(prompt)
    contract = TaskContract(
        requires_research=True,
        research=task_plan.research,
    )

    orchestrator = _started_orchestrator(prompt)
    orchestrator.prepare_task(task_plan, contract, prompt=prompt)
    orchestrator.on_agent_started(prompt)

    request = _build(orchestrator, ())

    joined = "\n".join(
        str(message.get("content", ""))
        for message in request.messages
    )

    assert "delegatable → subagent role='researcher'" in joined
    assert "power='low'" in joined

    # Synthesis step line carries no hint.
    for line in joined.split("\n"):
        if "synthesis" in line and line.startswith("- ["):
            assert "delegatable" not in line


def test_no_hints_no_suffix() -> None:
    orchestrator = _started_orchestrator("Read test.txt")

    request = _build(orchestrator, ())

    joined = "\n".join(
        str(message.get("content", ""))
        for message in request.messages
    )

    assert "delegatable" not in joined


def test_failed_fetches_carry_guidance() -> None:
    from core.entities.models.research_contract import ResearchCoverage

    orchestrator = _started_orchestrator()
    orchestrator.task_progress.research_coverage.failed_urls.add(
        "https://example.com/blocked"
    )

    assert isinstance(
        orchestrator.task_progress.research_coverage,
        ResearchCoverage,
    )

    request = _build(orchestrator, ())

    joined = "\n".join(
        str(message.get("content", ""))
        for message in request.messages
    )

    assert "https://example.com/blocked" in joined
    assert "Do not retry a failed URL as-is" in joined


def test_unfetched_subpages_suggested() -> None:
    orchestrator = _started_orchestrator()

    orchestrator.task_progress.research_coverage.discovered_urls.update(
        {
            "https://example.com/widget_gallery/",
            "https://example.com/about/",
        }
    )

    request = _build(orchestrator, ())

    joined = "\n".join(
        str(message.get("content", ""))
        for message in request.messages
    )

    assert "Unfetched promising subpages" in joined
    assert "https://example.com/widget_gallery/" in joined
    assert "https://example.com/about/" not in joined

    orchestrator.task_progress.research_coverage.fetched_urls.add(
        "https://example.com/widget_gallery/"
    )

    rerequest = _build(orchestrator, ())

    rejoined = "\n".join(
        str(message.get("content", ""))
        for message in rerequest.messages
    )

    assert "Unfetched promising subpages" not in rejoined


def test_tools_pass_through() -> None:
    orchestrator = _started_orchestrator()
    tools = (
        ToolDefinition(
            name="read_file",
            description="read",
            input_schema={},
        ),
    )

    request = _build(orchestrator, (), tools=tools)

    assert request.tools == tools


def test_budget_validation() -> None:
    with pytest.raises(ValueError):
        ContextBudget(
            maximum_tokens=100,
            reserved_system=50,
            reserved_task=50,
            reserved_output=50,
        )


def test_counter_divisor_validation() -> None:
    with pytest.raises(ValueError):
        ApproximateTokenCounter(divisor=0)

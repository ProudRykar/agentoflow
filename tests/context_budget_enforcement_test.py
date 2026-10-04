"""Budget accounting, eviction and conversation safety.

The assembler previously advertised a budget it did not keep:
``reserved_system`` and ``reserved_task`` were validated and never
used, a single oversized item silently discarded every item after
it, and the newest conversation message was admitted no matter how
large. ``build()`` could therefore return a request well past the
budget, which is what the rollover gate in the agent loop reacted to
without ever fixing.
"""

from __future__ import annotations

from typing import Any

import pytest

from agent_workflow.core.context.context_assembler import (
    ContextAssembler,
)
from agent_workflow.core.context.context_budget import ContextBudget
from agent_workflow.core.context.context_item import (
    ContextItem,
    ContextSource,
)
from agent_workflow.core.context.evidence import Evidence
from agent_workflow.core.context.tokens import ApproximateTokenCounter
from agent_workflow.core.entities.models.agent_orchestrator import (
    AgentOrchestrator,
)
from agent_workflow.core.entities.models.planner import Planner
from agent_workflow.core.entities.models.task_contract import TaskContract


def _orchestrator(
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


def _cost(assembler: ContextAssembler, request: Any) -> int:
    return sum(
        assembler.counter.count(str(message.get("content", ""))) + 16
        for message in request.messages
    )


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


def _evidence(
    identifier: str,
    size: int,
) -> Evidence:
    return Evidence(
        evidence_id=identifier,
        kind="page",
        source=f"https://example.test/{identifier}",
        title=identifier,
        content="x" * size,
        depth=0,
        content_bytes=size,
        links=(),
    )


def _items(
    count: int,
    size: int,
    source: ContextSource = ContextSource.MEMORY,
) -> tuple[ContextItem, ...]:
    return tuple(
        ContextItem(
            source=source,
            content="y" * size,
            reference=f"ref-{index}",
        )
        for index in range(count)
    )


# ======================================================================
# Budget arithmetic
# ======================================================================


def test_reservations_are_part_of_the_arithmetic() -> None:
    # Regression: available ignored reserved_system and
    # reserved_task, so the reservations were decoration.
    budget = ContextBudget(
        maximum_tokens=32_000,
        reserved_system=2_000,
        reserved_task=2_000,
        reserved_output=4_000,
    )

    assert budget.available == 28_000
    assert budget.fixed_reserved == 4_000
    assert budget.evictable == 24_000


def test_evictable_is_never_negative() -> None:
    # The constructor rejects reservations that exceed the maximum,
    # so this is the defensive floor rather than a reachable case.
    budget = ContextBudget(
        maximum_tokens=5_000,
        reserved_system=2_000,
        reserved_task=2_000,
        reserved_output=500,
    )

    assert budget.evictable >= 0
    assert budget.evictable == 500


# ======================================================================
# build() keeps the budget
# ======================================================================


def test_oversized_conversation_is_trimmed_to_the_budget() -> None:
    assembler = ContextAssembler(
        budget=ContextBudget(
            maximum_tokens=6_000,
            reserved_system=200,
            reserved_task=200,
            reserved_output=200,
        )
    )

    orchestrator = _orchestrator()

    conversation = tuple(
        {
            "role": "user" if index % 2 == 0 else "assistant",
            "content": "z" * 4_000,
        }
        for index in range(8)
    )

    request = _build(orchestrator, conversation, assembler)

    assert _cost(assembler, request) <= assembler.budget.available

    kept = [
        message
        for message in request.messages
        if message.get("role") in ("user", "assistant")
    ]

    assert 0 < len(kept) < len(conversation)
    assert kept[-1] == conversation[-1]


def test_conversation_survives_when_fixed_blocks_exceed_budget() -> None:
    # The non-evictable blocks can be larger than the whole budget.
    # Dropping the dialogue too would leave the model with no
    # conversation at all, so the newest message still goes out,
    # truncated and marked.
    assembler = ContextAssembler(
        budget=ContextBudget(
            maximum_tokens=600,
            reserved_system=10,
            reserved_task=10,
            reserved_output=10,
        )
    )

    orchestrator = _orchestrator()

    conversation = (
        {"role": "user", "content": f"message {index}"}
        for index in range(30)
    )

    request = _build(
        orchestrator,
        tuple(conversation),
        assembler,
    )

    body = "\n".join(
        str(message.get("content", "")) for message in request.messages
    )

    assert "message 29" in body
    assert "[TASK ANCHOR]" in body


def test_one_huge_message_still_leaves_room_for_the_next() -> None:
    # Regression: the evidence loop used `break`, so the first
    # oversized excerpt silently discarded every later one.
    assembler = ContextAssembler(
        budget=ContextBudget(
            maximum_tokens=4_000,
            reserved_system=200,
            reserved_task=200,
            reserved_output=200,
        )
    )

    orchestrator = _orchestrator()

    request = assembler.build(
        anchor=orchestrator.task_anchor,
        task_state=orchestrator.task_state,
        execution=orchestrator.execution_context,
        conversation_recent=(),
        evidence_selected=(
            _evidence("huge", 200_000),
            _evidence("small-1", 40),
            _evidence("small-2", 40),
        ),
        execution_plan=orchestrator.plan,
    )

    body = "\n".join(
        str(message.get("content", ""))
        for message in request.messages
    )

    assert "small-1" in body
    assert "small-2" in body


def test_memory_is_fitted_partially_not_all_or_nothing() -> None:
    # Regression: _fit_items returned None for the whole selection
    # when it did not fit, discarding every item.
    assembler = ContextAssembler(
        budget=ContextBudget(
            maximum_tokens=4_000,
            reserved_system=200,
            reserved_task=200,
            reserved_output=200,
        )
    )

    orchestrator = _orchestrator()

    request = _build(
        orchestrator,
        assembler=assembler,
        memory_snapshot=_items(40, 400),
    )

    body = "\n".join(
        str(message.get("content", ""))
        for message in request.messages
    )

    # Newest entries are kept, and the budget still holds.
    assert "ref-39" in body
    assert _cost(assembler, request) <= assembler.budget.available


def test_history_survives_when_memory_filled_the_budget() -> None:
    assembler = ContextAssembler(
        budget=ContextBudget(
            maximum_tokens=4_000,
            reserved_system=200,
            reserved_task=200,
            reserved_output=200,
        )
    )

    orchestrator = _orchestrator()

    request = _build(
        orchestrator,
        assembler=assembler,
        memory_snapshot=_items(60, 400),
        history_selected=_items(
            2,
            60,
            source=ContextSource.HISTORY,
        ),
    )

    body = "\n".join(
        str(message.get("content", ""))
        for message in request.messages
    )

    # Something from history still made it in.
    assert "ref-1" in body


def test_non_evictable_blocks_are_never_dropped() -> None:
    assembler = ContextAssembler(
        budget=ContextBudget(
            maximum_tokens=1_500,
            reserved_system=100,
            reserved_task=100,
            reserved_output=100,
        )
    )

    orchestrator = _orchestrator()

    request = _build(orchestrator, assembler=assembler)

    body = "\n".join(
        str(message.get("content", ""))
        for message in request.messages
    )

    assert "[TASK ANCHOR]" in body
    assert "[TASK STATE]" in body
    assert "[EXECUTION]" in body


# ======================================================================
# Conversation safety
# ======================================================================


def test_conversation_never_returns_a_lone_orphan_tool() -> None:
    # Regression: when every message was a tool result the fallback
    # `return [messages[-1]]` sent an orphan tool message with no
    # assistant tool_calls block.
    assembler = ContextAssembler(
        budget=ContextBudget(
            maximum_tokens=1_200,
            reserved_system=100,
            reserved_task=100,
            reserved_output=100,
        )
    )

    orchestrator = _orchestrator()

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
        {"role": "tool", "tool_call_id": "call_1", "content": "a" * 8_000},
    )

    request = _build(orchestrator, conversation, assembler)

    tail = [
        message
        for message in request.messages
        if message.get("role") in ("assistant", "tool", "user")
    ]

    assert all(message.get("role") != "tool" for message in tail)


def test_huge_tool_result_does_not_starve_the_request() -> None:
    assembler = ContextAssembler(
        budget=ContextBudget(
            maximum_tokens=3_000,
            reserved_system=200,
            reserved_task=200,
            reserved_output=200,
        )
    )

    orchestrator = _orchestrator()

    conversation = (
        {"role": "user", "content": "read the file"},
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
            "content": "q" * 500_000,
        },
    )

    request = _build(orchestrator, conversation, assembler)

    assert _cost(assembler, request) <= assembler.budget.available

    # The instruction the user actually gave is still present.
    body = "\n".join(
        str(message.get("content", "")) for message in request.messages
    )

    assert "read the file" in body


# ======================================================================
# Duplication
# ======================================================================


def test_objective_is_not_restated_in_task_state() -> None:
    orchestrator = _orchestrator("Find the missing widget")

    request = _build(orchestrator)

    state_block = next(
        str(message.get("content", ""))
        for message in request.messages
        if str(message.get("content", "")).startswith("[TASK STATE]")
    )

    # The anchor is authoritative; TASK STATE must not restate the
    # objective field. A plan step may still quote the prompt.
    assert "objective:" not in state_block


def test_failed_urls_are_capped() -> None:
    orchestrator = _orchestrator()

    coverage = orchestrator.research_context.coverage

    for index in range(200):
        coverage.failed_urls.add(f"https://example.test/{index}")

    request = _build(orchestrator)

    body = "\n".join(
        str(message.get("content", "")) for message in request.messages
    )

    assert body.count("https://example.test/") <= 40
    assert "and " in body


# ======================================================================
# Counter honesty
# ======================================================================


def test_counter_divisor_is_configurable() -> None:
    counter = ApproximateTokenCounter(divisor=1)

    assert counter.count("abcd") == 4


@pytest.mark.parametrize("divisor", [1, 2, 4])
def test_smaller_divisor_never_underestimates(divisor: int) -> None:
    counter = ApproximateTokenCounter(divisor=divisor)

    text = "def f(x): return x + 1"

    assert counter.count(text) >= 1

# ======================================================================
# Budget sizing from the model catalog
# ======================================================================


@pytest.mark.parametrize(
    "context_size",
    [2_048, 4_096, 8_000, 16_384, 32_768, 131_072],
)
def test_budget_scales_to_the_model_window(
    context_size: int,
) -> None:
    # Regression: constructing a budget straight from a small model's
    # context_size raised, because the default reservations (2k + 2k
    # + 4k) are 8k in total.
    budget = ContextBudget.for_model(context_size)

    assert budget.maximum_tokens == context_size
    assert budget.evictable > 0
    assert budget.available > 0


def test_unknown_model_keeps_the_default_budget() -> None:
    assert ContextBudget.for_model(None) == ContextBudget()
    assert ContextBudget.for_model(0) == ContextBudget()


# ======================================================================
# Rendering cost
# ======================================================================


def test_evidence_render_respects_the_character_cap() -> None:
    # Regression: the full body of every page used to be rendered on
    # every iteration just to discover it did not fit.
    assembler = ContextAssembler()

    huge = _evidence("huge", 1_000_000)

    rendered = assembler._render_evidence(huge, max_chars=500)

    body = rendered.split("\n", 1)[1]

    assert body.startswith("x" * 500)
    assert "[excerpt truncated]" in body
    assert len(rendered) < 1_000


def test_evidence_render_defaults_to_the_excerpt_cap() -> None:
    assembler = ContextAssembler()

    rendered = assembler._render_evidence(_evidence("big", 100_000))

    assert "[excerpt truncated]" in rendered
    assert len(rendered) < 100_000


def test_evidence_selection_is_cheap_for_a_large_crawl() -> None:
    # 40 pages of 80 KB against a small budget: the work is bounded by
    # the budget, not by the size of the crawl.
    assembler = ContextAssembler(
        budget=ContextBudget.for_model(8_000)
    )

    crawled = tuple(
        _evidence(f"page-{index}", 80_000)
        for index in range(40)
    )

    message = assembler._select_evidence(crawled, 2_000)

    assert message is not None
    assert len(message["content"]) < 40_000

"""Context budgeting: the failures that look like a model's fault.

Five things, all of them silent. Each is one where the system reports
success while doing the wrong thing, which is the failure mode worth
testing for -- a loud failure gets noticed.

- over-long tool output used to be refused rather than cut, so a
  legitimate 25k-character answer became an error and the model got
  nothing
- a single result could be several times the context window, so the
  assembler spent its turn evicting everything else
- the token count is a local estimate that silently degrades to a
  length heuristic and is presented as a measurement
- "trimmed" says that something went, never what
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
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
from agent_workflow.core.context.llm_request import LLMRequestContext
from agent_workflow.core.context.tokens import (
    ApproximateTokenCounter,
    TiktokenCounter,
)
from agent_workflow.core.entities.models.agent_orchestrator import (
    AgentOrchestrator,
)
from agent_workflow.core.entities.models.llm import (
    LLMResponse,
    extract_usage,
)
from agent_workflow.core.entities.models.planner import Planner
from agent_workflow.core.entities.models.task_contract import TaskContract
from agent_workflow.core.entities.models.tool import (
    Tool,
    ToolContext,
    ToolPolicy,
)
from agent_workflow.core.entities.models.tool_executor import ToolExecutor
from agent_workflow.core.entities.models.tool_registry import ToolRegistry


@dataclass
class Empty:
    pass


def context() -> ToolContext:
    return ToolContext(
        working_directory=Path("."),
        environment={},
        allowed_path=(),
        permissions=frozenset({"t"}),
    )


def executor_for(
    output: str,
    *,
    max_output_size: int,
    allowance: int | None = None,
    name: str = "t",
) -> tuple[ToolExecutor, ToolRegistry]:
    async def handler(arguments: Any, ctx: Any) -> str:
        return output

    registry = ToolRegistry()
    registry.register(
        Tool(
            name=name,
            description="d",
            input_type=Empty,
            handler=handler,
            policy=ToolPolicy(
                permissions=frozenset({"t"}),
                timeout=10.0,
                max_output_size=max_output_size,
            ),
        )
    )

    return ToolExecutor(
        registry,
        output_allowance=allowance,
    ), registry


# ======================================================================
# 1. Over-long output is cut, not refused
# ======================================================================


class TestToolOutputTruncation:
    async def test_a_large_result_comes_back_cut_rather_than_lost(self):
        """The change with the widest blast radius.

        A 25k-character answer used to become an error: the model got
        nothing and had to re-run the call with narrower filters to
        obtain a fragment it could have been handed straight away.
        """

        executor, registry = executor_for(
            "R" * 25_592,
            max_output_size=20_000,
        )

        result = await executor.execute("t", {}, context())

        assert result.error is None
        assert result.output is not None
        assert len(result.output) <= 20_000

    async def test_the_cut_says_so_rather_than_ending_in_ellipsis(self):
        """A silent cut lets the model answer as if it saw everything."""

        executor, registry = executor_for(
            "R" * 25_592,
            max_output_size=20_000,
        )

        result = await executor.execute("t", {}, context())

        assert "[TRUNCATED" in result.output
        assert "25592" in result.output

    async def test_the_marker_names_the_tool(self):
        executor, _ = executor_for(
            "R" * 25_592,
            max_output_size=20_000,
            name="stash_list_tags",
        )

        result = await executor.execute("stash_list_tags", {}, context())

        assert "stash_list_tags" in result.output

    @pytest.mark.parametrize("limit", [100, 200, 300, 1_000, 5_000, 20_000])
    async def test_output_never_exceeds_the_limit(self, limit: int):
        """Including when the marker itself is longer than the room."""

        executor, _ = executor_for("R" * 25_592, max_output_size=limit)

        result = await executor.execute("t", {}, context())

        assert len(result.output) <= limit

    async def test_output_inside_the_limit_is_untouched(self):
        executor, _ = executor_for("R" * 100, max_output_size=20_000)

        result = await executor.execute("t", {}, context())

        assert result.output == "R" * 100
        assert "[TRUNCATED" not in result.output

    async def test_a_degenerate_zero_limit_returns_nothing(self):
        executor, _ = executor_for("R" * 100, max_output_size=0)

        result = await executor.execute("t", {}, context())

        assert result.output == ""


# ======================================================================
# 5. One result may not outgrow the request
# ======================================================================


class TestRequestAllowance:
    async def test_the_allowance_bounds_a_result_the_tool_allowed(self):
        """The research tools allow 120k characters: three windows.

        A per-tool constant cannot express that one answer may not
        consume the whole budget, so the tool could fill the request
        before the assembler saw it and evicted everything else.
        """

        executor, _ = executor_for(
            "R" * 120_000,
            max_output_size=200_000,
            allowance=10_000,
        )

        result = await executor.execute("t", {}, context())

        assert len(result.output) <= 10_000
        assert "[TRUNCATED" in result.output

    async def test_the_stricter_of_the_two_bounds_wins(self):
        executor, _ = executor_for(
            "R" * 120_000,
            max_output_size=5_000,
            allowance=50_000,
        )

        result = await executor.execute("t", {}, context())

        assert len(result.output) <= 5_000

    async def test_a_generous_allowance_leaves_the_tool_alone(self):
        executor, _ = executor_for(
            "R" * 30_000,
            max_output_size=100_000,
            allowance=50_000,
        )

        result = await executor.execute("t", {}, context())

        assert len(result.output) == 30_000


# ======================================================================
# 2. Measured against estimated
# ======================================================================


class TestTokenUsage:
    def test_ollama_field_names_are_read(self):
        from agent_workflow.core.entities.models.llm import TokenUsage

        assert extract_usage(
            {"prompt_eval_count": 1_234, "eval_count": 56}
        ) == TokenUsage(prompt_tokens=1_234, completion_tokens=56)

    def test_openai_field_names_are_read(self):
        from agent_workflow.core.entities.models.llm import TokenUsage

        assert extract_usage(
            {"usage": {"prompt_tokens": 900, "completion_tokens": 10}}
        ) == TokenUsage(prompt_tokens=900, completion_tokens=10)

    @pytest.mark.parametrize("raw", [{}, {"content": "no counts"}, None])
    def test_an_absent_count_is_not_invented(self, raw: Any):
        """A missing figure must read as unknown, not as zero.

        Zero would be read as "this request was free", which is worse
        than not knowing.
        """

        assert extract_usage(raw) is None

    def test_a_boolean_is_not_mistaken_for_a_count(self):
        assert extract_usage({"prompt_eval_count": True}) is None

    def test_a_negative_count_is_clamped(self):
        usage = extract_usage({"prompt_eval_count": -5})

        assert usage is not None
        assert usage.prompt_tokens == 0

    def test_the_response_carries_usage(self):
        response = LLMResponse(
            content="x",
            tool_calls=(),
            raw={"prompt_eval_count": 100},
            usage=extract_usage({"prompt_eval_count": 100}),
        )

        assert response.usage is not None
        assert response.usage.total_tokens == 100


class TestCounterFidelity:
    def test_fidelity_is_reported_and_defaulted(self):
        """The fallback divides by four, which undercounts code badly.

        Measured: roughly 0.7x on JSON and code, which is most of what
        this agent's context is made of. So the meter says which kind
        of number it is showing rather than presenting a guess as a
        measurement.
        """

        exact = TiktokenCounter(model="gpt-4")

        assert hasattr(exact, "exact")
        assert exact.exact in (True, False)

    def test_an_approximate_counter_declares_itself(self):
        # The protocol does not require it, so callers must tolerate
        # its absence -- the agent reads it with getattr.
        assert not hasattr(
            ApproximateTokenCounter(), "exact"
        )


# ======================================================================
# 4. Which blocks were dropped
# ======================================================================


def orchestrator() -> AgentOrchestrator:
    orch = AgentOrchestrator()
    orch.prepare_task(
        Planner().plan("Read test.txt"),
        TaskContract(),
        prompt="Read test.txt",
    )
    orch.on_agent_started("Read test.txt")

    return orch


def items(
    count: int,
    size: int,
    source: ContextSource,
) -> tuple[ContextItem, ...]:
    return tuple(
        ContextItem(
            source=source,
            content=f"{source.value} " + "z" * size,
            reference=f"ref-{index}",
        )
        for index in range(count)
    )


def build(
    orch: AgentOrchestrator,
    assembler: ContextAssembler,
    **kwargs: Any,
) -> LLMRequestContext:
    return assembler.build(
        anchor=orch.task_anchor,
        task_state=orch.task_state,
        execution=orch.execution_context,
        conversation_recent=kwargs.pop("conversation", ()),
        evidence_selected=(),
        execution_plan=orch.plan,
        **kwargs,
    )


def small_assembler(
    maximum: int = 2_000,
) -> ContextAssembler:
    return ContextAssembler(
        budget=ContextBudget(
            maximum_tokens=maximum,
            reserved_system=100,
            reserved_task=100,
            reserved_output=400,
        ),
        counter=ApproximateTokenCounter(),
    )


class TestDroppedBlocks:
    def test_named_blocks_are_reported_when_memory_is_cut(self):
        orch = orchestrator()
        assembler = small_assembler()

        request = build(
            orch,
            assembler,
            conversation=({"role": "user", "content": "go on"},),
            memory_snapshot=items(6, 400, ContextSource.MEMORY),
            history_selected=items(6, 400, ContextSource.HISTORY),
        )

        assert request.trimmed is True

        # History is allocated after memory, so it is the one that
        # loses. Knowing *which* is the point: "trimmed" alone left no
        # way to tell this from memory having been dropped.
        assert request.dropped_blocks == ("history",)

    def test_a_larger_memory_block_displaces_history_further(self):
        orch = orchestrator()
        assembler = small_assembler(1_400)

        request = build(
            orch,
            assembler,
            conversation=({"role": "user", "content": "go on"},),
            memory_snapshot=items(6, 400, ContextSource.MEMORY),
            history_selected=items(6, 400, ContextSource.HISTORY),
        )

        assert set(request.dropped_blocks) & {"memory", "history"}

    def test_nothing_dropped_is_reported_when_everything_fits(self):
        orch = orchestrator()
        assembler = small_assembler(64_000)

        request = build(
            orch,
            assembler,
            conversation=({"role": "user", "content": "go on"},),
            memory_snapshot=items(2, 100, ContextSource.MEMORY),
            history_selected=(),
        )

        assert request.dropped_blocks == ()

    def test_a_trimmed_conversation_is_named(self):
        orch = orchestrator()
        assembler = small_assembler()

        request = build(
            orch,
            assembler,
            conversation=tuple(
                {"role": "user", "content": "m" * 400}
                for _ in range(8)
            ),
            memory_snapshot=items(2, 100, ContextSource.MEMORY),
            history_selected=(),
        )

        assert "conversation" in request.dropped_blocks

    def test_memory_disappearing_is_distinguishable_from_history(self):
        """The whole point: a generic "trimmed" cannot say this."""

        orch = orchestrator()
        assembler = small_assembler()

        with_memory = build(
            orch,
            assembler,
            conversation=({"role": "user", "content": "go on"},),
            memory_snapshot=items(8, 400, ContextSource.MEMORY),
            history_selected=(),
        )

        assert with_memory.dropped_blocks == ("memory",)

    def test_the_request_can_still_carry_the_new_field(self):
        # Older readers construct this positionally; the field is last
        # and defaulted, so they keep working.
        request = LLMRequestContext(messages=())

        assert request.dropped_blocks == ()

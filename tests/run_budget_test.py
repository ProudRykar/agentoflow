"""What a run is allowed to spend.

Accounting alone is a report card after the money is gone. These
tests are about the two ways a run actually goes wrong: spending
without a ceiling, and reporting a number that was never measured.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from agent_workflow.core.entities.models.agent import Agent
from agent_workflow.core.entities.models.agent_trace import AgentFinished
from agent_workflow.core.entities.models.llm import (
    LLMToolCall,
    LLMResponse,
    TokenUsage,
)
from agent_workflow.core.entities.models.llm_client import LLMClient
from agent_workflow.core.entities.models.run_usage import RunUsage
from agent_workflow.core.entities.models.tool import (
    Tool,
    ToolContext,
    ToolPolicy,
)
from agent_workflow.core.entities.models.tool_executor import ToolExecutor
from agent_workflow.core.entities.models.tool_registry import ToolRegistry


@dataclass
class NoArguments:
    path: str = "x"


class LoopingLLM(LLMClient):
    """Keeps calling a tool, so a run cannot finish on its own."""

    def __init__(
        self,
        prompt_tokens: int = 400,
        usage: bool = True,
    ) -> None:
        self.calls = 0
        self._prompt_tokens = prompt_tokens
        self._usage = usage

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: tuple[Any, ...] = (),
    ) -> LLMResponse:
        self.calls += 1

        return LLMResponse(
            content="",
            tool_calls=(
                LLMToolCall(
                    id=f"call-{self.calls}",
                    name="noop",
                    arguments={"path": "x"},
                ),
            ),
            raw={},
            usage=(
                TokenUsage(
                    prompt_tokens=self._prompt_tokens,
                    completion_tokens=50,
                )
                if self._usage
                else None
            ),
        )


class SpeakingLLM(LLMClient):
    def __init__(
        self,
        usage: bool = True,
    ) -> None:
        self._usage = usage

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: tuple[Any, ...] = (),
    ) -> LLMResponse:
        return LLMResponse(
            content="done",
            tool_calls=(),
            raw={},
            usage=(
                TokenUsage(prompt_tokens=100, completion_tokens=10)
                if self._usage
                else None
            ),
        )


def _context(tmp_path: Path) -> ToolContext:
    return ToolContext(
        working_directory=tmp_path,
        environment={},
        allowed_path=(tmp_path,),
        permissions=frozenset(),
    )


def _looping_agent(
    llm: LLMClient,
    **kwargs: Any,
) -> Agent:
    registry = ToolRegistry()

    async def handler(
        arguments: NoArguments,
        context: ToolContext,
    ) -> str:
        return "ok"

    registry.register(
        Tool(
            name="noop",
            description="does nothing",
            input_type=NoArguments,
            handler=handler,
            policy=ToolPolicy(
                permissions=frozenset(),
                timeout=1.0,
                max_output_size=100,
            ),
        ),
    )

    return Agent(
        llm=llm,
        registry=registry,
        executor=ToolExecutor(registry),
        max_iterations=8,
        **kwargs,
    )


async def _finished(
    events: list[Any],
) -> list[AgentFinished]:
    return [
        event
        for event in events
        if isinstance(event, AgentFinished)
    ]


# ======================================================================
# The ledger itself
# ======================================================================


def test_unmeasured_is_not_reported_as_free() -> None:
    """The distinction the whole module exists to preserve."""

    usage = RunUsage()

    usage.note_unreported()

    assert usage.measured is False
    assert usage.total_tokens == 0

    # None, not 0.0: this run has no numbers, which is not the same
    # statement as having spent nothing.
    assert usage.estimate_cost((3.0, 15.0)) is None


def test_a_measured_run_can_be_priced() -> None:
    usage = RunUsage()

    usage.add(
        TokenUsage(prompt_tokens=1_000_000, completion_tokens=0)
    )

    assert usage.estimate_cost((3.0, 15.0)) == 3.0


def test_an_unpriced_run_reports_no_number() -> None:
    usage = RunUsage()

    usage.add(TokenUsage(prompt_tokens=100, completion_tokens=10))

    assert usage.estimate_cost(None) is None


# ======================================================================
# Unmeasured providers
# ======================================================================


@pytest.mark.asyncio
async def test_a_provider_reporting_nothing_is_counted_separately(
    tmp_path: Path,
) -> None:
    """
    Ollama does not always report usage. Zero totals must not be
    presented as "this run was free".
    """

    events: list[Any] = []

    async def callback(event: Any) -> None:
        events.append(event)

    agent = _looping_agent(SpeakingLLM(usage=False))

    await agent.run(
        prompt="hello",
        context=_context(tmp_path),
        on_event=callback,
    )

    finished = await _finished(events)

    assert len(finished) == 1

    event = finished[0]

    assert event.prompt_tokens == 0

    # One response happened; it simply reported no counts.
    assert event.llm_calls == 1
    assert event.llm_calls_without_usage == 1
    assert event.estimated_cost is None


@pytest.mark.asyncio
async def test_a_measured_run_reports_no_unmeasured_calls(
    tmp_path: Path,
) -> None:
    events: list[Any] = []

    async def callback(event: Any) -> None:
        events.append(event)

    agent = _looping_agent(SpeakingLLM(usage=True))

    await agent.run(
        prompt="hello",
        context=_context(tmp_path),
        on_event=callback,
    )

    event = (await _finished(events))[0]

    assert event.prompt_tokens == 100
    assert event.llm_calls_without_usage == 0


# ======================================================================
# The ceiling
# ======================================================================


@pytest.mark.asyncio
async def test_the_ceiling_stops_the_run(
    tmp_path: Path,
) -> None:
    llm = LoopingLLM(prompt_tokens=400)

    agent = _looping_agent(
        llm,
        max_prompt_tokens=800,
    )

    result = await agent.run(
        prompt="go",
        context=_context(tmp_path),
    )

    # Checked before the request, so it stops as soon as the allowance
    # is spent rather than one more turn later.
    assert llm.calls == 2

    assert "allowance" in result
    assert "800" in result


@pytest.mark.asyncio
async def test_the_ceiling_reports_what_was_spent(
    tmp_path: Path,
) -> None:
    """'Stopped for budget' without a number is not actionable."""

    events: list[Any] = []

    async def callback(event: Any) -> None:
        events.append(event)

    agent = _looping_agent(
        LoopingLLM(prompt_tokens=400),
        max_prompt_tokens=800,
        price_per_million=(3.0, 15.0),
    )

    await agent.run(
        prompt="go",
        context=_context(tmp_path),
        on_event=callback,
    )

    event = (await _finished(events))[0]

    assert event.prompt_tokens == 800
    assert event.completion_tokens == 100
    assert event.llm_calls == 2

    # 800 prompt + 100 completion at 3/15 per million.
    assert event.estimated_cost == pytest.approx(
        (800 * 3.0 + 100 * 15.0) / 1_000_000
    )


@pytest.mark.asyncio
async def test_without_a_ceiling_only_iterations_bound_the_run(
    tmp_path: Path,
) -> None:
    llm = LoopingLLM(prompt_tokens=400)

    agent = _looping_agent(llm)

    with pytest.raises(RuntimeError, match="maximum iterations"):
        await agent.run(
            prompt="go",
            context=_context(tmp_path),
        )

    assert llm.calls == 8


@pytest.mark.asyncio
async def test_an_unmeasured_run_is_not_stopped_by_the_ceiling(
    tmp_path: Path,
) -> None:
    """
    A provider that reports no usage cannot be shown to have exceeded
    anything. Ceiling off a number that does not exist would stop every
    such run on its first turn.
    """

    llm = LoopingLLM(usage=False)

    agent = _looping_agent(
        llm,
        max_prompt_tokens=10,
    )

    with pytest.raises(RuntimeError, match="maximum iterations"):
        await agent.run(
            prompt="go",
            context=_context(tmp_path),
        )

    assert llm.calls == 8


@pytest.mark.asyncio
async def test_a_generous_ceiling_does_not_interfere(
    tmp_path: Path,
) -> None:
    agent = _looping_agent(
        LoopingLLM(),
        max_prompt_tokens=10_000_000,
    )

    with pytest.raises(RuntimeError, match="maximum iterations"):
        await agent.run(
            prompt="go",
            context=_context(tmp_path),
        )


@pytest.mark.asyncio
async def test_the_allowance_resets_for_the_next_turn(
    tmp_path: Path,
) -> None:
    """A ceiling is per run, not a lifetime total."""

    llm = LoopingLLM(prompt_tokens=400)

    agent = _looping_agent(
        llm,
        max_prompt_tokens=800,
    )

    first = await agent.run(
        prompt="one",
        context=_context(tmp_path),
    )

    second = await agent.continue_run(
        prompt="two",
        context=_context(tmp_path),
    )

    assert "allowance" in first
    assert "allowance" in second

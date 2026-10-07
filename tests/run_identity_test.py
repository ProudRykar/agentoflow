"""Two turns are two runs, not one long run.

One run id for a whole session made a run record mean "this
conversation" rather than "this attempt". Two consequences, both
wrong:

- usage could not be attributed to the turn that spent it;
- the frontend clears a streaming flag by run id, so the current turn
  marked every earlier answer as no longer streaming.

Resume is the exception and stays on the same id: it is the same
execution continuing, not a new attempt.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from agent_workflow.core.entities.models.agent import (
    Agent,
    emit_to,
)
from agent_workflow.core.entities.models.agent_trace import (
    AgentFinished,
    AgentStarted,
)
from agent_workflow.core.entities.models.llm import (
    LLMResponse,
    TokenUsage,
)
from agent_workflow.core.entities.models.llm_client import LLMClient
from agent_workflow.core.entities.models.tool import ToolContext
from agent_workflow.core.entities.models.tool_executor import ToolExecutor
from agent_workflow.core.entities.models.tool_registry import ToolRegistry


@dataclass
class NoArguments:
    pass


class CountingLLM(LLMClient):
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
            usage=TokenUsage(
                prompt_tokens=100,
                completion_tokens=10,
            ),
        )


def _context(tmp_path: Path) -> ToolContext:
    return ToolContext(
        working_directory=tmp_path,
        environment={},
        allowed_path=(tmp_path,),
        permissions=frozenset(),
    )


def _agent() -> Agent:
    registry = ToolRegistry()

    return Agent(
        llm=CountingLLM(),
        registry=registry,
        executor=ToolExecutor(registry),
    )


async def _collect() -> tuple[list[Any], Any]:
    events: list[Any] = []

    async def callback(event: Any) -> None:
        events.append(event)

    return events, callback


@pytest.mark.asyncio
async def test_each_turn_gets_its_own_run_id(
    tmp_path: Path,
) -> None:
    events, callback = await _collect()

    agent = _agent()

    await agent.run(
        prompt="one",
        context=_context(tmp_path),
        on_event=callback,
    )

    await agent.continue_run(
        prompt="two",
        context=_context(tmp_path),
        on_event=callback,
    )

    started = [
        event
        for event in events
        if isinstance(event, AgentStarted)
    ]

    assert len(started) == 2
    assert started[0].run_id != started[1].run_id


@pytest.mark.asyncio
async def test_usage_is_attributable_to_its_turn(
    tmp_path: Path,
) -> None:
    events, callback = await _collect()

    agent = _agent()

    await agent.run(
        prompt="one",
        context=_context(tmp_path),
        on_event=callback,
    )

    await agent.continue_run(
        prompt="two",
        context=_context(tmp_path),
        on_event=callback,
    )

    finished = [
        event
        for event in events
        if isinstance(event, AgentFinished)
    ]

    assert len(finished) == 2

    # Each turn is counted on its own, not accumulated onto the first.
    for event in finished:
        assert event.prompt_tokens == 100
        assert event.llm_calls == 1

    assert finished[0].run_id != finished[1].run_id


@pytest.mark.asyncio
async def test_regenerate_is_a_third_distinct_run(
    tmp_path: Path,
) -> None:
    events, callback = await _collect()

    agent = _agent()

    await agent.run(
        prompt="one",
        context=_context(tmp_path),
        on_event=callback,
    )

    await agent.regenerate(
        context=_context(tmp_path),
        on_event=callback,
    )

    started = [
        event
        for event in events
        if isinstance(event, AgentStarted)
    ]

    ids = [event.run_id for event in started]

    assert len(set(ids)) == len(ids)


@pytest.mark.asyncio
async def test_resume_stays_on_the_same_run(
    tmp_path: Path,
) -> None:
    """Resume is the same execution continuing, not a new attempt."""

    events, callback = await _collect()

    agent = _agent()

    await agent.run(
        prompt="one",
        context=_context(tmp_path),
        on_event=callback,
    )

    before = agent.run_id

    # Resume refuses a finished run, so the id must not have moved
    # before it got the chance to refuse.
    with pytest.raises(RuntimeError):
        await agent.resume_run(
            context=_context(tmp_path),
            on_event=callback,
        )

    assert agent.run_id == before


def test_run_ids_are_predictable() -> None:
    """A replay must reproduce them, and a test must be able to name them."""

    agent = _agent()

    assert agent._mint_run_id() == "run-1"
    assert agent._mint_run_id() == "run-2"


def test_a_configured_run_id_becomes_the_prefix() -> None:
    registry = ToolRegistry()

    agent = Agent(
        llm=CountingLLM(),
        registry=registry,
        executor=ToolExecutor(registry),
        run_id="sess7",
    )

    assert agent._mint_run_id() == "sess7-1"


# ======================================================================
# Synchronous callbacks
# ======================================================================


@pytest.mark.asyncio
async def test_a_synchronous_callback_is_accepted(
    tmp_path: Path,
) -> None:
    """events.append is the obvious thing to write."""

    events: list[Any] = []

    agent = _agent()

    result = await agent.run(
        prompt="one",
        context=_context(tmp_path),
        on_event=events.append,
    )

    assert result
    assert len(events) > 0


@pytest.mark.asyncio
async def test_an_async_callback_is_still_awaited(
    tmp_path: Path,
) -> None:
    events: list[Any] = []

    async def callback(event: Any) -> None:
        # A yield point: if the result were not awaited, this would
        # never run before the caller resumed.
        await __import__("asyncio").sleep(0)

        events.append(event)

    agent = _agent()

    await agent.run(
        prompt="one",
        context=_context(tmp_path),
        on_event=callback,
    )

    assert len(events) > 0


@pytest.mark.asyncio
async def test_emit_to_handles_both_and_none() -> None:
    seen: list[Any] = []

    await emit_to(None, "ignored")
    await emit_to(seen.append, "sync")

    async def callback(value: Any) -> None:
        seen.append(value)

    await emit_to(callback, "async")

    assert seen == ["sync", "async"]

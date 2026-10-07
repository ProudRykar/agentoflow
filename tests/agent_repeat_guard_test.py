"""The agent must not loop forever on a failing tool call.

Observed: a tool that kept returning an error was called again with
identical arguments, iteration after iteration, until the iteration
cap killed the run with a stack trace instead of an answer. The model
had no signal that it was repeating itself, so it kept "trying one
more time".
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_workflow.core.entities.models.agent import Agent
from agent_workflow.core.entities.models.llm import (
    LLMResponse,
    LLMToolCall,
)
from agent_workflow.core.entities.models.llm_client import LLMClient
from agent_workflow.core.entities.models.tool import (
    Tool,
    ToolContext,
    ToolPolicy,
)
from agent_workflow.core.entities.models.tool_executor import (
    ToolExecutor,
)
from agent_workflow.core.entities.models.tool_registry import ToolRegistry


def _call(name: str, page: int) -> LLMToolCall:
    from agent_workflow.core.entities.models.llm import LLMToolCall

    return LLMToolCall(
        id=f"call_{name}_{page}",
        name=name,
        arguments={"page": page},
    )


class AlwaysFailingLLM(LLMClient):
    """Returns the same failing tool call every iteration."""

    def __init__(self) -> None:
        self.calls: list[object] = []

    async def chat(
        self,
        messages: list[dict],
        tools: tuple = (),
    ) -> LLMResponse:
        self.calls.append(messages)

        return LLMResponse(
            content="",
            tool_calls=[_call("flaky", 1)],
            raw={},
        )


class ScriptedLLM(LLMClient):
    """Fails twice, then answers in prose."""

    def __init__(self, failures: int) -> None:
        self.remaining = failures
        self.tool_iterations = 0

    async def chat(
        self,
        messages: list[dict],
        tools: tuple = (),
    ) -> LLMResponse:
        if self.remaining > 0:
            self.remaining -= 1
            self.tool_iterations += 1

            return LLMResponse(
                content="",
                tool_calls=[_call("flaky", 1)],
                raw={},
            )

        return LLMResponse(
            content=(
                "The tool keeps failing, so I could not retrieve the "
                "tags. Here is what happened instead."
            ),
            tool_calls=[],
            raw={},
        )


async def _flaky() -> str:
    return "Error: Input validation error"


def _context() -> ToolContext:
    return ToolContext(
        working_directory=Path.cwd(),
        environment={},
        allowed_path=(Path.cwd(),),
        permissions=frozenset(),
    )


def _agent(llm: LLMClient) -> tuple[Agent, ToolRegistry]:
    registry = ToolRegistry()
    registry.register(
        Tool(
            name="flaky",
            description="Always fails.",
            input_type=dict,
            handler=_flaky,
            policy=ToolPolicy(
                permissions=frozenset(),
                timeout=5.0,
                max_output_size=1024,
            ),
        )
    )

    return (
        Agent(
            llm=llm,
            registry=registry,
            executor=ToolExecutor(registry),
        ),
        registry,
    )


# ======================================================================
# Repeated identical calls
# ======================================================================


async def test_identical_failing_call_is_not_repeated_forever() -> None:
    llm = AlwaysFailingLLM()
    agent, _ = _agent(llm)

    result = await agent.run("fetch the tags", _context())

    tool_calls = [
        event
        for event in _events(agent)
        if event.__class__.__name__ == "ToolStarted"
    ]

    # Regression: the same call was issued once per iteration until
    # the cap aborted the run.
    assert len(tool_calls) <= 2

    # And the agent said something instead of raising.
    assert isinstance(result, str)
    assert result


async def test_repeat_budget_is_exhausted_then_reported() -> None:
    llm = AlwaysFailingLLM()
    agent, _ = _agent(llm)

    result = await agent.run("fetch the tags", _context())

    text = result.lower()

    # The final answer has to acknowledge the failure, otherwise the
    # user just sees a stopped run.
    assert "error" in text or "fail" in text or "could not" in text


async def test_agent_recovers_when_it_changes_its_approach() -> None:
    # Two failures, then the model gives up and answers: the budget
    # must not fire before it had a chance to stop by itself.
    llm = ScriptedLLM(failures=2)
    agent, _ = _agent(llm)

    result = await agent.run("fetch the tags", _context())

    assert llm.tool_iterations == 2
    assert "could not retrieve" in result.lower()


async def test_different_calls_are_not_counted_as_repeats() -> None:
    class VaryingLLM(LLMClient):
        def __init__(self) -> None:
            self.index = 0

        async def chat(
            self,
            messages: list[dict],
            tools: tuple = (),
        ) -> LLMResponse:
            self.index += 1

            if self.index <= 3:
                return LLMResponse(
                    content="",
                    tool_calls=[_call("flaky", self.index)],
                    raw={},
                )

            return LLMResponse(
                content="done",
                tool_calls=[],
                raw={},
            )

    llm = VaryingLLM()
    agent, _ = _agent(llm)

    result = await agent.run("fetch the tags", _context())

    assert result == "done"
    assert llm.index == 4


def _events(agent: Agent) -> list[object]:
    """Everything the agent emitted, via its own bus."""

    collected: list[object] = []

    bus = getattr(agent, "_events", None)

    if bus is None:
        return collected

    for stored in bus.snapshot():
        collected.append(stored.event)

    return collected
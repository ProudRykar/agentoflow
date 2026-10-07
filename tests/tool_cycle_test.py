"""A ping-pong is a loop too.

The existing guard counts each call against itself. A, B, A, B never
reaches any single signature's threshold, so an agent stuck between two
tools walks through it -- while each pass spends a request and pushes
the previous result out of the context window.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from agent_workflow.core.entities.models.agent import Agent
from agent_workflow.core.entities.models.llm import (
    LLMToolCall,
    LLMResponse,
)
from agent_workflow.core.entities.models.llm_client import LLMClient
from agent_workflow.core.entities.models.tool import (
    Tool,
    ToolContext,
    ToolPolicy,
)
from agent_workflow.core.entities.models.tool_executor import ToolExecutor
from agent_workflow.core.entities.models.tool_registry import ToolRegistry


@dataclass
class NoArguments:
    name: str = "x"


class ScriptedToolLLM(LLMClient):
    """Issues the given tool names in order, then answers.

    Answers rather than repeating the last name once the script runs
    out: a fake that calls a tool forever cannot tell a detected cycle
    apart from a run that simply exhausted its iterations.
    """

    def __init__(self, sequence: list[str]) -> None:
        self._sequence = sequence
        self.turn = 0
        self.saw_no_tools = False
        self.prompts: list[list[dict[str, Any]]] = []

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: tuple[Any, ...] = (),
    ) -> LLMResponse:
        self.prompts.append(list(messages))

        if not tools:
            self.saw_no_tools = True

            return LLMResponse(
                content="final answer",
                tool_calls=(),
                raw={},
            )

        if self.turn >= len(self._sequence):
            return LLMResponse(
                content="final answer",
                tool_calls=(),
                raw={},
            )

        name = self._sequence[self.turn]

        self.turn += 1

        return LLMResponse(
            content="",
            tool_calls=(
                LLMToolCall(
                    id=f"call-{self.turn}",
                    name=name,
                    arguments={"name": name},
                ),
            ),
            raw={},
        )


def _agent(
    llm: LLMClient,
    **kwargs: Any,
) -> Agent:
    registry = ToolRegistry()

    async def handler(
        arguments: NoArguments,
        context: ToolContext,
    ) -> str:
        return f"result for {arguments.name}"

    for name in ("alpha", "beta", "gamma"):
        registry.register(
            Tool(
                name=name,
                description=f"tool {name}",
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


def _context(tmp_path: Path) -> ToolContext:
    return ToolContext(
        working_directory=tmp_path,
        environment={},
        allowed_path=(tmp_path,),
        permissions=frozenset(),
    )


@pytest.mark.asyncio
async def test_an_alternating_cycle_is_caught(
    tmp_path: Path,
) -> None:
    """No single signature repeats enough; the pattern does."""

    # Longer than the cycle takes to appear, so detection is what ends
    # the run and not the script simply running out.
    llm = ScriptedToolLLM(
        ["alpha", "beta"] * 5
    )

    agent = _agent(llm)

    result = await agent.run(
        prompt="go",
        context=_context(tmp_path),
    )

    # Tools withdrawn: the loop cannot continue by calling them.
    assert llm.saw_no_tools
    assert result == "final answer"

    # The pattern is only visible once the fourth call has run, so the
    # earliest possible reaction is the request after it -- five calls
    # issued, the sixth request offering nothing.
    assert llm.turn == 5


@pytest.mark.asyncio
async def test_a_normal_progression_is_not_a_cycle(
    tmp_path: Path,
) -> None:
    """Distinct work in sequence is not ping-ponging."""

    llm = ScriptedToolLLM(
        ["alpha", "beta", "gamma", "alpha", "beta"]
    )

    agent = _agent(llm)

    await agent.run(
        prompt="go",
        context=_context(tmp_path),
    )

    # The whole script ran: no bounce was mistaken for a cycle.
    assert llm.turn == 5
    assert not llm.saw_no_tools


@pytest.mark.asyncio
async def test_one_bounce_is_tolerated(
    tmp_path: Path,
) -> None:
    """
    Checking a result and then going back is ordinary work. Two
    consecutive bounces is the signal.
    """

    llm = ScriptedToolLLM(["alpha", "beta", "alpha", "gamma"])

    agent = _agent(llm)

    await agent.run(
        prompt="go",
        context=_context(tmp_path),
    )

    # One bounce back is ordinary work, so tools were never withdrawn.
    assert not llm.saw_no_tools
    assert llm.turn == 4


@pytest.mark.asyncio
async def test_the_model_is_told_which_cycle_it_is_in(
    tmp_path: Path,
) -> None:
    """It cannot see its own pattern; naming it is the whole point."""

    llm = ScriptedToolLLM(
        ["alpha", "beta"] * 5
    )

    agent = _agent(llm)

    await agent.run(
        prompt="go",
        context=_context(tmp_path),
    )

    # The notice reaches the model as a message, which is where an
    # instruction has to arrive to be acted on.
    prompts = "\n".join(
        str(message.get("content"))
        for call in llm.prompts
        for message in call
    )

    assert "TOOL CALL CYCLE DETECTED" in prompts

    # Naming the pattern it cannot see is the point of the notice.
    assert "alpha" in prompts
    assert "beta" in prompts


def test_the_sequence_is_capped() -> None:
    """A longer window eventually matches unrelated earlier work."""

    agent = _agent(ScriptedToolLLM([]))

    for index in range(20):
        agent._note_call_sequence(
            LLMToolCall(
                id=f"c{index}",
                name=f"tool-{index}",
                arguments={},
            ),
        )

    assert len(agent._call_sequence) <= 4


def test_the_counter_resets_at_a_turn_boundary() -> None:
    agent = _agent(ScriptedToolLLM([]))

    for index in range(3):
        agent._note_call_sequence(
            LLMToolCall(
                id=f"c{index}",
                name="alpha",
                arguments={},
            ),
        )

    agent._reset_repeats()

    assert agent._cycles == 0
    assert agent._call_sequence == []

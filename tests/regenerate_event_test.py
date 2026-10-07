"""A retry must be distinguishable from a new question.

The retry deliberately repeats a question that is already on screen,
so a consumer that treats the prompt as a fresh turn shows it twice
and the user is left with one answer and two questions. The flag is
what keeps the transcript honest about what happened.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_workflow.core.entities.models.agent import Agent
from agent_workflow.core.entities.models.agent_trace import AgentStarted
from agent_workflow.core.entities.models.llm import LLMResponse
from agent_workflow.core.entities.models.llm_client import LLMClient
from agent_workflow.core.entities.models.tool import ToolContext
from agent_workflow.core.entities.models.tool_executor import ToolExecutor
from agent_workflow.core.entities.models.tool_registry import ToolRegistry
from agent_workflow.web.serialization import event_type


class ScriptedLLM(LLMClient):
    def __init__(self, answers: tuple[str, ...]) -> None:
        self._answers = iter(answers)

    async def chat(self, messages, tools=()) -> LLMResponse:
        return LLMResponse(
            content=next(self._answers),
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


def _agent(answers: tuple[str, ...]) -> Agent:
    registry = ToolRegistry()

    return Agent(
        llm=ScriptedLLM(answers),
        registry=registry,
        executor=ToolExecutor(registry),
    )


def _started(
    events: list[object],
) -> list[AgentStarted]:
    return [
        event
        for event in events
        if isinstance(event, AgentStarted)
    ]


@pytest.mark.asyncio
async def test_first_run_is_not_marked_a_retry(
    tmp_path: Path,
) -> None:
    events: list[object] = []

    async def collect(event: object) -> None:
        events.append(event)

    agent = _agent(("Paris.",))

    await agent.run(
        prompt="Capital of France?",
        context=_context(tmp_path),
        on_event=collect,
    )

    started = _started(events)

    assert len(started) == 1
    assert started[0].regenerated is False
    assert event_type(started[0]) == "agent.started"


@pytest.mark.asyncio
async def test_retry_repeats_the_question_and_says_so(
    tmp_path: Path,
) -> None:
    events: list[object] = []

    async def collect(event: object) -> None:
        events.append(event)

    agent = _agent(("Paris.", "Lyon."))

    await agent.run(
        prompt="Capital of France?",
        context=_context(tmp_path),
        on_event=collect,
    )

    await agent.regenerate(
        context=_context(tmp_path),
        on_event=collect,
    )

    started = _started(events)

    assert len(started) == 2

    # The prompt repeats a turn that is already in the transcript, so
    # the flag is the only thing telling the two apart.
    assert started[1].prompt == "Capital of France?"
    assert started[1].regenerated is True


@pytest.mark.asyncio
async def test_hint_is_a_separate_user_turn(
    tmp_path: Path,
) -> None:
    events: list[object] = []

    async def collect(event: object) -> None:
        events.append(event)

    agent = _agent(("Paris.", "Lyon."))

    await agent.run(
        prompt="Capital of France?",
        context=_context(tmp_path),
        on_event=collect,
    )

    await agent.regenerate(
        context=_context(tmp_path),
        on_event=collect,
        hint="be shorter",
    )

    started = _started(events)

    # The hint is a genuine new turn, so it is its own prompt rather
    # than replacing the question being retried.
    assert started[1].prompt == "Capital of France?"

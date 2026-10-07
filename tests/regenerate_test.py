"""Answering the same question again must not accumulate the old one.

A discarded answer is still in the conversation when regenerate
merely appends: the model then treats it as context to stay
consistent with, and argues with itself instead of trying again. And
with the answer never stored at all, "why?" has nothing to refer to.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from agent_workflow.core.entities.models.agent import Agent
from agent_workflow.core.entities.models.context_manager import (
    ContextManager,
    ContextPolicy,
)
from agent_workflow.core.entities.models.llm import LLMResponse
from agent_workflow.core.entities.models.llm_client import LLMClient
from agent_workflow.core.entities.models.tool import ToolContext
from agent_workflow.core.entities.models.tool_executor import ToolExecutor
from agent_workflow.core.entities.models.tool_registry import ToolRegistry


@dataclass
class NoArguments:
    pass


class ScriptedLLM(LLMClient):
    """Answers in order, and remembers what it was shown."""

    def __init__(
        self,
        answers: Sequence[str],
    ) -> None:
        self._answers = iter(answers)
        self.seen: list[list[dict[str, Any]]] = []

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: tuple[Any, ...] = (),
    ) -> LLMResponse:
        self.seen.append(list(messages))

        return LLMResponse(
            content=next(self._answers),
            tool_calls=(),
            raw={},
        )

    def dialogue_seen(
        self,
        call: int,
    ) -> list[tuple[str, str]]:
        return [
            (
                str(message.get("role")),
                str(message.get("content")),
            )
            for message in self.seen[call]
            if message.get("role") in ("user", "assistant")
        ]


def _tool_context(
    tmp_path: Path,
) -> ToolContext:
    return ToolContext(
        working_directory=tmp_path,
        environment={},
        allowed_path=(tmp_path,),
        permissions=frozenset(),
    )


def _agent(
    llm: LLMClient,
) -> Agent:
    registry = ToolRegistry()

    return Agent(
        llm=llm,
        registry=registry,
        executor=ToolExecutor(registry),
    )


@pytest.mark.asyncio
async def test_regenerate_replaces_the_answer_not_appends_it(
    tmp_path: Path,
) -> None:
    llm = ScriptedLLM(["Paris.", "Lyon."])

    agent = _agent(llm)

    assert await agent.run(
        prompt="Capital of France?",
        context=_tool_context(tmp_path),
    ) == "Paris."

    assert await agent.regenerate(
        context=_tool_context(tmp_path),
    ) == "Lyon."

    roles = [
        role
        for role, _ in llm.dialogue_seen(1)
    ]

    # The retry saw the question and nothing else: no "Paris." to
    # stay consistent with.
    assert roles == ["user"]

    assert [
        (role, content)
        for role, content in llm.dialogue_seen(1)
        if role == "user"
    ] == [("user", "Capital of France?")]


@pytest.mark.asyncio
async def test_kept_answer_is_visible_to_the_next_question(
    tmp_path: Path,
) -> None:
    """The defect regenerate exposed: an answer nothing could follow."""

    llm = ScriptedLLM(["Paris.", "Because of history.", "Ile-de-France."])

    agent = _agent(llm)

    await agent.run(
        prompt="Capital of France?",
        context=_tool_context(tmp_path),
    )

    await agent.continue_run(
        prompt="Why?",
        context=_tool_context(tmp_path),
    )

    assert ("assistant", "Paris.") in llm.dialogue_seen(1)

    await agent.continue_run(
        prompt="And its region?",
        context=_tool_context(tmp_path),
    )

    assert (
        "assistant",
        "Because of history.",
    ) in llm.dialogue_seen(2)


@pytest.mark.asyncio
async def test_hint_becomes_a_turn_not_an_injection(
    tmp_path: Path,
) -> None:
    llm = ScriptedLLM(["Long answer.", "Short answer."])

    agent = _agent(llm)

    await agent.run(
        prompt="Explain it",
        context=_tool_context(tmp_path),
    )

    assert await agent.regenerate(
        context=_tool_context(tmp_path),
        hint="be shorter",
    ) == "Short answer."

    # Recorded as the user's turn, so it takes part in context
    # selection later instead of being a one-off injection that later
    # turns know nothing about.
    assert ("user", "be shorter") in llm.dialogue_seen(1)

    assert [
        message["content"]
        for message in agent.context_manager.dialogue()
        if message.get("role") == "user"
    ] == ["Explain it", "be shorter"]


@pytest.mark.asyncio
async def test_refuses_when_the_user_spoke_last(
    tmp_path: Path,
) -> None:
    """Silently deleting the user's message would be worse than saying no."""

    llm = ScriptedLLM(["Paris.", "Lyon."])

    agent = _agent(llm)

    await agent.run(
        prompt="Capital of France?",
        context=_tool_context(tmp_path),
    )

    agent.context_manager.add_user_message("Actually, make it two words")

    with pytest.raises(RuntimeError, match="Nothing to regenerate"):
        await agent.regenerate(context=_tool_context(tmp_path))

    # Nothing was dropped on the way to refusing.
    assert [
        message["content"]
        for message in agent.context_manager.dialogue()
        if message.get("role") == "user"
    ] == ["Capital of France?", "Actually, make it two words"]


def test_tool_round_travels_with_the_discarded_answer() -> None:
    """A tool result without its call is rejected by the provider."""

    context_manager = ContextManager(policy=ContextPolicy())

    context_manager.create("question")
    context_manager.add_assistant_message(
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "call-1"}],
        },
    )
    context_manager.add_tool_result(
        {
            "role": "tool",
            "content": "result",
            "tool_call_id": "call-1",
        },
    )
    context_manager.add_assistant_message(
        {
            "role": "assistant",
            "content": "answer",
        },
    )

    assert context_manager.has_trailing_assistant()
    assert context_manager.drop_trailing_assistant() == 3

    assert [
        message.get("role")
        for message in context_manager.dialogue()
    ] == ["user"]


def test_has_trailing_assistant_does_not_consume_the_answer() -> None:
    """Asking by dropping would take the conversation with it."""

    context_manager = ContextManager(policy=ContextPolicy())

    context_manager.create("question")
    context_manager.add_assistant_message(
        {
            "role": "assistant",
            "content": "answer",
        },
    )

    assert context_manager.has_trailing_assistant()
    assert context_manager.has_trailing_assistant()

    assert context_manager.drop_trailing_assistant() == 1
    assert not context_manager.has_trailing_assistant()


def test_nothing_to_regenerate_on_a_fresh_conversation() -> None:
    context_manager = ContextManager(policy=ContextPolicy())

    context_manager.create("question")

    assert not context_manager.has_trailing_assistant()
    assert context_manager.drop_trailing_assistant() == 0

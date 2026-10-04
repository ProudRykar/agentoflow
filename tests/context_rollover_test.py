"""Rollover must not throw away the user's instruction.

``rollover`` claimed to keep "the trailing user message", but it is
called at the top of an iteration, where the last dialogue message
is a tool result. The condition never held, so the whole window --
including the original prompt -- was cleared, and the restored
history selection excluded tool traffic entirely.
"""

from __future__ import annotations

from typing import Any

import pytest

from agent_workflow.core.context.context_budget import ContextBudget
from agent_workflow.core.context.context_controller import (
    ContextController,
)
from agent_workflow.core.context.evidence import Evidence
from agent_workflow.core.context.history import HistoryKind
from agent_workflow.core.context.task_anchor import TaskAnchor
from agent_workflow.core.entities.models.context_manager import (
    ContextManager,
)
from agent_workflow.core.entities.models.context_policy import (
    ContextPolicy,
)
from agent_workflow.core.entities.models.agent_orchestrator import (
    AgentOrchestrator,
)
from agent_workflow.core.entities.models.planner import Planner
from agent_workflow.core.entities.models.task_contract import TaskContract


def _orchestrator() -> AgentOrchestrator:
    orchestrator = AgentOrchestrator()
    orchestrator.prepare_task(
        Planner().plan("Read the report"),
        TaskContract(),
        prompt="Read the report",
    )
    orchestrator.on_agent_started("Read the report")

    return orchestrator


def _conversation() -> ContextManager:
    return ContextManager(ContextPolicy(max_messages=None))


def _tool_call(call_id: str) -> dict[str, Any]:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": "read_file", "arguments": {}},
    }


def _tool_result(call_id: str, text: str) -> dict[str, Any]:
    return {"role": "tool", "tool_call_id": call_id, "content": text}


@pytest.mark.anyio
async def test_rollover_keeps_the_original_instruction() -> None:
    orchestrator = _orchestrator()
    controller = ContextController()
    conversation = _conversation()

    conversation.create("Read the report and summarise it")
    conversation.add_message(
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [_tool_call("call_1")],
        }
    )
    conversation.add_tool_result(
        _tool_result("call_1", "the file body")
    )

    controller.rollover(orchestrator, conversation)

    roles = [
        message.get("role") for message in conversation.dialogue()
    ]
    contents = [
        str(message.get("content", ""))
        for message in conversation.dialogue()
    ]

    # Regression: keep was [] because the last message was a tool
    # result, so the user's prompt was discarded too.
    assert "user" in roles
    assert any(
        "Read the report" in content for content in contents
    )


@pytest.mark.anyio
async def test_rollover_keeps_tool_results_in_history() -> None:
    # Regression: _select_restore_context only accepted user and
    # assistant messages, so every tool result was dropped on the
    # floor at rollover.
    orchestrator = _orchestrator()
    controller = ContextController()
    conversation = _conversation()

    conversation.create("Read the report")

    task_id = orchestrator.task_anchor.task_id

    # The agent records tool traffic into history as it runs; the
    # controller is what has to read it back.
    controller.record(
        task_id=task_id,
        run_id="run-1",
        kind=HistoryKind.TOOL_RESULT,
        content="IMPORTANT FACT 4711",
        reference="call_1",
    )

    result = controller.rollover(orchestrator, conversation)

    restored = " ".join(
        item.content for item in result.history_context
    )

    assert "IMPORTANT FACT 4711" in restored


@pytest.mark.anyio
async def test_rollover_never_leaves_an_orphan_tool_message() -> None:
    orchestrator = _orchestrator()
    controller = ContextController()
    conversation = _conversation()

    conversation.create("Read the report")
    conversation.add_message(
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [_tool_call("call_1")],
        }
    )
    conversation.add_tool_result(_tool_result("call_1", "body"))

    controller.rollover(orchestrator, conversation)

    for index, message in enumerate(conversation.dialogue()):
        if message.get("role") == "tool":
            preceding = conversation.dialogue()[index - 1]

            assert preceding.get("role") == "assistant"
            assert preceding.get("tool_calls")


@pytest.mark.anyio
async def test_rollover_survives_a_restored_session() -> None:
    # A restored session has no live task anchor, so save_checkpoint
    # has nothing to work from. Rollover must degrade, not raise and
    # kill the run.
    controller = ContextController()
    orchestrator = AgentOrchestrator()
    conversation = _conversation()

    conversation.create("hello")

    result = controller.rollover(orchestrator, conversation)

    # No checkpoint was possible, so nothing was cleared.
    assert result.performed is False
    assert result.checkpoint is None
    assert result.cleared_messages == 0
    assert result.window_id == result.previous_window_id

    assert [
        message.get("role") for message in conversation.dialogue()
    ] == ["user"]


@pytest.mark.anyio
async def test_trimmed_history_prefers_recent_items() -> None:
    controller = ContextController()
    orchestrator = _orchestrator()

    for index in range(30):
        controller.record(
            task_id=orchestrator.task_anchor.task_id,
            run_id="run-1",
            kind=HistoryKind.ASSISTANT_MESSAGE,
            content=f"turn {index}",
            reference="window-01",
        )

    controller.new_window()

    result = controller.rollover(orchestrator, _conversation())

    contents = [item.content for item in result.history_context]

    # Bounded, and biased to the newest turns.
    assert 0 < len(contents) <= 5
    assert contents[-1] == "turn 29"


def test_budget_available_is_respected_by_the_controller() -> None:
    from agent_workflow.core.context.llm_request import LLMRequestContext

    controller = ContextController(
        budget=ContextBudget(
            maximum_tokens=1_000,
            reserved_system=100,
            reserved_task=100,
            reserved_output=100,
        )
    )

    tiny = LLMRequestContext(
        messages=({"role": "system", "content": "small"},),
    )

    assert controller.is_over_budget(tiny) is False

def test_trim_never_leaves_an_orphan_tool_message() -> None:
    """A tool result without its assistant block is invalid input."""

    from agent_workflow.core.entities.models.context_policy import (
        ContextPolicy as _Policy,
    )

    conversation = ContextManager(_Policy(max_messages=2))

    conversation.create("read the report")
    conversation.add_message(
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [_tool_call("call_1")],
        }
    )
    conversation.add_tool_result(_tool_result("call_1", "body"))

    roles = [
        message.get("role") for message in conversation.dialogue()
    ]

    # Whichever way the trim landed, no tool may lead the window.
    assert not roles or roles[0] != "tool" or roles[0] == "system"
    assert "tool" not in roles or "assistant" in roles

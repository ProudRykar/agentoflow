from typing import Any

import pytest

from core.context.context_budget import ContextBudget
from core.context.context_controller import ContextController
from core.context.history import HistoryKind
from core.context.llm_request import LLMRequestContext
from core.entities.models.agent_orchestrator import AgentOrchestrator
from core.entities.models.context_manager import ContextManager
from core.entities.models.planner import Planner
from core.entities.models.task_contract import TaskContract


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


def _request_with_size(
    chars: int,
) -> LLMRequestContext:
    return LLMRequestContext(
        messages=(
            {
                "role": "system",
                "content": "x" * chars,
            },
        ),
    )


def test_windows_start_and_increment() -> None:
    controller = ContextController()

    assert controller.current_window_id == "window-01"
    assert controller.new_window() == "window-01"
    assert controller.new_window() == "window-02"
    assert controller.current_window_id == "window-02"


def test_save_and_restore_checkpoint() -> None:
    orchestrator = _started_orchestrator()
    controller = ContextController()

    checkpoint = controller.save_checkpoint(orchestrator)

    assert checkpoint.objective == "test.txt"
    assert (
        controller.restore_checkpoint(checkpoint.task_id)
        == checkpoint
    )


def test_save_without_task_is_forbidden() -> None:
    controller = ContextController()

    with pytest.raises(RuntimeError, match="new context is forbidden"):
        controller.save_checkpoint(AgentOrchestrator())


def test_restore_missing_is_blocked() -> None:
    controller = ContextController()

    with pytest.raises(RuntimeError, match="run is blocked"):
        controller.restore_checkpoint("task-missing")


def test_budget_gate() -> None:
    controller = ContextController(
        budget=ContextBudget(
            maximum_tokens=1000,
            reserved_system=10,
            reserved_task=10,
            reserved_output=100,
        ),
    )

    assert not controller.is_over_budget(
        _request_with_size(100)
    )
    assert controller.is_over_budget(
        _request_with_size(10_000)
    )


def test_record_uses_current_window() -> None:
    controller = ContextController()
    controller.new_window()

    item = controller.record(
        task_id="task-01",
        run_id="run-01",
        kind=HistoryKind.USER_MESSAGE,
        content="hello",
    )

    assert item.window_id == "window-01"
    assert (
        controller.history.window_items(
            "task-01", "window-01"
        )
        == (item,)
    )


def test_rollover_clears_but_keeps_current_instruction() -> None:
    orchestrator = _started_orchestrator()
    controller = ContextController()
    # Agent.run() opens a window before the dialogue starts.
    controller.new_window()

    conversation = ContextManager()
    conversation.create("Read test.txt")
    conversation.add_message(
        {"role": "assistant", "content": "working"}
    )
    conversation.add_message(
        {"role": "user", "content": "continue now"}
    )

    result = controller.rollover(
        orchestrator,
        conversation,
    )

    assert result.previous_window_id == "window-01"
    assert result.window_id == "window-02"
    assert result.cleared_messages == 2
    assert conversation.dialogue() == [
        {"role": "user", "content": "continue now"}
    ]

    kinds = [
        item.kind
        for item in controller.history.task_items(
            result.checkpoint.task_id
        )
    ]

    assert HistoryKind.CHECKPOINT in kinds


def test_rollover_restores_user_context_once() -> None:
    orchestrator = _started_orchestrator()
    controller = ContextController()

    task_id = orchestrator.task_anchor.task_id

    controller.record(
        task_id=task_id,
        run_id="",
        kind=HistoryKind.USER_MESSAGE,
        content="original question",
    )

    conversation = ContextManager()
    conversation.create("follow-up")

    result = controller.rollover(
        orchestrator,
        conversation,
    )

    assert len(result.history_context) == 1
    assert (
        result.history_context[0].content
        == "original question"
    )
    assert result.history_context[0].reference == "window-01"


def test_rollover_without_task_forbidden_and_untouched() -> None:
    controller = ContextController()
    conversation = ContextManager()
    conversation.create("hello")

    with pytest.raises(RuntimeError):
        controller.rollover(
            AgentOrchestrator(),
            conversation,
        )

    assert conversation.dialogue() == [
        {"role": "user", "content": "hello"}
    ]


def test_rollover_clears_fully_without_trailing_user() -> None:
    orchestrator = _started_orchestrator()
    controller = ContextController()
    conversation = ContextManager()
    conversation.create("Read test.txt")
    conversation.add_message(
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": "body",
        }
    )

    result = controller.rollover(
        orchestrator,
        conversation,
    )

    assert result.cleared_messages == 2
    assert conversation.dialogue() == []

    # Only the harness system message remains.
    assert len(conversation.messages()) == 1

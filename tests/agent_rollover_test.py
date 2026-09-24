from pathlib import Path
from typing import Any

import pytest

from core.context.context_assembler import ContextAssembler
from core.context.context_budget import ContextBudget
from core.context.context_controller import ContextController
from core.context.history import HistoryKind
from core.context.tokens import ApproximateTokenCounter
from core.entities.models.agent import Agent
from core.entities.models.agent_orchestrator import AgentOrchestrator
from core.entities.models.builtin.read_file import (
    ReadFileInput,
    read_file,
)
from core.entities.models.llm import LLMResponse, LLMToolCall
from core.entities.models.llm_client import LLMClient
from core.entities.models.tool import Tool, ToolContext, ToolPolicy
from core.entities.models.tool_executor import ToolExecutor
from core.entities.models.tool_registry import ToolRegistry


class OneToolThenDoneLLM(LLMClient):
    def __init__(self) -> None:
        self.calls = 0

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: tuple[Any, ...] = (),
    ) -> LLMResponse:
        self.calls += 1

        if self.calls == 1:
            return LLMResponse(
                content="",
                tool_calls=(
                    LLMToolCall(
                        id="call_1",
                        name="read_file",
                        arguments={"path": "test.txt"},
                    ),
                ),
                raw={},
            )

        return LLMResponse(
            content="done",
            tool_calls=(),
            raw={},
        )


def _make_agent(
    tmp_path: Path,
    budget: ContextBudget | None = None,
) -> tuple[Agent, ToolContext]:
    (tmp_path / "test.txt").write_text(
        "Hello, agent!",
        encoding="utf-8",
    )

    async def handler(
        arguments: ReadFileInput,
        context: ToolContext,
    ) -> str:
        return await read_file(arguments, context)

    registry = ToolRegistry()
    registry.register(
        Tool(
            name="read_file",
            description="Read a UTF-8 text file",
            input_type=ReadFileInput,
            handler=handler,
            policy=ToolPolicy(
                permissions=frozenset({"filesystem.read"}),
                timeout=1.0,
                max_output_size=100_000,
            ),
        )
    )

    kwargs: dict[str, Any] = {}

    if budget is not None:
        counter = ApproximateTokenCounter()
        kwargs["assembler"] = ContextAssembler(
            budget=budget,
            counter=counter,
        )
        kwargs["controller"] = ContextController(
            budget=budget,
            counter=counter,
        )

    agent = Agent(
        llm=OneToolThenDoneLLM(),
        registry=registry,
        executor=ToolExecutor(registry),
        **kwargs,
    )

    context = ToolContext(
        working_directory=tmp_path,
        environment={},
        allowed_path=(tmp_path,),
        permissions=frozenset({"filesystem.read"}),
    )

    return agent, context


@pytest.mark.asyncio
async def test_run_records_history_events(tmp_path: Path) -> None:
    agent, context = _make_agent(tmp_path)

    result = await agent.run(
        prompt="Read test.txt",
        context=context,
    )

    assert result == "done"

    task_id = agent.task_anchor.task_id
    kinds = [
        item.kind
        for item in agent.controller.history.task_items(task_id)
    ]

    assert HistoryKind.USER_MESSAGE in kinds
    assert HistoryKind.TOOL_CALL in kinds
    assert HistoryKind.TOOL_RESULT in kinds
    assert HistoryKind.PHASE_CHANGE in kinds
    assert HistoryKind.ASSISTANT_MESSAGE in kinds


@pytest.mark.asyncio
async def test_tiny_budget_triggers_rollover(tmp_path: Path) -> None:
    budget = ContextBudget(
        maximum_tokens=500,
        reserved_system=10,
        reserved_task=10,
        reserved_output=50,
    )

    agent, context = _make_agent(tmp_path, budget=budget)

    result = await agent.run(
        prompt="Read test.txt",
        context=context,
    )

    # The task still completes despite rollovers.
    assert result == "done"

    task_id = agent.task_anchor.task_id

    # A checkpoint was proven retrievable before clearing.
    checkpoint = agent.controller.restore_checkpoint(task_id)

    assert checkpoint.objective == agent.task_anchor.objective

    # A new window was opened.
    assert agent.controller.current_window_id != "window-01"

    kinds = [
        item.kind
        for item in agent.controller.history.task_items(task_id)
    ]

    assert HistoryKind.CHECKPOINT in kinds

"""The checklist belongs to the model, not to the planner.

The planner parses, so all it can produce is phase-shaped steps -- a
restatement of the harness rather than a decomposition of the task.
Real subtasks come from the thing that read the task and decided what
it was going to do, which is why the list is written through a tool
while the run is in progress.

Two vocabularies, deliberately not merged: the AgentPlan keeps its
invariants so completion stays decidable, and the todo list carries no
invariants so the model can keep it true.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from agent_workflow.core.entities.models.agent import Agent
from agent_workflow.core.entities.models.agent_orchestrator import (
    AgentOrchestrator,
)
from agent_workflow.core.entities.models.builtin.memory_registry import (
    create_todo_tools,
)
from agent_workflow.core.entities.models.builtin.todo_tools import (
    TodoReadInput,
    TodoWriteInput,
    todoread,
    todowrite,
)
from agent_workflow.core.entities.models.llm import LLMResponse
from agent_workflow.core.entities.models.llm_client import LLMClient
from agent_workflow.core.entities.models.planner import Planner
from agent_workflow.core.entities.models.task_contract import TaskContract
from agent_workflow.core.entities.models.todo_list import (
    AgentPlanTodoItem,
    TodoList,
    TodoStatus,
)
from agent_workflow.core.entities.models.tool import ToolContext
from agent_workflow.core.entities.models.tool_executor import ToolExecutor
from agent_workflow.core.entities.models.tool_registry import ToolRegistry


# ======================================================================
# The list
# ======================================================================


def test_it_renders_as_a_checklist() -> None:
    todos = TodoList()

    todos.replace(
        (
            AgentPlanTodoItem(
                content="Read the config",
                status=TodoStatus.COMPLETED,
            ),
            AgentPlanTodoItem(
                content="Fetch the list",
                status=TodoStatus.IN_PROGRESS,
            ),
            AgentPlanTodoItem(content="Write it up"),
        ),
    )

    rendered = todos.render()

    assert rendered.startswith("▼Todo")
    assert "[✓] Read the config" in rendered
    assert "[•] Fetch the list" in rendered
    assert "[○] Write it up" in rendered


def test_progress_is_counted_per_item() -> None:
    """A bar that can read as "done" invites marking everything done."""

    todos = TodoList()

    todos.replace(
        (
            AgentPlanTodoItem(content="a"),
            AgentPlanTodoItem(
                content="b",
                status=TodoStatus.COMPLETED,
            ),
        ),
    )

    assert "(1/2)" in todos.render()


def test_an_empty_item_is_rejected() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        AgentPlanTodoItem(content="   ")


def test_an_absurdly_long_item_is_rejected() -> None:
    with pytest.raises(ValueError, match="characters"):
        AgentPlanTodoItem(content="x" * 500)


def test_content_is_trimmed() -> None:
    item = AgentPlanTodoItem(content="  do it  ")

    assert item.content == "do it"


def test_the_active_item_is_found() -> None:
    todos = TodoList()

    todos.replace(
        (
            AgentPlanTodoItem(content="a"),
            AgentPlanTodoItem(
                content="b",
                status=TodoStatus.IN_PROGRESS,
            ),
        ),
    )

    assert todos.active is not None
    assert todos.active.content == "b"


def test_an_overlong_list_is_trimmed_rather_than_refused() -> None:
    """Refusing would leave a lost model writing ever longer lists."""

    todos = TodoList()

    todos.replace(
        tuple(
            AgentPlanTodoItem(content=f"item {index}")
            for index in range(200)
        ),
    )

    assert len(todos.items) == 60


def test_the_planner_seeds_it_once() -> None:
    orchestrator = AgentOrchestrator()
    orchestrator.prepare_task(
        Planner().plan("Напиши тест для парсера"),
        TaskContract(),
        prompt="Напиши тест для парсера",
    )
    orchestrator.on_agent_started("Напиши тест для парсера")
    orchestrator.initialize_plan()

    assert not orchestrator.todo_list.is_empty()

    orchestrator.todo_list.replace(
        (AgentPlanTodoItem(content="my own list"),),
    )

    # A second plan must not discard what the model decided.
    orchestrator.initialize_plan()
    orchestrator._todo_list.seed(orchestrator._plan.steps)

    assert orchestrator.todo_list.items[0].content == "my own list"


# ======================================================================
# The tool
# ======================================================================


def _agent_with_todos(
) -> tuple[Agent, ToolContext, AgentOrchestrator]:
    registry = ToolRegistry()

    for tool in create_todo_tools():
        registry.register(tool)

    agent = Agent(
        llm=_SilentLLM(),
        registry=registry,
        executor=ToolExecutor(registry),
    )

    orchestrator = agent.orchestrator

    orchestrator.prepare_task(
        Planner().plan("summarise the stash"),
        TaskContract(),
        prompt="summarise the stash",
    )
    orchestrator.on_agent_started("summarise the stash")
    orchestrator.initialize_plan()

    context = ToolContext(
        working_directory=Path(tempfile.mkdtemp()),
        environment={},
        allowed_path=(Path(tempfile.mkdtemp()),),
        permissions=frozenset({"plan.write", "plan.read"}),
    )

    context.agent = agent

    return agent, context, orchestrator


class _SilentLLM(LLMClient):
    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: tuple[Any, ...] = (),
    ) -> LLMResponse:
        return LLMResponse(
            content="done",
            tool_calls=(),
            raw={},
        )


@pytest.mark.asyncio
async def test_the_tool_replaces_the_whole_list() -> None:
    _, context, orchestrator = _agent_with_todos()

    result = await todowrite(
        TodoWriteInput(
            todos=(
                AgentPlanTodoItem(
                    content="Read the config",
                    status=TodoStatus.COMPLETED,
                ),
                AgentPlanTodoItem(
                    content="Fetch the list",
                    status=TodoStatus.IN_PROGRESS,
                ),
            ),
        ),
        context,
    )

    assert "Read the config" in result
    assert len(orchestrator.todo_list.items) == 2

    # The planner's seed is gone: the model's list replaced it.
    assert "Выполнить задачу" not in result


@pytest.mark.asyncio
async def test_omitting_an_item_removes_it() -> None:
    """The contract promises replacement, not merging."""

    _, context, orchestrator = _agent_with_todos()

    await todowrite(
        TodoWriteInput(
            todos=(
                AgentPlanTodoItem(content="one"),
                AgentPlanTodoItem(content="two"),
                AgentPlanTodoItem(content="three"),
            ),
        ),
        context,
    )

    await todowrite(
        TodoWriteInput(
            todos=(AgentPlanTodoItem(content="one"),),
        ),
        context,
    )

    assert len(orchestrator.todo_list.items) == 1


@pytest.mark.asyncio
async def test_an_empty_write_is_refused() -> None:
    """Otherwise a confused model erases the list with nothing to replace it."""

    _, context, orchestrator = _agent_with_todos()

    before = len(orchestrator.todo_list.items)

    result = await todowrite(
        TodoWriteInput(todos=()),
        context,
    )

    assert "Refused" in result
    assert len(orchestrator.todo_list.items) == before


@pytest.mark.asyncio
async def test_it_reads_the_list_back() -> None:
    _, context, orchestrator = _agent_with_todos()

    await todowrite(
        TodoWriteInput(
            todos=(AgentPlanTodoItem(content="one"),),
        ),
        context,
    )

    assert "one" in await todoread(TodoReadInput(), context)


@pytest.mark.asyncio
async def test_without_a_task_there_is_nothing_to_keep() -> None:
    context = ToolContext(
        working_directory=Path(tempfile.mkdtemp()),
        environment={},
        allowed_path=(Path(tempfile.mkdtemp()),),
        permissions=frozenset({"plan.write"}),
    )

    result = await todowrite(
        TodoWriteInput(
            todos=(AgentPlanTodoItem(content="one"),),
        ),
        context,
    )

    assert "No task is active" in result


def test_the_tools_are_registered_with_permissions() -> None:
    names = {
        tool.name: tool
        for tool in create_todo_tools()
    }

    assert "todowrite" in names
    assert "todoread" in names

    assert "plan.write" in names["todowrite"].policy.permissions
    assert "plan.read" in names["todoread"].policy.permissions

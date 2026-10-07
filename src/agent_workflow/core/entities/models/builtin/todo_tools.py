"""Let the model own the checklist.

The planner can only derive what it can parse: with no LLM it produces
phase-shaped steps ("research", "execution"), which is a restatement of
the harness rather than a decomposition of the task. Real subtasks come
from the thing that read the task and decided what it was going to do.

So the list is written by the model, through a tool, while it works --
which is also what makes the ticks mean anything. A checklist assembled
in advance by regex and then ticked by the runtime records the runtime's
guess, not what the agent actually did.

The planner still seeds the list. A model asked to plan from nothing
invents a generic one; a model shown the task anchor and the contract's
requirements writes a better first draft. The seed is a floor, not the
answer.
"""

from __future__ import annotations

from dataclasses import dataclass

from agent_workflow.core.entities.models.todo_list import (
    AgentPlanTodoItem,
    TodoList,
)
from agent_workflow.core.entities.models.tool import ToolContext

TODOWRITE_DESCRIPTION = (
    "Write or update the task checklist.\n\n"
    "Pass the FULL list every time -- items you leave out are "
    "removed, not kept. Each item is one concrete action with a "
    "verifiable end state, not a phase or a topic.\n\n"
    "status:\n"
    "  pending   -- not started\n"
    "  in_progress -- being worked on now; at most one at a time\n"
    "  completed -- finished and checked\n"
    "  cancelled -- deliberately dropped, with a reason in the text\n\n"
    "Call this again when the plan changes, not on every step. Add an "
    "item as soon as you discover work that was not on the list."
)

TODOREAD_DESCRIPTION = (
    "Read the current task checklist."
)


@dataclass(slots=True, frozen=True)
class TodoWriteInput:
    todos: tuple[AgentPlanTodoItem, ...]
    """The complete list. Omitted items are removed."""


@dataclass(slots=True, frozen=True)
class TodoReadInput:
    pass


def _todo_list_of(
    context: ToolContext,
) -> TodoList | None:
    agent = context.agent

    if agent is None:
        return None

    orchestrator = getattr(agent, "orchestrator", None)

    if orchestrator is None:
        return None

    return getattr(orchestrator, "todo_list", None)


async def todowrite(
    arguments: TodoWriteInput,
    context: ToolContext,
) -> str:
    todos = _todo_list_of(context)

    if todos is None:
        return (
            "No task is active, so there is nothing to keep a "
            "checklist against."
        )

    if not arguments.todos:
        return (
            "Refused: an empty list would erase the checklist with "
            "nothing to replace it. Cancel items you no longer want "
            "instead."
        )

    rejected = [
        item.content
        for item in arguments.todos
        if not item.content.strip()
    ]

    if rejected:
        return (
            f"Refused: {len(rejected)} item(s) had empty text. Every "
            "item needs a description, since that is what the user "
            "reads."
        )

    todos.replace(arguments.todos)

    return todos.render()


async def todoread(
    arguments: TodoReadInput,
    context: ToolContext,
) -> str:
    todos = _todo_list_of(context)

    if todos is None:
        return "No task is active."

    return todos.render()

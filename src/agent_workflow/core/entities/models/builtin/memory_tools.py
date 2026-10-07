"""Tools for writing and reading what the agent has worked out.

The retrieval block exists, but a model cannot use a mechanism it
cannot ask questions of. ``recall`` needs an exact key, which means the
model has to remember the key it chose several turns ago -- and when it
does not, the notes it carefully wrote are unreachable. So there is a
second tool that takes a topic and returns whatever matches, which is
the question a person would actually ask.

Writes are scoped to the running task unless the model asks for a
global note, and reads are limited to what the task may see. That is
not decoration: without it a note about one game is handed to an
unrelated request, and a model that has seen one irrelevant note
stops trusting the block.
"""

from __future__ import annotations

from dataclasses import dataclass

from agent_workflow.core.context.memory import (
    retrieve_snapshot,
    visible_entries,
)
from agent_workflow.core.entities.models.memory import GLOBAL_SCOPE
from agent_workflow.core.entities.models.memory_manager import (
    MemoryManager,
)
from agent_workflow.core.entities.models.tool import ToolContext


@dataclass(slots=True, frozen=True)
class RememberInput:
    key: str
    value: str
    scope: str = "task"
    """``task`` (default), ``global``, or a named scope."""

    tags: tuple[str, ...] = ()


@dataclass(slots=True, frozen=True)
class RecallInput:
    key: str


@dataclass(slots=True, frozen=True)
class RecallMatchingInput:
    topic: str
    limit: int = 5


def task_id_of(context: ToolContext) -> str:
    """The running task's id, or the global scope when there is none.

    Falls back to the global scope rather than raising: a memory tool
    that refuses to work outside a task would be a strange thing to
    discover, and the fallback is the safe direction anyway -- a note
    written with no task in sight is visible everywhere, which is what
    an unscoped caller asked for.
    """

    agent = context.agent

    if agent is None:
        return GLOBAL_SCOPE

    anchor = getattr(agent, "task_anchor", None)

    task_id = getattr(anchor, "task_id", None)

    if not isinstance(task_id, str) or not task_id:
        return GLOBAL_SCOPE

    return task_id


def resolve_scope(
    requested: str,
    context: ToolContext,
) -> str:
    """Turn the model's word for a scope into an identifier."""

    current = task_id_of(context)

    value = requested.strip().lower()

    if value in ("", "task", "current", "local", "session"):
        return current

    if value in ("global", "all", "persistent", "forever"):
        return GLOBAL_SCOPE

    # An explicit name is taken as-is, so two tasks can share a named
    # board without either owning it.
    return requested.strip()


def create_memory_handlers(
    memory: MemoryManager,
):
    async def remember(
        arguments: RememberInput,
        context: ToolContext,
    ) -> str:
        scope = resolve_scope(arguments.scope, context)

        entry = await memory.remember(
            key=arguments.key.strip(),
            value=arguments.value,
            scope=scope,
            tags=arguments.tags,
        )

        where = (
            "this task"
            if entry.scope != GLOBAL_SCOPE
            else "global"
        )

        return (
            f"Saved to {where} memory: '{entry.key}' = "
            f"'{entry.value}'"
        )

    async def recall(
        arguments: RecallInput,
        context: ToolContext,
    ) -> str:
        scope = task_id_of(context)

        # Both scopes are tried: a note promoted to global is no longer
        # in the task's own scope, and reporting "not found" for a
        # note the model wrote would be a lie.
        for candidate in (scope, GLOBAL_SCOPE):
            entry = await memory.get(
                arguments.key,
                scope=candidate,
            )

            if entry is not None:
                return (
                    f"Found '{entry.key}' = '{entry.value}' "
                    f"({'global' if candidate == GLOBAL_SCOPE else 'task'})"
                )

        return (
            f"No memory for key '{arguments.key}'. "
            "Use recall_matching with a topic to search by what the "
            "note is about rather than by its exact key."
        )

    async def recall_matching(
        arguments: RecallMatchingInput,
        context: ToolContext,
    ) -> str:
        if arguments.limit < 1:
            return "limit must be at least 1."

        entries = visible_entries(
            await memory.all(),
            task_id_of(context),
        )

        snapshot = retrieve_snapshot(
            entries,
            arguments.topic,
            limit=arguments.limit,
        )

        if not snapshot.items:
            return (
                f"Nothing in memory matches '{arguments.topic}'."
            )

        lines = [
            f"Memory matching '{arguments.topic}' "
            f"({len(snapshot.items)} of {len(entries)} notes):",
            "",
        ]

        for item in snapshot.items:
            lines.append(f"  {item.content}")

        return "\n".join(lines)

    return remember, recall, recall_matching

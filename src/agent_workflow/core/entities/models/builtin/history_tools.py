from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from agent_workflow.core.context.history import HistoryStore
from agent_workflow.core.entities.models.tool import (
    Tool,
    ToolContext,
    ToolPolicy,
)

HISTORY_RECALL_LIMIT = 5
MAX_HISTORY_EXCERPT_CHARS = 1_000


@dataclass(slots=True, frozen=True)
class RecallHistoryInput:
    query: str


def create_history_tool(
    current_task_id: Callable[[], str | None],
    history: HistoryStore,
) -> Tool:
    async def handler(
        arguments: RecallHistoryInput,
        context: ToolContext,
    ) -> str:
        del context

        task_id = current_task_id()

        if task_id is None:
            return "No active task to recall history from."

        query = arguments.query.strip()

        if not query:
            return "Query must not be empty."

        found = history.search(
            task_id,
            query,
            limit=HISTORY_RECALL_LIMIT,
        )

        if not found:
            return (
                f"No history found for query '{query}'."
            )

        parts: list[str] = []

        for item in found:
            excerpt = item.content

            if len(excerpt) > MAX_HISTORY_EXCERPT_CHARS:
                excerpt = (
                    excerpt[:MAX_HISTORY_EXCERPT_CHARS]
                    + "\n[excerpt truncated]"
                )

            parts.append(
                f"[HISTORY {item.window_id} "
                f"{item.kind.value}] {excerpt}"
            )

        return "\n\n---\n\n".join(parts)

    return Tool(
        name="recall_history",
        description=(
            "Search the current task's history for past "
            "messages, tool activity, and checkpoints. "
            "History is never injected automatically; "
            "use this tool to retrieve it explicitly."
        ),
        input_type=RecallHistoryInput,
        handler=handler,
        policy=ToolPolicy(
            permissions=frozenset({"history.read"}),
            timeout=5.0,
            max_output_size=20_000,
        ),
    )

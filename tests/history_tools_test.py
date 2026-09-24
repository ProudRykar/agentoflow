from pathlib import Path

import pytest

from core.context.history import HistoryKind, InMemoryHistoryStore
from core.entities.models.builtin.history_tools import (
    RecallHistoryInput,
    create_history_tool,
)
from core.entities.models.tool import ToolContext


def _context(tmp_path: Path) -> ToolContext:
    return ToolContext(
        working_directory=tmp_path,
        environment={},
        allowed_path=(tmp_path,),
        permissions=frozenset({"history.read"}),
    )


def _seed(store: InMemoryHistoryStore) -> None:
    store.append(
        "task-01", "run-01", "window-01",
        HistoryKind.USER_MESSAGE, "Analyze Textual widgets",
    )
    store.append(
        "task-01", "run-01", "window-01",
        HistoryKind.ASSISTANT_MESSAGE, "Textual has widgets",
    )
    store.append(
        "task-01", "run-01", "window-02",
        HistoryKind.USER_MESSAGE, "Compare with Rich",
    )


@pytest.mark.asyncio
async def test_recall_finds_matching_items(
    tmp_path: Path,
) -> None:
    store = InMemoryHistoryStore()
    _seed(store)

    tool = create_history_tool(
        lambda: "task-01",
        store,
    )

    assert tool.name == "recall_history"

    output = await tool.handler(
        RecallHistoryInput(query="textual"),
        _context(tmp_path),
    )

    assert output.count("[HISTORY") == 2
    assert "Analyze Textual widgets" in output
    assert "Compare with Rich" not in output


@pytest.mark.asyncio
async def test_recall_no_match(tmp_path: Path) -> None:
    store = InMemoryHistoryStore()
    _seed(store)

    tool = create_history_tool(
        lambda: "task-01",
        store,
    )

    output = await tool.handler(
        RecallHistoryInput(query="django"),
        _context(tmp_path),
    )

    assert "No history found" in output


@pytest.mark.asyncio
async def test_recall_without_active_task(
    tmp_path: Path,
) -> None:
    store = InMemoryHistoryStore()

    tool = create_history_tool(
        lambda: None,
        store,
    )

    output = await tool.handler(
        RecallHistoryInput(query="textual"),
        _context(tmp_path),
    )

    assert "No active task" in output


@pytest.mark.asyncio
async def test_recall_rejects_empty_query(
    tmp_path: Path,
) -> None:
    store = InMemoryHistoryStore()

    tool = create_history_tool(
        lambda: "task-01",
        store,
    )

    output = await tool.handler(
        RecallHistoryInput(query="   "),
        _context(tmp_path),
    )

    assert "must not be empty" in output


@pytest.mark.asyncio
async def test_recall_respects_limit_and_truncates(
    tmp_path: Path,
) -> None:
    store = InMemoryHistoryStore()

    store.append(
        "task-01", "run-01", "window-01",
        HistoryKind.TOOL_RESULT, "textual " + "y" * 5_000,
    )

    for index in range(7):
        store.append(
            "task-01", "run-01", "window-01",
            HistoryKind.USER_MESSAGE, f"textual note {index}",
        )

    tool = create_history_tool(
        lambda: "task-01",
        store,
    )

    output = await tool.handler(
        RecallHistoryInput(query="textual"),
        _context(tmp_path),
    )

    assert output.count("[HISTORY") == 5
    assert "truncated" in output

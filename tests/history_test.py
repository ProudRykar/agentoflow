import pytest

from core.context.history import (
    HistoryItem,
    HistoryKind,
    InMemoryHistoryStore,
)


def test_append_and_get() -> None:
    store = InMemoryHistoryStore()

    first = store.append(
        "task-01",
        "run-01",
        "window-01",
        HistoryKind.USER_MESSAGE,
        "hello",
    )
    second = store.append(
        "task-01",
        "run-01",
        "window-01",
        HistoryKind.ASSISTANT_MESSAGE,
        "hi",
        reference="window-01",
    )

    assert first.item_id == "hist-000001"
    assert second.item_id == "hist-000002"
    assert len(store) == 2
    assert store.get("hist-000001") == first
    assert store.get("missing") is None


def test_append_requires_task_and_window() -> None:
    store = InMemoryHistoryStore()

    with pytest.raises(ValueError):
        store.append(
            "",
            "run-01",
            "window-01",
            HistoryKind.USER_MESSAGE,
            "hello",
        )

    with pytest.raises(ValueError):
        store.append(
            "task-01",
            "run-01",
            "",
            HistoryKind.USER_MESSAGE,
            "hello",
        )


def test_append_truncates_long_content() -> None:
    store = InMemoryHistoryStore()

    item = store.append(
        "task-01",
        "run-01",
        "window-01",
        HistoryKind.TOOL_RESULT,
        "x" * 10_000,
    )

    assert len(item.content) < 10_000
    assert "truncated" in item.content


def test_window_task_recent_scoping() -> None:
    store = InMemoryHistoryStore()
    store.append(
        "task-01", "run-01", "window-01",
        HistoryKind.USER_MESSAGE, "one",
    )
    store.append(
        "task-01", "run-01", "window-02",
        HistoryKind.USER_MESSAGE, "two",
    )
    store.append(
        "task-02", "run-01", "window-01",
        HistoryKind.USER_MESSAGE, "other",
    )

    assert len(store.window_items("task-01", "window-01")) == 1
    assert len(store.task_items("task-01")) == 2
    assert store.recent_items("task-01", 1)[0].content == "two"

    with pytest.raises(ValueError):
        store.recent_items("task-01", 0)


def test_search() -> None:
    store = InMemoryHistoryStore()
    store.append(
        "task-01", "run-01", "window-01",
        HistoryKind.USER_MESSAGE, "Analyze Textual widgets",
    )
    store.append(
        "task-01", "run-01", "window-01",
        HistoryKind.USER_MESSAGE, "Compare with Rich",
    )

    found = store.search("task-01", "textual")

    assert len(found) == 1
    assert found[0].content == "Analyze Textual widgets"

    assert store.search("task-02", "textual") == ()

    with pytest.raises(ValueError):
        store.search("task-01", "")


def test_history_item_roundtrip() -> None:
    item = HistoryItem(
        item_id="hist-000001",
        task_id="task-01",
        run_id="run-01",
        window_id="window-01",
        kind=HistoryKind.PHASE_CHANGE,
        content="planning -> execution",
    )

    assert HistoryItem.from_dict(item.to_dict()) == item

    with pytest.raises(ValueError):
        HistoryItem(
            item_id="",
            task_id="task-01",
            run_id="",
            window_id="window-01",
            kind=HistoryKind.USER_MESSAGE,
            content="hi",
        )

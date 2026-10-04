from __future__ import annotations

from pathlib import Path

import pytest

from agent_workflow.core.application.tool_stats import (
    ToolStats,
    ToolUsage,
)
from agent_workflow.core.application.tool_stats_store import (
    ToolStatsStore,
    UsageRecord,
)


@pytest.fixture
def store(tmp_path: Path) -> ToolStatsStore:
    return ToolStatsStore(tmp_path / "usage.db")


# ======================================================================
# Writes and reads
# ======================================================================


def test_record_and_read_tool(store: ToolStatsStore) -> None:
    store.record_tool_delta(
        "s1",
        tool_name="read_file",
        calls=3,
        successes=2,
        errors=1,
        duration=1.5,
        error_code="not_found",
        error_codes={"not_found": 1},
    )

    [record] = store.tool_rows("s1")

    assert record.tool_name == "read_file"
    assert record.calls == 3
    assert record.successes == 2
    assert record.errors == 1
    assert record.total_duration == pytest.approx(1.5)
    assert record.last_error_code == "not_found"
    assert record.error_codes == {"not_found": 1}


def test_deltas_accumulate(store: ToolStatsStore) -> None:
    for _ in range(3):
        store.record_tool_delta(
            "s1",
            tool_name="read_file",
            calls=1,
            successes=1,
            duration=0.5,
        )

    [record] = store.tool_rows("s1")

    assert record.calls == 3
    assert record.successes == 3
    assert record.total_duration == pytest.approx(1.5)


def test_sessions_are_isolated(store: ToolStatsStore) -> None:
    store.record_tool_delta(
        "s1", tool_name="read_file", calls=1
    )
    store.record_tool_delta(
        "s2", tool_name="read_file", calls=5
    )

    assert store.tool_rows("s1")[0].calls == 1
    assert store.tool_rows("s2")[0].calls == 5
    assert store.total_tool_calls("s1") == 1
    assert store.total_tool_calls() == 6


def test_last_error_code_is_preserved(store: ToolStatsStore) -> None:
    store.record_tool_delta(
        "s1",
        tool_name="t",
        errors=1,
        error_code="boom",
    )
    store.record_tool_delta("s1", tool_name="t", successes=1)

    [record] = store.tool_rows("s1")

    assert record.last_error_code == "boom"


def test_empty_tool_name_is_ignored(
    store: ToolStatsStore,
) -> None:
    store.record_tool_delta("s1", tool_name="", calls=1)

    assert store.tool_rows("s1") == []


# ======================================================================
# Approvals
# ======================================================================


def test_approval_round_trip(store: ToolStatsStore) -> None:
    store.record_approval_delta(
        "s1",
        requested=2,
        allowed=1,
        denied=1,
        by_permission={"shell.network": 2},
    )

    row = store.approval_row("s1")

    assert row["requested"] == 2
    assert row["allowed"] == 1
    assert row["denied"] == 1
    assert row["by_permission"] == {"shell.network": 2}


def test_approval_accumulates(store: ToolStatsStore) -> None:
    store.record_approval_delta("s1", requested=1)
    store.record_approval_delta("s1", allowed=1)

    assert store.approval_row("s1")["requested"] == 1
    assert store.approval_row("s1")["allowed"] == 1


def test_missing_approval_row_is_zeroed(
    store: ToolStatsStore,
) -> None:
    row = store.approval_row("nope")

    assert row["requested"] == 0
    assert row["by_permission"] == {}


# ======================================================================
# Aggregates
# ======================================================================


def test_top_tools_across_sessions(
    store: ToolStatsStore,
) -> None:
    store.record_tool_delta("s1", tool_name="a", calls=2)
    store.record_tool_delta("s1", tool_name="b", calls=7)
    store.record_tool_delta("s2", tool_name="a", calls=5)

    assert store.top_tools() == [("a", 7), ("b", 7)]


def test_top_tools_respects_limit(
    store: ToolStatsStore,
) -> None:
    for index in range(5):
        store.record_tool_delta(
            "s1", tool_name=f"t{index}", calls=index + 1
        )

    assert len(store.top_tools(limit=2)) == 2


def test_total_calls_empty(store: ToolStatsStore) -> None:
    assert store.total_tool_calls() == 0
    assert store.top_tools() == []


# ======================================================================
# Maintenance
# ======================================================================


def test_forget_removes_session(store: ToolStatsStore) -> None:
    store.record_tool_delta("s1", tool_name="a", calls=1)
    store.record_approval_delta("s1", requested=1)

    store.forget("s1")

    assert store.tool_rows("s1") == []
    assert store.approval_row("s1")["requested"] == 0


def test_creates_parent_directory(tmp_path: Path) -> None:
    store = ToolStatsStore(
        tmp_path / "nested" / "deep" / "usage.db"
    )

    store.record_tool_delta("s1", tool_name="a", calls=1)

    assert store.total_tool_calls() == 1


def test_survives_reopen(tmp_path: Path) -> None:
    path = tmp_path / "usage.db"

    ToolStatsStore(path).record_tool_delta(
        "s1", tool_name="a", calls=4
    )

    # A fresh instance reads the same file.
    assert ToolStatsStore(path).total_tool_calls("s1") == 4


def test_corrupt_error_codes_are_tolerated(
    store: ToolStatsStore,
) -> None:
    store.record_tool_delta(
        "s1",
        tool_name="t",
        calls=1,
        error_codes={"x": 1},
    )

    import sqlite3

    with sqlite3.connect(store._database) as connection:
        connection.execute(
            "UPDATE tool_usage SET error_codes = 'not json'"
        )

    [record] = store.tool_rows("s1")

    assert record.error_codes == {}


# ======================================================================
# Derived values
# ======================================================================


def test_usage_record_derived_values() -> None:
    record = UsageRecord(
        session_id="s",
        tool_name="t",
        calls=4,
        successes=3,
        errors=1,
        total_duration=2.0,
        last_error_code=None,
        error_codes={},
    )

    assert record.average_duration == pytest.approx(0.5)
    assert record.error_rate == pytest.approx(0.25)


def test_usage_record_before_completion() -> None:
    record = UsageRecord(
        session_id="s",
        tool_name="t",
        calls=1,
        successes=0,
        errors=0,
        total_duration=0.0,
        last_error_code=None,
        error_codes={},
    )

    assert record.average_duration is None
    assert record.error_rate == 0.0


# ======================================================================
# Sink protocol
# ======================================================================


def test_sink_accepts_tool_usage_objects(
    store: ToolStatsStore,
) -> None:
    usage = ToolUsage(
        name="read_file",
        calls=2,
        successes=1,
        errors=1,
        total_duration=1.0,
        last_error_code="boom",
        error_codes={"boom": 1},
    )

    store.record_tool("s1", usage)

    [record] = store.tool_rows("s1")

    assert record.calls == 2
    assert record.errors == 1
    assert record.error_codes == {"boom": 1}


def test_stats_flush_writes_to_store(
    store: ToolStatsStore,
) -> None:
    from agent_workflow.core.entities.models.agent_trace import (
        ToolFinished,
        ToolStarted,
    )

    stats = ToolStats(session_id="s1", sink=store)

    stats.observe(
        ToolStarted(
            iteration=1,
            tool_call_id="c1",
            tool_name="read_file",
            arguments={},
        )
    )
    stats.observe(
        ToolFinished(
            iteration=1,
            tool_call_id="c1",
            tool_name="read_file",
            output="ok",
            error_code=None,
            error_message=None,
            duration_seconds=0.25,
        )
    )

    stats.flush()

    [record] = store.tool_rows("s1")

    assert record.calls == 1
    assert record.successes == 1


def test_flush_without_sink_is_noop() -> None:
    ToolStats(session_id="s1").flush()


def test_flush_contains_sink_errors() -> None:
    class Broken:
        def record_tool(self, *_: object) -> None:
            raise RuntimeError("db gone")

        def record_approval(self, *_: object) -> None:
            raise RuntimeError("db gone")

    stats = ToolStats(session_id="s1", sink=Broken())

    stats.tools["x"] = ToolUsage(name="x")

    stats.flush()


# ======================================================================
# Aggregate endpoint
# ======================================================================


def test_aggregate_endpoint(stubbed, tmp_path: Path) -> None:
    from fastapi.testclient import TestClient

    manager = stubbed.app.state.session_manager

    store = manager.stats_store

    assert store is not None

    store.record_tool_delta(
        "s1", tool_name="read_file", calls=5, successes=4,
        errors=1, duration=2.0,
    )
    store.record_tool_delta(
        "s1", tool_name="write_file", calls=1, successes=1,
    )
    store.record_tool_delta(
        "s2", tool_name="read_file", calls=3, successes=3,
    )

    response: TestClient = stubbed.get(
        "/api/tools/history/aggregate"
    )

    assert response.status_code == 200

    body = response.json()

    assert body["total_calls"] == 9
    assert body["distinct_tools"] == 2

    names = [t["tool_name"] for t in body["top_tools"]]

    assert names == ["read_file", "write_file"]
    assert body["top_tools"][0]["calls"] == 8
    assert body["top_tools"][0]["successes"] == 7


def test_aggregate_endpoint_limit(stubbed, tmp_path: Path) -> None:
    store = stubbed.app.state.session_manager.stats_store

    for index in range(5):
        store.record_tool_delta(
            "s1", tool_name=f"t{index}", calls=index + 1
        )

    body = stubbed.get(
        "/api/tools/history/aggregate?limit=2"
    ).json()

    assert len(body["top_tools"]) == 2


def test_aggregate_endpoint_empty(stubbed) -> None:
    body = stubbed.get(
        "/api/tools/history/aggregate"
    ).json()

    assert body["total_calls"] == 0
    assert body["top_tools"] == []

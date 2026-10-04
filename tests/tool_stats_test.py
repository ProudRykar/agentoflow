from __future__ import annotations

import pytest

from agent_workflow.core.application.tool_stats import ToolStats
from agent_workflow.core.entities.models.agent_trace import (
    ToolFinished,
    ToolStarted,
)
from agent_workflow.core.entities.models.approval import (
    ApprovalRequested,
    ApprovalResolved,
)


def started(name: str = "read_file", call: str = "c1") -> ToolStarted:
    return ToolStarted(
        iteration=1,
        tool_call_id=call,
        tool_name=name,
        arguments={},
    )


def finished(
    name: str = "read_file",
    call: str = "c1",
    error: str | None = None,
    duration: float | None = 1.0,
) -> ToolFinished:
    return ToolFinished(
        iteration=1,
        tool_call_id=call,
        tool_name=name,
        output=None if error else "ok",
        error_code=error,
        error_message=None if error else "failed",
        duration_seconds=duration,
        run_id="",
    )


def test_counts_calls_and_successes() -> None:
    stats = ToolStats()

    stats.observe(started())
    stats.observe(finished())

    usage = stats.get("read_file")

    assert usage is not None
    assert usage.calls == 1
    assert usage.successes == 1
    assert usage.errors == 0
    assert usage.running == 0


def test_tracks_running_between_events() -> None:
    stats = ToolStats()

    stats.observe(started())

    usage = stats.get("read_file")

    assert usage is not None
    assert usage.running == 1
    assert usage.finished == 0


def test_tracks_errors_and_rate() -> None:
    stats = ToolStats()

    for _ in range(3):
        stats.observe(started())
        stats.observe(finished())

    stats.observe(started())
    stats.observe(finished(error="not_found"))

    usage = stats.get("read_file")

    assert usage is not None
    assert usage.calls == 4
    assert usage.successes == 3
    assert usage.errors == 1
    assert usage.error_rate == pytest.approx(0.25)
    assert usage.last_error_code == "not_found"
    assert usage.error_codes == {"not_found": 1}


def test_averages_duration() -> None:
    stats = ToolStats()

    stats.observe(started())
    stats.observe(finished(duration=1.0))
    stats.observe(started())
    stats.observe(finished(duration=3.0))

    usage = stats.get("read_file")

    assert usage is not None
    assert usage.average_duration == pytest.approx(2.0)


def test_average_duration_none_before_completion() -> None:
    stats = ToolStats()

    stats.observe(started())

    assert stats.get("read_file").average_duration is None


def test_keeps_tools_separate() -> None:
    stats = ToolStats()

    stats.observe(started("read_file", "c1"))
    stats.observe(started("write_file", "c2"))
    stats.observe(finished("write_file", "c2"))

    assert stats.get("read_file").running == 1
    assert stats.get("write_file").successes == 1


def test_ranked_orders_by_calls() -> None:
    stats = ToolStats()

    stats.observe(started("a", "1"))
    stats.observe(started("a", "2"))
    stats.observe(started("b", "3"))

    ranked = stats.ranked()

    assert [usage.name for usage in ranked] == ["a", "b"]
    assert ranked[0].calls == 2


def test_summary_aggregates() -> None:
    stats = ToolStats()

    stats.observe(started("a", "1"))
    stats.observe(finished("a", "1"))
    stats.observe(started("b", "2"))
    stats.observe(finished("b", "2", error="boom"))

    summary = stats.summary()

    assert summary["tools_used"] == 2
    assert summary["total_calls"] == 2
    assert summary["total_successes"] == 1
    assert summary["total_errors"] == 1
    assert summary["error_rate"] == pytest.approx(0.5)


def test_summary_empty_is_zeroed() -> None:
    summary = ToolStats().summary()

    assert summary["tools_used"] == 0
    assert summary["total_calls"] == 0
    assert summary["error_rate"] == 0.0


def test_approval_lifecycle() -> None:
    stats = ToolStats()

    stats.observe(
        ApprovalRequested(
            approval_id="a1",
            tool_name="execute_shell",
            arguments={},
            permission="shell.network",
            reason="r",
        )
    )

    assert stats.approval.requested == 1
    assert stats.approval.pending is True
    assert stats.approval.by_permission == {"shell.network": 1}

    stats.observe(
        ApprovalResolved(approval_id="a1", approved=True)
    )

    assert stats.approval.pending is False
    assert stats.approval.allowed == 1
    assert stats.approval.denied == 0


def test_approval_denial_counted() -> None:
    stats = ToolStats()

    stats.observe(
        ApprovalResolved(approval_id="a1", approved=False)
    )

    assert stats.approval.denied == 1
    assert stats.approval.allowed == 0


def test_approval_summary_fields() -> None:
    stats = ToolStats()

    stats.observe(
        ApprovalRequested(
            approval_id="a1",
            tool_name="t",
            arguments={},
            permission="p",
            reason="r",
        )
    )
    stats.observe(
        ApprovalResolved(approval_id="a1", approved=True)
    )

    summary = stats.summary()

    assert summary["approvals_requested"] == 1
    assert summary["approvals_allowed"] == 1
    assert summary["approvals_denied"] == 0
    assert summary["approval_pending"] is False


def test_ignores_unknown_events() -> None:
    stats = ToolStats()

    stats.observe(object())
    stats.observe("nonsense")
    stats.observe(None)

    assert stats.summary()["total_calls"] == 0


def test_running_never_goes_negative() -> None:
    """A stray ToolFinished must not corrupt the counters."""

    stats = ToolStats()

    stats.observe(finished())

    assert stats.get("read_file").running == 0

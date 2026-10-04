from __future__ import annotations

from dataclasses import dataclass, field

from agent_workflow.core.entities.models.agent_trace import (
    ToolFinished,
    ToolStarted,
)
from agent_workflow.core.entities.models.approval import (
    ApprovalRequested,
    ApprovalResolved,
)


@dataclass(slots=True)
class ToolUsage:
    """Per-tool counters derived from the event stream."""

    name: str
    calls: int = 0
    running: int = 0
    successes: int = 0
    errors: int = 0
    total_duration: float = 0.0
    last_error_code: str | None = None
    last_used_at: float = 0.0
    error_codes: dict[str, int] = field(default_factory=dict)

    @property
    def finished(self) -> int:
        return self.successes + self.errors

    @property
    def error_rate(self) -> float:
        if self.finished == 0:
            return 0.0

        return self.errors / self.finished

    @property
    def average_duration(self) -> float | None:
        if not self.finished:
            return None

        return self.total_duration / self.finished


@dataclass(slots=True)
class ApprovalUsage:
    requested: int = 0
    allowed: int = 0
    denied: int = 0
    pending: bool = False
    by_permission: dict[str, int] = field(default_factory=dict)


class StatsSink:
    """Where flushed statistics are written."""

    def record_tool(self, session_id: str, usage: ToolUsage) -> None:
        raise NotImplementedError

    def record_approval(
        self,
        session_id: str,
        usage: ApprovalUsage,
    ) -> None:
        raise NotImplementedError


@dataclass(slots=True)
class ToolStats:
    """Usage statistics for one session.

    Fed from the session's own event bus, so the numbers reflect
    what actually ran rather than what was merely available.
    """

    session_id: str = ""
    tools: dict[str, ToolUsage] = field(default_factory=dict)
    approval: ApprovalUsage = field(default_factory=ApprovalUsage)
    sink: StatsSink | None = None

    def flush(self) -> None:
        """Push current counters to the durable store, if any."""

        sink = self.sink

        if sink is None:
            return

        try:
            for usage in self.tools.values():
                sink.record_tool(self.session_id, usage)

            sink.record_approval(self.session_id, self.approval)
        except Exception:
            # Never let persistence break the agent loop.
            return

    def observe(
        self,
        event: object,
    ) -> None:
        if isinstance(event, ToolStarted):
            self._on_started(event)
            return

        if isinstance(event, ToolFinished):
            self._on_finished(event)
            return

        if isinstance(event, ApprovalRequested):
            self.approval.requested += 1
            self.approval.pending = True

            key = event.permission

            self.approval.by_permission[key] = (
                self.approval.by_permission.get(key, 0) + 1
            )

            return

        if isinstance(event, ApprovalResolved):
            self.approval.pending = False

            if event.approved:
                self.approval.allowed += 1
            else:
                self.approval.denied += 1

    def _on_started(self, event: ToolStarted) -> None:
        usage = self.tools.get(event.tool_name)

        if usage is None:
            usage = ToolUsage(name=event.tool_name)
            self.tools[event.tool_name] = usage

        usage.calls += 1
        usage.running += 1

    def _on_finished(self, event: ToolFinished) -> None:
        usage = self.tools.get(event.tool_name)

        if usage is None:
            usage = ToolUsage(name=event.tool_name)
            self.tools[event.tool_name] = usage

        usage.running = max(0, usage.running - 1)

        if event.duration_seconds is not None:
            usage.total_duration += event.duration_seconds

        if event.error_code is None:
            usage.successes += 1
        else:
            usage.errors += 1
            usage.last_error_code = event.error_code
            usage.error_codes[event.error_code] = (
                usage.error_codes.get(event.error_code, 0) + 1
            )

    def get(
        self,
        name: str,
    ) -> ToolUsage | None:
        return self.tools.get(name)

    def summary(self) -> dict[str, object]:
        total_calls = sum(
            usage.calls for usage in self.tools.values()
        )

        total_errors = sum(
            usage.errors for usage in self.tools.values()
        )

        finished = sum(
            usage.finished for usage in self.tools.values()
        )

        return {
            # Tools that have actually been invoked at least once.
            # The number of *available* tools is an inventory
            # concern and lives in the web layer, not here.
            "tools_used": len(self.tools),
            "total_calls": total_calls,
            "total_successes": sum(
                usage.successes for usage in self.tools.values()
            ),
            "total_errors": total_errors,
            "error_rate": (
                total_errors / finished if finished else 0.0
            ),
            "approvals_requested": self.approval.requested,
            "approvals_allowed": self.approval.allowed,
            "approvals_denied": self.approval.denied,
            "approval_pending": self.approval.pending,
        }

    def ranked(
        self,
    ) -> list[ToolUsage]:
        """Most-called first, then by name for stability."""

        return sorted(
            self.tools.values(),
            key=lambda usage: (-usage.calls, usage.name),
        )

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from agent_workflow.core.entities.models.agent_trace import AgentEvent
from agent_workflow.core.entities.models.approval import (
    ApprovalRequested,
    ApprovalResolved,
)


DomainEvent = (
    AgentEvent
    | ApprovalRequested
    | ApprovalResolved
)

Subscriber = Callable[[int, DomainEvent], Awaitable[None]]

DEFAULT_HISTORY_LIMIT = 500


@dataclass(slots=True, frozen=True)
class StoredEvent:
    seq: int
    event: DomainEvent


@dataclass(slots=True)
class EventBus:
    """Fan-out of domain events to any number of subscribers.

    Agent Core keeps emitting the events it always emitted; the
    bus only adapts the single ``on_event`` callback into a
    broadcast with sequence numbers and a bounded replay buffer.

    Sequence numbers let a reconnecting client ask for
    everything it missed instead of polling.
    """

    history_limit: int = DEFAULT_HISTORY_LIMIT

    _subscribers: dict[int, Subscriber] = field(default_factory=dict)
    _history: deque[StoredEvent] = field(default_factory=deque)
    _seq: int = 0
    _next_key: int = 0
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    @property
    def seq(self) -> int:
        return self._seq

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)

    def subscribe(self, subscriber: Subscriber) -> int:
        key = self._next_key
        self._next_key += 1
        self._subscribers[key] = subscriber

        return key

    def unsubscribe(self, key: int) -> None:
        self._subscribers.pop(key, None)

    def replay_since(
        self,
        seq: int,
        *,
        backfill: Callable[[int], list[StoredEvent]] | None = None,
    ) -> list[StoredEvent]:
        """Events strictly newer than ``seq``, oldest first.

        The in-memory buffer is bounded, so for a long session it
        holds only the most recent slice. Replaying a cursor from
        before that slice would silently hand back a truncated
        history and the reader would find the top of the
        conversation missing with no way to scroll to it.

        ``backfill`` reads the evicted prefix from durable storage.
        It is asked for only when the gap is real, and its result is
        merged ahead of the retained events so the sequence stays
        contiguous.
        """

        oldest = self.oldest_retained_seq()

        # An empty buffer retains nothing at all, so the durable log is
        # the only copy of the session. Guarding on a non-empty buffer
        # treated that as "no gap" and replayed nothing, which is how a
        # session whose events all failed to decode came back blank.
        retained = bool(self._history)
        has_gap = not retained or seq + 1 < oldest

        prefix: list[StoredEvent] = []

        if backfill is not None and has_gap:
            prefix = [
                stored
                for stored in backfill(seq)
                if stored.seq > seq and (not retained or stored.seq < oldest)
            ]

        return prefix + [
            stored
            for stored in self._history
            if stored.seq > seq
        ]

    def oldest_retained_seq(self) -> int:
        if not self._history:
            return 0

        return self._history[0].seq

    def restore(
        self,
        events: list[StoredEvent],
    ) -> None:
        """Seed the replay buffer from durable storage.

        Subscribers are not called and the events are not
        republished: this runs before anyone is listening, so a
        reconnecting client receives the restored transcript through
        the normal replay path instead of a second mechanism.
        """

        if not events:
            return

        for stored in events:
            if stored.seq <= self._seq:
                # Already accounted for: restoring must not rewind
                # the counter or duplicate entries.
                continue

            self._history.append(stored)
            self._seq = stored.seq

        while len(self._history) > self.history_limit:
            self._history.popleft()

    async def publish(self, event: DomainEvent) -> int:
        async with self._lock:
            self._seq += 1
            seq = self._seq

            stored = StoredEvent(seq=seq, event=event)

            self._history.append(stored)

            while len(self._history) > self.history_limit:
                self._history.popleft()

            subscribers = list(self._subscribers.items())

        for _, subscriber in subscribers:
            try:
                await subscriber(seq, event)
            except Exception:
                # A broken subscriber must never stall the agent.
                continue

        return seq

    async def __call__(
        self,
        event: AgentEvent,
    ) -> None:
        """Pass-through so the bus can be used as ``on_event``."""

        await self.publish(event)

    def snapshot(self) -> list[StoredEvent]:
        return list(self._history)

    def to_dict(self, event: DomainEvent) -> dict[str, Any]:
        return {"type": type(event).__name__, "event": event}

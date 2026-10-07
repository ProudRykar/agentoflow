from __future__ import annotations

import asyncio

import pytest

from agent_workflow.core.application.event_store import EventStore
from agent_workflow.core.application.events import EventBus, StoredEvent
from agent_workflow.core.entities.models.agent_trace import (
    AgentFinished,
    AgentStarted,
    LLMContentChunk,
)
from agent_workflow.core.entities.models.approval import (
    ApprovalRequested,
    ApprovalResolved,
)


def started(prompt: str = "hi") -> AgentStarted:
    return AgentStarted(prompt=prompt)


async def test_sequences_increase_monotonically() -> None:
    bus = EventBus()

    first = await bus.publish(started())
    second = await bus.publish(started())

    assert first == 1
    assert second == 2
    assert bus.seq == 2


async def test_broadcasts_to_all_subscribers() -> None:
    bus = EventBus()

    a: list[int] = []
    b: list[int] = []

    async def collect_a(seq, event) -> None:
        a.append(seq)

    async def collect_b(seq, event) -> None:
        b.append(seq)

    bus.subscribe(collect_a)
    bus.subscribe(collect_b)

    await bus.publish(started())

    assert a == [1]
    assert b == [1]


async def test_unsubscribe_stops_delivery() -> None:
    bus = EventBus()

    received: list[int] = []

    async def collect(seq, event) -> None:
        received.append(seq)

    key = bus.subscribe(collect)

    await bus.publish(started())

    bus.unsubscribe(key)

    await bus.publish(started())

    assert received == [1]
    assert bus.subscriber_count == 0


async def test_broken_subscriber_does_not_block_others() -> None:
    bus = EventBus()

    received: list[int] = []

    async def boom(seq, event) -> None:
        raise RuntimeError("subscriber failed")

    async def collect(seq, event) -> None:
        received.append(seq)

    bus.subscribe(boom)
    bus.subscribe(collect)

    seq = await bus.publish(started())

    assert received == [seq]


async def test_replay_since_returns_newer_events() -> None:
    bus = EventBus()

    await bus.publish(started("one"))
    await bus.publish(started("two"))
    await bus.publish(started("three"))

    replay = bus.replay_since(1)

    assert [stored.seq for stored in replay] == [2, 3]
    assert replay[0].event.prompt == "two"


async def test_replay_since_current_returns_nothing() -> None:
    bus = EventBus()

    await bus.publish(started())

    assert bus.replay_since(bus.seq) == []


async def test_replay_respects_negative_cursor() -> None:
    bus = EventBus()

    await bus.publish(started("one"))
    await bus.publish(started("two"))

    replay = bus.replay_since(-1)

    assert [stored.seq for stored in replay] == [1, 2]


async def test_history_is_bounded() -> None:
    bus = EventBus(history_limit=3)

    for index in range(10):
        await bus.publish(started(f"p{index}"))

    replay = bus.replay_since(0)

    assert [stored.seq for stored in replay] == [8, 9, 10]


async def test_oldest_retained_seq_reports_eviction() -> None:
    bus = EventBus(history_limit=2)

    await bus.publish(started("a"))
    await bus.publish(started("b"))
    await bus.publish(started("c"))

    assert bus.oldest_retained_seq() == 2


async def test_callable_form_works_as_agent_callback() -> None:
    bus = EventBus()

    received: list[object] = []

    async def collect(seq, event) -> None:
        received.append(event)

    bus.subscribe(collect)

    await bus(AgentStarted(prompt="via callback"))

    assert len(received) == 1


async def test_streaming_order_is_preserved() -> None:
    bus = EventBus()

    order: list[str] = []

    async def collect(seq, event) -> None:
        order.append(
            getattr(event, "content", type(event).__name__)
        )

    bus.subscribe(collect)

    for chunk in ["Hel", "lo", "!"]:
        await bus.publish(
            LLMContentChunk(iteration=1, content=chunk)
        )

    assert order == ["Hel", "lo", "!"]


async def test_approval_events_share_the_stream() -> None:
    bus = EventBus()

    received: list[object] = []

    async def collect(seq, event) -> None:
        received.append(event)

    bus.subscribe(collect)

    await bus.publish(AgentStarted(prompt="go"))
    await bus.publish(
        ApprovalRequested(
            approval_id="a1",
            tool_name="execute_shell",
            arguments={},
            permission="shell.execute",
            reason="r",
        )
    )
    await bus.publish(
        ApprovalResolved(approval_id="a1", approved=True)
    )
    await bus.publish(AgentFinished(result="done"))

    assert [type(event).__name__ for event in received] == [
        "AgentStarted",
        "ApprovalRequested",
        "ApprovalResolved",
        "AgentFinished",
    ]


async def test_concurrent_publish_keeps_sequence_unique() -> None:
    bus = EventBus()

    seen: list[int] = []

    async def collect(seq, event) -> None:
        seen.append(seq)

    bus.subscribe(collect)

    await asyncio.gather(
        *(bus.publish(started()) for _ in range(20))
    )

    assert sorted(seen) == list(range(1, 21))


def test_snapshot_returns_history() -> None:
    bus = EventBus()

    assert bus.snapshot() == []


def _chunk(index: int):
    return LLMContentChunk(iteration=1, content=f"chunk {index} ")


def test_replay_backfills_when_the_buffer_is_empty(tmp_path) -> None:
    """An empty buffer means the durable log is the only copy.

    ``oldest_retained_seq`` answers 0 for an empty buffer, so the
    ``seq < oldest`` filter would keep nothing, and a guard that
    required a non-empty buffer then skipped the store altogether: the
    replay came back empty even though every event was on disk.
    """

    store = EventStore(tmp_path / "events.db")

    store.append(
        "s1",
        [StoredEvent(seq=index, event=_chunk(index)) for index in range(1, 6)],
    )

    bus = EventBus(history_limit=10)

    replayed = bus.replay_since(
        -1,
        backfill=lambda seq: store.load("s1", after_seq=seq),
    )

    assert [stored.seq for stored in replayed] == [1, 2, 3, 4, 5]


def test_replay_stays_contiguous_across_the_backfill_boundary(
    tmp_path,
) -> None:
    store = EventStore(tmp_path / "events.db")

    store.append(
        "s1",
        [StoredEvent(seq=index, event=_chunk(index)) for index in range(1, 61)],
    )

    bus = EventBus(history_limit=10)
    bus.restore(store.load("s1"))

    assert bus.oldest_retained_seq() == 51

    replayed = bus.replay_since(
        -1,
        backfill=lambda seq: store.load("s1", after_seq=seq),
    )

    assert [stored.seq for stored in replayed] == list(range(1, 61))


def test_replay_from_a_cursor_inside_the_buffer_skips_the_store(
    tmp_path,
) -> None:
    """A reconnect that lost nothing must not re-read the whole log."""

    store = EventStore(tmp_path / "events.db")

    store.append(
        "s1",
        [StoredEvent(seq=index, event=_chunk(index)) for index in range(1, 21)],
    )

    bus = EventBus(history_limit=20)
    bus.restore(store.load("s1"))

    asked: list[int] = []

    replayed = bus.replay_since(
        15,
        backfill=lambda seq: asked.append(seq) or [],
    )

    assert [stored.seq for stored in replayed] == [16, 17, 18, 19, 20]
    # No gap, so the store is not consulted at all.
    assert asked == []

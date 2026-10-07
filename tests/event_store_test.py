"""Sessions and transcripts must survive a server restart.

A restart used to leave the sidebar empty and every chat blank:
only the session identity was on disk, and the transcript was held
in the event bus, which dies with the process.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from agent_workflow.core.application.event_store import (
    EventStore,
    decode_event,
    dialogue_from_events,
    encode_event,
)
from agent_workflow.core.application.events import EventBus, StoredEvent
from agent_workflow.core.application.session_store import SessionStore
from agent_workflow.core.entities.models.agent_phase import AgentPhase
from agent_workflow.core.entities.models.agent_trace import (
    AgentFinished,
    AgentPhaseChanged,
    AgentStarted,
    LLMContentChunk,
    LLMResponded,
    RunFailed,
    ToolFinished,
    ToolStarted,
)
from agent_workflow.core.entities.models.approval import (
    ApprovalRequested,
    ApprovalResolved,
)


# ======================================================================
# Round trip
# ======================================================================


@pytest.mark.parametrize(
    "event",
    [
        AgentStarted(prompt="hello", run_id="r1", model="gemma4:12b"),
        AgentPhaseChanged(
            previous_phase=AgentPhase.IDLE,
            phase=AgentPhase.PLANNING,
            reason="start",
            iteration=0,
        ),
        LLMContentChunk(iteration=1, content="chunk", run_id="r1"),
        LLMResponded(
            iteration=1,
            content="answer",
            thinking="hmm",
            tool_call_count=2,
        ),
        ToolStarted(
            iteration=1,
            tool_call_id="c1",
            tool_name="read_file",
            arguments={"path": "a.txt", "nested": {"k": [1, 2]}},
        ),
        ToolFinished(
            iteration=1,
            tool_call_id="c1",
            tool_name="read_file",
            output="text",
            error_code=None,
            error_message=None,
            duration_seconds=0.25,
        ),
        AgentFinished(result="done", agent_id="main", role="main"),
        ApprovalRequested(
            approval_id="a1",
            tool_name="shell",
            arguments={"cmd": "ls"},
            permission="shell.execute",
            reason="mutating",
        ),
        ApprovalResolved(approval_id="a1", approved=True),
    ],
)
def test_event_round_trip(event) -> None:
    restored = decode_event(encode_event(event))

    assert restored == event


def test_enum_survives_round_trip() -> None:
    event = AgentPhaseChanged(
        previous_phase=AgentPhase.IDLE,
        phase=AgentPhase.EXECUTION,
        reason="loop",
        iteration=2,
    )

    restored = decode_event(encode_event(event))

    assert isinstance(restored, AgentPhaseChanged)
    assert restored.phase is AgentPhase.EXECUTION


def test_unknown_event_type_is_skipped() -> None:
    assert decode_event('{"type":"Nope","fields":{}}') is None


def test_corrupt_payload_is_skipped() -> None:
    assert decode_event("{not json") is None


# ======================================================================
# Store
# ======================================================================


def test_store_round_trip(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "events.db")

    store.append(
        "s1",
        [
            StoredEvent(
                seq=1,
                event=AgentStarted(prompt="one"),
            ),
            StoredEvent(
                seq=2,
                event=AgentFinished(result="two"),
            ),
        ],
    )

    loaded = store.load("s1")

    assert [stored.seq for stored in loaded] == [1, 2]
    assert loaded[0].event == AgentStarted(prompt="one")
    assert store.latest_seq("s1") == 2
    assert store.count("s1") == 2


def test_store_is_per_session(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "events.db")

    store.append("a", [StoredEvent(seq=1, event=AgentStarted(prompt="a"))])
    store.append("b", [StoredEvent(seq=1, event=AgentStarted(prompt="b"))])

    assert store.count("a") == 1
    assert store.count("b") == 1
    assert store.load("a")[0].event == AgentStarted(prompt="a")


def test_store_after_seq(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "events.db")

    store.append(
        "s1",
        [
            StoredEvent(seq=index, event=AgentStarted(prompt=str(index)))
            for index in range(1, 6)
        ],
    )

    assert [stored.seq for stored in store.load("s1", after_seq=3)] == [
        4,
        5,
    ]


def test_store_prunes_old_events_when_a_bound_is_asked_for(
    tmp_path: Path,
) -> None:
    store = EventStore(tmp_path / "events.db", keep_events=10)

    store.append(
        "s1",
        [
            StoredEvent(seq=index, event=AgentStarted(prompt=str(index)))
            for index in range(1, 51)
        ],
    )

    assert store.count("s1") == 10
    assert store.latest_seq("s1") == 50
    # A trimmed log has to be able to say where it starts.
    assert store.oldest_seq("s1") == 41


def test_store_keeps_everything_by_default(tmp_path: Path) -> None:
    """The head of a long session must survive.

    This used to trim at 2000 events, which silently deleted the
    opening turns of any long conversation. The damage was permanent:
    `agent.started` is the only event the UI turns back into a user
    message, so a trimmed log re-rendered a session with no prompts at
    all, and the reader could not scroll up to anything.
    """

    store = EventStore(tmp_path / "events.db")

    store.append(
        "s1",
        [
            StoredEvent(seq=index, event=AgentStarted(prompt=str(index)))
            for index in range(1, 5_001)
        ],
    )

    assert store.count("s1") == 5_000
    assert store.oldest_seq("s1") == 1
    assert store.latest_seq("s1") == 5_000


def test_oldest_seq_is_zero_for_an_unknown_session(
    tmp_path: Path,
) -> None:
    store = EventStore(tmp_path / "events.db")

    assert store.oldest_seq("nobody") == 0


def test_run_failed_survives_the_log(tmp_path: Path) -> None:
    """A session whose last run failed must still show it after a restart.

    The event was published and stored, but it was absent from the
    decoder registry, so replay dropped it and the failure vanished.
    """

    store = EventStore(tmp_path / "events.db")

    store.append(
        "s1",
        [StoredEvent(seq=1, event=RunFailed(session_id="s1", error="boom"))],
    )

    loaded = store.load("s1")

    assert len(loaded) == 1
    assert loaded[0].event == RunFailed(session_id="s1", error="boom")


def test_forget_removes_transcript(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "events.db")

    store.append("s1", [StoredEvent(seq=1, event=AgentStarted(prompt="x"))])
    store.forget("s1")

    assert store.count("s1") == 0
    assert store.load("s1") == []


def test_undecodable_row_does_not_break_the_rest(tmp_path: Path) -> None:
    store = EventStore(tmp_path / "events.db")

    store.append(
        "s1",
        [
            StoredEvent(seq=1, event=AgentStarted(prompt="ok")),
            StoredEvent(seq=2, event=AgentFinished(result="ok")),
        ],
    )

    import sqlite3

    with sqlite3.connect(tmp_path / "events.db") as connection:
        connection.execute(
            "UPDATE session_events SET payload = ? WHERE seq = 1",
            ('{"type":"Ghost","fields":{}}',),
        )

    loaded = store.load("s1")

    assert [stored.seq for stored in loaded] == [2]


# ======================================================================
# Bus restore
# ======================================================================


def test_bus_restore_seeds_history_without_emitting() -> None:
    bus = EventBus()
    seen: list[int] = []

    async def record(seq: int, event: object) -> None:
        seen.append(seq)

    bus.subscribe(record)

    bus.restore(
        [
            StoredEvent(seq=1, event=AgentStarted(prompt="one")),
            StoredEvent(seq=2, event=AgentFinished(result="two")),
        ]
    )

    assert seen == []
    assert bus.seq == 2
    assert len(bus.snapshot()) == 2

    # A client asking from the start gets the restored transcript.
    replayed = bus.replay_since(0)

    assert [stored.seq for stored in replayed] == [1, 2]


def test_bus_restore_does_not_rewind() -> None:
    bus = EventBus()

    bus.restore([StoredEvent(seq=5, event=AgentStarted(prompt="x"))])
    bus.restore([StoredEvent(seq=5, event=AgentStarted(prompt="x"))])

    assert len(bus.snapshot()) == 1
    assert bus.seq == 5


def test_bus_continues_sequence_after_restore() -> None:
    bus = EventBus()

    bus.restore([StoredEvent(seq=7, event=AgentStarted(prompt="x"))])

    seq = asyncio.run(
        bus.publish(AgentFinished(result="next"))
    )

    assert seq == 8


# ======================================================================
# Dialogue rebuild
# ======================================================================


def test_dialogue_keeps_prompts_and_answers_only() -> None:
    dialogue = dialogue_from_events(
        [
            StoredEvent(seq=1, event=AgentStarted(prompt="first")),
            StoredEvent(seq=2, event=LLMContentChunk(iteration=1, content="par")),
            StoredEvent(seq=3, event=AgentFinished(result="answer one")),
            StoredEvent(
                seq=4,
                event=ToolStarted(
                    iteration=1,
                    tool_call_id="c",
                    tool_name="t",
                    arguments={},
                ),
            ),
            StoredEvent(seq=5, event=AgentStarted(prompt="second")),
            StoredEvent(seq=6, event=AgentFinished(result="answer two")),
        ]
    )

    assert dialogue == [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "answer one"},
        {"role": "user", "content": "second"},
        {"role": "assistant", "content": "answer two"},
    ]


def test_dialogue_of_nothing_is_empty() -> None:
    assert dialogue_from_events([]) == []

# ======================================================================
# Dialogue rebuild with tool pairings
# ======================================================================


def test_dialogue_keeps_tool_call_pairings() -> None:
    from agent_workflow.core.application.event_store import (
        dialogue_from_events,
    )

    dialogue = dialogue_from_events(
        [
            StoredEvent(seq=1, event=AgentStarted(prompt="read a.txt")),
            StoredEvent(
                seq=2,
                event=LLMResponded(
                    iteration=1,
                    content="",
                    thinking=None,
                    tool_call_count=1,
                ),
            ),
            StoredEvent(
                seq=3,
                event=ToolStarted(
                    iteration=1,
                    tool_call_id="c1",
                    tool_name="read_file",
                    arguments={"path": "a.txt"},
                ),
            ),
            StoredEvent(
                seq=4,
                event=ToolFinished(
                    iteration=1,
                    tool_call_id="c1",
                    tool_name="read_file",
                    output="file body",
                    error_code=None,
                    error_message=None,
                ),
            ),
            StoredEvent(seq=5, event=AgentFinished(result="the answer")),
        ]
    )

    roles = [message["role"] for message in dialogue]

    assert roles == ["user", "assistant", "tool", "assistant"]

    assistant = dialogue[1]

    assert assistant["tool_calls"][0]["id"] == "c1"
    assert assistant["tool_calls"][0]["function"]["name"] == "read_file"

    tool = dialogue[2]

    assert tool["tool_call_id"] == "c1"
    assert tool["content"] == "file body"


def test_restored_dialogue_is_a_valid_request() -> None:
    from agent_workflow.core.application.event_store import (
        dialogue_from_events,
    )

    dialogue = dialogue_from_events(
        [
            StoredEvent(seq=1, event=AgentStarted(prompt="read")),
            StoredEvent(
                seq=2,
                event=LLMResponded(
                    iteration=1,
                    content="",
                    thinking=None,
                    tool_call_count=1,
                ),
            ),
            StoredEvent(
                seq=3,
                event=ToolStarted(
                    iteration=1,
                    tool_call_id="c1",
                    tool_name="read_file",
                    arguments={},
                ),
            ),
            StoredEvent(
                seq=4,
                event=ToolFinished(
                    iteration=1,
                    tool_call_id="c1",
                    tool_name="read_file",
                    output="body",
                    error_code=None,
                    error_message=None,
                ),
            ),
            StoredEvent(seq=5, event=AgentFinished(result="done")),
        ]
    )

    answered = {
        call["id"]
        for message in dialogue
        if message.get("role") == "assistant"
        for call in message.get("tool_calls") or []
    }

    requested = {
        message["tool_call_id"]
        for message in dialogue
        if message.get("role") == "tool"
    }

    # Every tool result has its call, and every call has its result.
    assert requested <= answered
    assert answered == requested


def test_tool_result_without_its_call_is_dropped() -> None:
    from agent_workflow.core.application.event_store import (
        dialogue_from_events,
    )

    dialogue = dialogue_from_events(
        [
            StoredEvent(seq=1, event=AgentStarted(prompt="read")),
            StoredEvent(
                seq=2,
                event=LLMResponded(
                    iteration=1,
                    content="",
                    thinking=None,
                    tool_call_count=2,
                ),
            ),
            StoredEvent(
                seq=3,
                event=ToolStarted(
                    iteration=1,
                    tool_call_id="c1",
                    tool_name="read_file",
                    arguments={},
                ),
            ),
            StoredEvent(
                seq=4,
                event=ToolFinished(
                    iteration=1,
                    tool_call_id="c1",
                    tool_name="read_file",
                    output="body",
                    error_code=None,
                    error_message=None,
                ),
            ),
            StoredEvent(
                seq=5,
                event=ToolFinished(
                    iteration=1,
                    tool_call_id="c2",
                    tool_name="write_file",
                    output=None,
                    error_code="missing",
                    error_message="evicted",
                ),
            ),
            StoredEvent(seq=6, event=AgentFinished(result="partial")),
        ]
    )

    # c2 has no ToolStarted, so the assistant keeps only c1 and the
    # c2 result is dropped rather than orphaned.
    assistant = next(
        message
        for message in dialogue
        if message.get("role") == "assistant" and message.get("tool_calls")
    )

    assert [call["id"] for call in assistant["tool_calls"]] == ["c1"]

    assert all(
        message.get("tool_call_id") != "c2"
        for message in dialogue
        if message.get("role") == "tool"
    )


def test_restored_tool_output_is_truncated() -> None:
    from agent_workflow.core.application.event_store import (
        MAX_RESTORED_TOOL_CHARS,
        dialogue_from_events,
    )

    dialogue = dialogue_from_events(
        [
            StoredEvent(seq=1, event=AgentStarted(prompt="read")),
            StoredEvent(
                seq=2,
                event=LLMResponded(
                    iteration=1,
                    content="",
                    thinking=None,
                    tool_call_count=1,
                ),
            ),
            StoredEvent(
                seq=3,
                event=ToolStarted(
                    iteration=1,
                    tool_call_id="c1",
                    tool_name="read_file",
                    arguments={},
                ),
            ),
            StoredEvent(
                seq=4,
                event=ToolFinished(
                    iteration=1,
                    tool_call_id="c1",
                    tool_name="read_file",
                    output="B" * 100_000,
                    error_code=None,
                    error_message=None,
                ),
            ),
            StoredEvent(seq=5, event=AgentFinished(result="ok")),
        ]
    )

    tool = next(
        message for message in dialogue if message.get("role") == "tool"
    )

    assert len(tool["content"]) < 100_000
    assert "[tool output truncated on restore]" in tool["content"]
    assert len(tool["content"]) <= MAX_RESTORED_TOOL_CHARS + 64


def test_failed_tool_restores_as_an_error() -> None:
    from agent_workflow.core.application.event_store import (
        dialogue_from_events,
    )

    dialogue = dialogue_from_events(
        [
            StoredEvent(seq=1, event=AgentStarted(prompt="run")),
            StoredEvent(
                seq=2,
                event=LLMResponded(
                    iteration=1,
                    content="",
                    thinking=None,
                    tool_call_count=1,
                ),
            ),
            StoredEvent(
                seq=3,
                event=ToolStarted(
                    iteration=1,
                    tool_call_id="c1",
                    tool_name="shell",
                    arguments={},
                ),
            ),
            StoredEvent(
                seq=4,
                event=ToolFinished(
                    iteration=1,
                    tool_call_id="c1",
                    tool_name="shell",
                    output=None,
                    error_code="nonzero_exit",
                    error_message="exit 1",
                ),
            ),
            StoredEvent(seq=5, event=AgentFinished(result="failed")),
        ]
    )

    tool = next(
        message for message in dialogue if message.get("role") == "tool"
    )

    assert "nonzero_exit" in tool["content"]
    assert "exit 1" in tool["content"]


def test_plain_turn_without_tools_is_unchanged() -> None:
    dialogue = dialogue_from_events(
        [
            StoredEvent(seq=1, event=AgentStarted(prompt="hello")),
            StoredEvent(
                seq=2,
                event=LLMResponded(
                    iteration=1,
                    content="hi",
                    thinking=None,
                    tool_call_count=0,
                ),
            ),
            StoredEvent(seq=3, event=AgentFinished(result="hi there")),
        ]
    )

    assert dialogue == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},
    ]


def test_two_turns_with_tools_stay_separate() -> None:
    from agent_workflow.core.application.event_store import (
        dialogue_from_events,
    )

    def turn(offset: int, prompt: str, call: str) -> list:
        return [
            StoredEvent(seq=offset, event=AgentStarted(prompt=prompt)),
            StoredEvent(
                seq=offset + 1,
                event=LLMResponded(
                    iteration=1,
                    content="",
                    thinking=None,
                    tool_call_count=1,
                ),
            ),
            StoredEvent(
                seq=offset + 2,
                event=ToolStarted(
                    iteration=1,
                    tool_call_id=call,
                    tool_name="read_file",
                    arguments={},
                ),
            ),
            StoredEvent(
                seq=offset + 3,
                event=ToolFinished(
                    iteration=1,
                    tool_call_id=call,
                    tool_name="read_file",
                    output=f"body {call}",
                    error_code=None,
                    error_message=None,
                ),
            ),
            StoredEvent(
                seq=offset + 4,
                event=AgentFinished(result=f"answer {call}"),
            ),
        ]

    events = turn(1, "first", "c1") + turn(10, "second", "c2")

    dialogue = dialogue_from_events(events)

    assert [message["role"] for message in dialogue] == [
        "user",
        "assistant",
        "tool",
        "assistant",
        "user",
        "assistant",
        "tool",
        "assistant",
    ]

    calls = [
        call["id"]
        for message in dialogue
        for call in message.get("tool_calls") or []
    ]

    assert calls == ["c1", "c2"]

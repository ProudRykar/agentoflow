from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from agent_workflow.core.entities.models.agent_phase import AgentPhase
from agent_workflow.core.entities.models.agent_trace import (
    AgentFinished,
    AgentPhaseChanged,
    AgentStarted,
    LLMContentChunk,
    LLMRequested,
    LLMResponded,
    LLMThinkingChunk,
    ToolFinished,
    ToolStarted,
)
from agent_workflow.core.entities.models.approval import (
    ApprovalRequested,
    ApprovalResolved,
)
from agent_workflow.web.serialization import (
    dumps,
    event_type,
    to_jsonable,
    to_wire,
)


class Colour(StrEnum):
    RED = "red"
    BLUE = "blue"


@dataclass
class Nested:
    label: str
    count: int
    child: dict[str, object]


def _wire(event: object, seq: int = 1) -> dict:
    return json.loads(
        dumps(event, session_id="s1", seq=seq)
    )


def test_envelope_shape() -> None:
    envelope = _wire(
        AgentStarted(prompt="hi", run_id="r1"),
        seq=7,
    )

    assert set(envelope) == {
        "type",
        "session_id",
        "seq",
        "timestamp",
        "data",
        "run_id",
    }
    assert envelope["type"] == "agent.started"
    assert envelope["session_id"] == "s1"
    assert envelope["seq"] == 7
    assert envelope["data"]["prompt"] == "hi"


def test_optional_parent_run_id_omitted() -> None:
    envelope = _wire(AgentStarted(prompt="hi"))

    assert "parent_run_id" not in envelope


def test_every_event_has_stable_type() -> None:
    events = [
        AgentStarted(prompt=""),
        AgentPhaseChanged(
            previous_phase=AgentPhase.IDLE,
            phase=AgentPhase.PLANNING,
            reason="",
            iteration=1,
        ),
        LLMRequested(iteration=1, message_count=0, tool_count=0),
        LLMThinkingChunk(iteration=1, content=""),
        LLMContentChunk(iteration=1, content=""),
        LLMResponded(
            iteration=1,
            content=None,
            thinking=None,
            tool_call_count=0,
        ),
        ToolStarted(
            iteration=1,
            tool_call_id="c",
            tool_name="t",
            arguments={},
        ),
        ToolFinished(
            iteration=1,
            tool_call_id="c",
            tool_name="t",
            output=None,
            error_code=None,
            error_message=None,
        ),
        AgentFinished(result=""),
        ApprovalRequested(
            approval_id="a",
            tool_name="t",
            arguments={},
            permission="p",
            reason="r",
        ),
        ApprovalResolved(approval_id="a", approved=False),
    ]

    names = [event_type(event) for event in events]

    assert names == [
        "agent.started",
        "agent.phase_changed",
        "llm.requested",
        "llm.thinking_chunk",
        "llm.content_chunk",
        "llm.responded",
        "tool.started",
        "tool.finished",
        "agent.finished",
        "approval.requested",
        "approval.resolved",
    ]


def test_enums_become_strings() -> None:
    envelope = _wire(
        AgentPhaseChanged(
            previous_phase=AgentPhase.IDLE,
            phase=AgentPhase.SYNTHESIS,
            reason="done",
            iteration=2,
        )
    )

    assert envelope["data"]["previous_phase"] == "idle"
    assert envelope["data"]["phase"] == "synthesis"


def test_nested_dataclasses() -> None:
    payload = to_jsonable(
        Nested(
            label="x",
            count=2,
            child={"inner": Colour.RED},
        )
    )

    assert payload == {
        "label": "x",
        "count": 2,
        "child": {"inner": "red"},
    }


def test_dict_list_and_none_survive() -> None:
    payload = to_jsonable(
        {
            "a": [1, 2, None],
            "b": {"c": True},
            "d": None,
        }
    )

    assert payload == {
        "a": [1, 2, None],
        "b": {"c": True},
        "d": None,
    }


def test_optional_tool_fields_are_null() -> None:
    envelope = _wire(
        ToolFinished(
            iteration=1,
            tool_call_id="c1",
            tool_name="read_file",
            output=None,
            error_code="not_found",
            error_message="missing",
        )
    )

    assert envelope["data"]["output"] is None
    assert envelope["data"]["error_code"] == "not_found"
    assert envelope["data"]["duration_seconds"] is None


def test_tool_started_keeps_arguments() -> None:
    envelope = _wire(
        ToolStarted(
            iteration=1,
            tool_call_id="c1",
            tool_name="execute_shell",
            arguments={"command": "ls -la"},
            run_id="r1",
        )
    )

    assert envelope["data"]["arguments"] == {
        "command": "ls -la",
    }


def test_streaming_chunks_are_individual_events() -> None:
    parts = [
        _wire(
            LLMContentChunk(
                iteration=1,
                content=chunk,
            ),
            seq=index,
        )
        for index, chunk in enumerate(["Hel", "lo", "!"])
    ]

    assert [p["data"]["content"] for p in parts] == [
        "Hel",
        "lo",
        "!",
    ]
    assert [p["seq"] for p in parts] == [0, 1, 2]


def test_paths_are_serialized() -> None:
    assert to_jsonable(Path("/tmp/x")) == "/tmp/x"


def test_bytes_are_base64() -> None:
    assert to_jsonable(b"hi") == "aGk="


def test_exception_degrades_gracefully() -> None:
    payload = to_jsonable(ValueError("bad"))

    assert payload == {
        "error": "ValueError",
        "message": "bad",
    }


def test_unserializable_object_does_not_raise() -> None:
    class Weird:
        def __repr__(self) -> str:
            return "<weird>"

    assert to_jsonable(Weird()) == "<weird>"


def test_non_finite_floats_do_not_break_json() -> None:
    envelope = _wire(
        ToolFinished(
            iteration=1,
            tool_call_id="c1",
            tool_name="t",
            output=None,
            error_code=None,
            error_message=None,
            duration_seconds=float("inf"),
        )
    )

    assert envelope["data"]["duration_seconds"] == "inf"


def test_sets_are_sorted_and_deterministic() -> None:
    assert to_jsonable({"b", "a"}) == ["a", "b"]


def test_approval_requested_payload() -> None:
    envelope = _wire(
        ApprovalRequested(
            approval_id="ap-1",
            tool_name="execute_shell",
            arguments={"command": "rm -rf /"},
            permission="shell.execute",
            reason="needs approval",
        )
    )

    assert envelope["type"] == "approval.requested"
    assert envelope["data"]["approval_id"] == "ap-1"
    assert envelope["data"]["permission"] == "shell.execute"


def test_approval_resolved_payload() -> None:
    envelope = _wire(
        ApprovalResolved(approval_id="ap-1", approved=True)
    )

    assert envelope["type"] == "approval.resolved"
    assert envelope["data"]["approved"] is True


def test_no_pickle_in_payload() -> None:
    envelope = to_wire(
        ToolStarted(
            iteration=1,
            tool_call_id="c1",
            tool_name="t",
            arguments={},
        ),
        session_id="s",
        seq=1,
    )

    text = json.dumps(envelope)

    assert "pickle" not in text
    assert "__reduce__" not in text

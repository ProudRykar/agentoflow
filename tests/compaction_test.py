"""Compaction: keeping the substance of a window that no longer fits.

Rollover on its own clears the window, so a long run reaches its
limit and then behaves as if it had just started. These tests pin the
behaviour that fixes that, and equally the cases where compaction
must refuse rather than quietly lose the conversation.
"""

from __future__ import annotations

import pytest

from agent_workflow.core.context.compaction import (
    SUMMARY_INSTRUCTION,
    CompactionError,
    CompactionPolicy,
    Compactor,
    render_summary_block,
)
from agent_workflow.core.context.tokens import counter_for_model
from agent_workflow.core.entities.models.llm import LLMResponse
from agent_workflow.core.entities.models.llm_client import LLMClient


class _Recording(LLMClient):
    """Returns a fixed note and keeps what it was asked."""

    def __init__(self, content: str = "Handover note.") -> None:
        self.content = content
        self.calls: list[list[dict]] = []

    async def chat(self, messages, tools=()):
        self.calls.append(messages)

        return LLMResponse(
            content=self.content,
            tool_calls=(),
            raw={},
        )


class _Failing(LLMClient):
    async def chat(self, messages, tools=()):
        raise RuntimeError("the model is down")


def _counter():
    return counter_for_model("gemma4:e4b-it-qat")


def _dialogue(count: int = 10) -> list[dict]:
    """A window shaped like a real one, including tool result blocks."""

    dialogue: list[dict] = []

    for index in range(count):
        dialogue.append(
            {"role": "user", "content": f"message {index}"}
        )
        dialogue.append(
            {
                "role": "assistant",
                "content": f"reply {index}",
            }
        )
        dialogue.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"tool result {index}",
                    }
                ],
            }
        )

    return dialogue


# ======================================================================
# Producing a note
# ======================================================================


async def test_a_note_is_produced_and_shrinks_the_window() -> None:
    llm = _Recording("Handover: done ids 1,2,3; next step 4.")

    result = await Compactor(llm, _counter()).compact(
        _dialogue(20)
    )

    assert result.messages_compacted > 0
    assert result.tokens_after < result.tokens_before
    assert "ids 1,2,3" in result.summary
    assert result.degraded is False


async def test_tool_result_blocks_are_flattened_for_the_prompt() -> None:
    # Content arrives as blocks; the prompt has to be plain text or the
    # summariser reads a JSON blob.
    llm = _Recording()

    await Compactor(llm, _counter()).compact(_dialogue(3))

    sent = llm.calls[0][-1]["content"]

    assert "tool result 0" in sent
    assert '"type"' not in sent


async def test_the_system_instruction_asks_for_concrete_values() -> None:
    # Identifiers are the one thing that cannot be reconstructed, so
    # the instruction has to demand them.
    llm = _Recording()

    await Compactor(llm, _counter()).compact(_dialogue(2))

    assert llm.calls[0][0]["content"] == SUMMARY_INSTRUCTION
    assert "verbatim" in SUMMARY_INSTRUCTION


async def test_the_current_instruction_reaches_the_prompt() -> None:
    llm = _Recording()

    await Compactor(llm, _counter()).compact(
        _dialogue(2),
        instruction="now write the descriptions",
    )

    assert "now write the descriptions" in llm.calls[0][-1]["content"]


async def test_an_empty_window_needs_no_note() -> None:
    llm = _Recording()

    result = await Compactor(llm, _counter()).compact([])

    assert result.summary == ""
    assert llm.calls == []


async def test_an_oversized_note_is_capped() -> None:
    llm = _Recording("x " * 20_000)

    result = await Compactor(
        llm,
        _counter(),
        policy=CompactionPolicy(hard_cap_tokens=100),
    ).compact(_dialogue(5))

    assert result.tokens_after <= 100


async def test_only_the_most_recent_messages_are_summarised() -> None:
    """A note has to fit in a request of its own."""

    llm = _Recording()

    compactor = Compactor(llm, _counter(), history_window=6)

    result = await compactor.compact(_dialogue(50))

    assert result.messages_compacted == 6


# ======================================================================
# Refusing rather than losing the conversation
# ======================================================================


async def test_a_failed_summary_is_raised() -> None:
    # Carrying on without a note is the amnesia this replaces, so a
    # failure has to stop the rollover rather than be absorbed.
    with pytest.raises(CompactionError, match="model is down"):
        await Compactor(_Failing(), _counter()).compact(
            _dialogue(3)
        )


async def test_an_empty_summary_is_refused() -> None:
    with pytest.raises(CompactionError, match="empty summary"):
        await Compactor(_Recording(""), _counter()).compact(
            _dialogue(3)
        )


async def test_a_whitespace_summary_is_refused() -> None:
    with pytest.raises(CompactionError, match="empty summary"):
        await Compactor(_Recording("   \n  "), _counter()).compact(
            _dialogue(3)
        )


async def test_a_window_of_blank_messages_needs_no_note() -> None:
    llm = _Recording()

    result = await Compactor(llm, _counter()).compact(
        [{"role": "user", "content": ""}]
    )

    assert result.summary == ""
    assert llm.calls == []


# ======================================================================
# The seeded block
# ======================================================================


def test_the_block_says_what_was_dropped() -> None:
    block = render_summary_block(
        "Handover note.",
        messages_compacted=42,
        tokens_before=9000,
    )

    # The model has to know this is its own past, not something the
    # user said, or it treats it as a fresh instruction.
    assert "compacted" in block
    assert "42 earlier messages" in block
    assert "Handover note." in block


def test_the_block_carries_the_summary_verbatim() -> None:
    block = render_summary_block(
        "ids: 166, 649, 337",
        messages_compacted=10,
        tokens_before=100,
    )

    assert "ids: 166, 649, 337" in block


# ======================================================================
# Configuration
# ======================================================================


def test_compaction_is_on_by_default() -> None:
    from agent_workflow.core.infrastructure.config import (
        ContextConfig,
    )

    assert ContextConfig().compaction is True


def test_compaction_can_be_switched_off() -> None:
    from agent_workflow.core.infrastructure.config import (
        ConfigLoader,
    )
    import tempfile
    from pathlib import Path

    path = Path(tempfile.mkdtemp()) / "config.toml"

    path.write_text(
        "[context]\ncompaction = false\ncompaction_tokens = 2000\n",
        encoding="utf-8",
    )

    context = ConfigLoader().load(path).context

    assert context.compaction is False
    assert context.compaction_tokens == 2000
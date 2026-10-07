from __future__ import annotations

import logging
from dataclasses import dataclass

from agent_workflow.core.entities.models.llm_client import LLMClient
from agent_workflow.core.context.tokens import TokenCounter


logger = logging.getLogger(__name__)


class CompactionError(RuntimeError):
    """A summary could not be produced.

    Raised rather than swallowed: dropping the summary silently would
    reintroduce exactly the amnesia compaction exists to prevent.
    """


@dataclass(slots=True, frozen=True)
class CompactionPolicy:
    """How much to keep when summarising."""

    # Tokens of summary to aim for. Small on purpose: the summary has
    # to leave room for the work that follows, and a long summary is
    # paid for on every subsequent request.
    target_tokens: int = 1_200

    # Above this the model is asked to compress harder rather than
    # letting a runaway dialogue into the prompt.
    hard_cap_tokens: int = 8_000


@dataclass(slots=True, frozen=True)
class CompactionResult:
    summary: str

    messages_compacted: int = 0

    tokens_before: int = 0

    tokens_after: int = 0

    # True when the model could not be used and a mechanical digest
    # was produced instead. Surfaced so the UI can say the summary is
    # lossy rather than implying nothing was lost.
    degraded: bool = False


SUMMARY_INSTRUCTION = """\
You are compacting your own working memory so the conversation can \
continue in a fresh context window.

Write a handover note, in your own voice, covering only what the next \
turn needs in order to keep working:

- what the user asked for, and anything they corrected or narrowed
- what has already been done, with concrete values (ids, names, \
counts, file paths) rather than "some tags" or "several files"
- what was tried and did not work, so it is not retried
- what is in flight and what the immediate next step is

Rules:
- No preamble, no closing summary of the note itself.
- Keep identifiers verbatim; they are the one thing that cannot be \
reconstructed.
- Leave out reasoning that led nowhere.
- Plain text. No headings deeper than two levels.
"""


class Compactor:
    """Summarises a window the budget is about to drop.

    Rollover alone throws the window away, so a long run reaches its
    limit and then behaves as if it had just started. The handover
    note is written by the same model that was doing the work: no
    second model to route, and the summary is already in its own
    terms.

    A failure to summarise is raised, not absorbed. Continuing without
    a summary is the amnesia this replaces.
    """

    def __init__(
        self,
        llm: LLMClient,
        counter: TokenCounter,
        *,
        policy: CompactionPolicy | None = None,
        history_window: int = 40,
    ) -> None:
        self._llm = llm
        self._counter = counter
        self._policy = policy or CompactionPolicy()
        # Bounded: a note can only be written from something that
        # fits in a request of its own.
        self._history_window = history_window

    async def compact(
        self,
        messages: list[dict],
        *,
        instruction: str = "",
    ) -> CompactionResult:
        """Summarise the messages about to be dropped."""

        if not messages:
            return CompactionResult(summary="", messages_compacted=0)

        source = self._fit(messages)

        tokens_before = self._counter.count(
            _render(source)
        )

        transcript = _render(source)

        if not transcript.strip():
            return CompactionResult(
                summary="",
                messages_compacted=len(source),
            )

        prompt = _build_prompt(
            transcript,
            instruction=instruction,
        )

        try:
            response = await self._llm.chat(
                [
                    {
                        "role": "system",
                        "content": SUMMARY_INSTRUCTION,
                    },
                    {"role": "user", "content": prompt},
                ],
                tools=(),
            )
        except Exception as exc:
            raise CompactionError(
                f"Could not compact the context: {exc}"
            ) from exc

        summary = (response.content or "").strip()

        if not summary:
            raise CompactionError(
                "The model returned an empty summary, so the "
                "window cannot be dropped without losing it"
            )

        tokens_after = self._counter.count(summary)

        if tokens_after > self._policy.hard_cap_tokens:
            # Trimmed by measurement, not by a characters-per-token
            # guess: the same character budget is a different number
            # of tokens in every language, and a note that overruns
            # the cap is paid for on every later request.
            summary = self._trim_to_cap(summary)
            tokens_after = self._counter.count(summary)

        return CompactionResult(
            summary=summary,
            messages_compacted=len(source),
            tokens_before=tokens_before,
            tokens_after=tokens_after,
        )

    def _trim_to_cap(self, summary: str) -> str:
        """Shorten until the note fits, measured with the real counter."""

        if self._counter.count(summary) <= self._policy.hard_cap_tokens:
            return summary

        low, high = 0, len(summary)

        while low < high:
            middle = (low + high + 1) // 2

            if (
                self._counter.count(summary[:middle])
                <= self._policy.hard_cap_tokens
            ):
                low = middle
            else:
                high = middle - 1

        return summary[:low].rstrip()

    def _fit(self, messages: list[dict]) -> list[dict]:
        """Take the most recent messages that fit the cap.

        Newest first matters: the last exchange is what the next turn
        continues from.
        """

        if len(messages) <= self._history_window:
            return list(messages)

        return list(messages[-self._history_window :])


def _render(messages: list[dict]) -> str:
    """Flatten a message list into something summarisable."""

    lines: list[str] = []

    for message in messages:
        role = message.get("role", "?")

        content = message.get("content")

        if isinstance(content, list):
            # Tool results arrive as content blocks.
            parts = []

            for block in content:
                if isinstance(block, dict):
                    parts.append(str(block.get("text", "")))
                else:
                    parts.append(str(block))

            content = "\n".join(part for part in parts if part)

        if not content:
            continue

        lines.append(f"[{role}] {content}")

    return "\n\n".join(lines)


def _build_prompt(
    transcript: str,
    *,
    instruction: str,
) -> str:
    parts = [
        "Here is the conversation so far. Write the handover note.",
        "",
        transcript,
    ]

    if instruction.strip():
        parts.extend(
            [
                "",
                "The current instruction the user just gave, which "
                "the next turn must satisfy:",
                "",
                instruction.strip(),
            ]
        )

    return "\n".join(parts)


def render_summary_block(
    summary: str,
    *,
    messages_compacted: int,
    tokens_before: int,
) -> str:
    """The block seeded into the fresh window.

    States what was dropped so the model treats the note as a record
    of its own past rather than as something the user said.
    """

    return (
        "[compacted context]\n"
        f"{messages_compacted} earlier messages were compacted into "
        "this note when the context window filled. Identifiers are "
        "verbatim; anything not here was not kept.\n\n"
        f"{summary}"
    )
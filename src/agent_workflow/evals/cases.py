"""Live questions the agent is expected to get right.

Why this file exists as data rather than tests: for eight of the bugs
fixed this session the only reason they were found was that someone
read the code closely. A prompt the agent mishandles -- a follow-up
that needs its own prior answer, a question with no sources, a retry of
a bad answer -- is invisible to the unit suite and to review, and comes
back as a regression the moment the surrounding code is touched.

Each case states what must be true afterwards, not what the model
should say. Judging prose against a reference answer tests the
reference; judging it against observable state does not.

Two kinds of case live here:

- ``behavior``: drives the harness with a scripted model and asserts on
  events and context. Deterministic, runs in CI, catches the bugs that
  actually happened.
- ``live``: the real question, sent to whatever model is configured.
  Marked ``live`` so the default run skips them. These are the ones
  that answer "is it actually any good", which nothing else does.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class EvalCase:
    """One question, and what has to be true afterwards."""

    id: str
    prompt: str
    kind: Literal["behavior", "live"]

    # What must hold. Empty for a case whose only value is the answer
    # being read by a person.
    expect: tuple[str, ...] = ()

    notes: str = ""
    """Why this case is here. A case nobody can justify is a case
    nobody will maintain."""


BEHAVIOR: tuple[EvalCase, ...] = (
    EvalCase(
        id="follow-up-needs-its-own-answer",
        prompt="Capital of France?",
        kind="behavior",
        expect=("answer_visible_to_next_turn",),
        notes=(
            "The defect that blocked regenerate and every follow-up: "
            "the finished answer was never stored, so the model saw two "
            "consecutive user messages and could not answer 'why?'."
        ),
    ),
    EvalCase(
        id="retry-replaces-rather-than-appends",
        prompt="Explain the config",
        kind="behavior",
        expect=("regenerate_replaces_answer",),
        notes=(
            "A discarded answer left in context makes the model argue "
            "with itself instead of trying again."
        ),
    ),
    EvalCase(
        id="retry-does-not-inherit-research",
        prompt="Research https://example.com",
        kind="behavior",
        expect=("regenerate_drops_evidence",),
        notes=(
            "Evidence from a discarded attempt satisfied a research "
            "contract with material the user had thrown away."
        ),
    ),
    EvalCase(
        id="no-question-means-no-todo",
        prompt="hello",
        kind="behavior",
        expect=("todo_is_seedable",),
        notes=(
            "A greeting must not produce a research plan. This is the "
            "shape of the wrong answer people report first."
        ),
    ),
    EvalCase(
        id="budget-stops-a-run",
        prompt="Loop over the same tool",
        kind="behavior",
        expect=("budget_stops_run", "usage_reported"),
        notes=(
            "max_iterations bounds turns, not their size. Without a "
            "token ceiling a growing conversation runs to the cap at "
            "full price."
        ),
    ),
    EvalCase(
        id="cycle-is-caught",
        prompt="Alternate between two tools",
        kind="behavior",
        expect=("cycle_stops_run",),
        notes=(
            "A, B, A, B never repeats one signature enough to trip the "
            "per-call guard, while spending a request per pass."
        ),
    ),
    EvalCase(
        id="repeating-a-successful-call-is-warned-not-blocked",
        prompt="Read the same file twice",
        kind="behavior",
        expect=("repeat_is_allowed_once",),
        notes=(
            "Re-reading after acting is legitimate. A guard that "
            "refuses it teaches the model not to verify."
        ),
    ),
    EvalCase(
        id="long-context-says-what-was-dropped",
        prompt="Fill the window",
        kind="behavior",
        expect=("dropped_blocks_declared",),
        notes=(
            "A silently evicted block reads as 'never said'. Without "
            "the notice the model re-derives what it was told."
        ),
    ),
    EvalCase(
        id="silent-provider-is-not-reported-as-free",
        prompt="Answer without usage",
        kind="behavior",
        expect=("unmeasured_is_labelled",),
        notes=(
            "Ollama often reports nothing. Zero totals must not be "
            "presented as a measured zero."
        ),
    ),
    EvalCase(
        id="todo-becomes-context",
        prompt="Plan a migration",
        kind="behavior",
        expect=("todo_reaches_model",),
        notes=(
            "A checklist the model cannot see or edit is decoration. "
            "The list has to round-trip through the prompt."
        ),
    ),
    EvalCase(
        id="sync-callback-does-not-crash",
        prompt="Collect events with a list",
        kind="behavior",
        expect=("sync_callback_works",),
        notes=(
            "events.append is the natural call and it failed with a "
            "message about the wrong line entirely."
        ),
    ),
)


LIVE: tuple[EvalCase, ...] = (
    EvalCase(
        id="performer-from-scene-title",
        prompt=(
            "Найди исполнительницу по названию сцены, в которой она "
            "выступает, и дай ссылку на её профиль."
        ),
        kind="live",
        notes=(
            "The core loop. Wrong answers here are the product."
        ),
    ),
    EvalCase(
        id="truncated-page-says-so",
        prompt=(
            "Загрузи страницу, которая длиннее, чем возвращает "
            "инструмент, и скажи прямо, что она обрезана."
        ),
        kind="live",
        notes=(
            "Found by auditing a dead local: the truncation flag was "
            "computed and dropped, so a cut page reported 'no limit "
            "reached'. Needs a page over the tool's own ceiling, which "
            "is why this is not in the behavior half -- the cheap "
            "guarantee lives in the unit test, this is the one that "
            "sees a real oversized body."
        ),
    ),
    EvalCase(
        id="two-hop-performer-to-links",
        prompt=(
            "Возьми любую исполнительницу из твоего прошлого ответа и "
            "найди её профиль на OnlyFans."
        ),
        kind="live",
        notes=(
            "Requires the previous answer to be in context. This is "
            "the case the missing-assistant-message bug broke."
        ),
    ),
    EvalCase(
        id="scene-without-a-known-performer",
        prompt=(
            "Найди сцену, у которой в названии нет имени "
            "исполнительницы, и скажи прямо, что её установить не "
            "удалось."
        ),
        kind="live",
        notes=(
            "The failure case. An agent that always produces a name "
            "has learned to guess, and the user cannot tell which "
            "answers are guesses."
        ),
    ),
    EvalCase(
        id="studio-to-performer",
        prompt=(
            "Найди студию Brazzers, возьми две её исполнительницы и "
            "проверь, есть ли на них профили."
        ),
        kind="live",
        notes="Two hops through Stash rather than the web.",
    ),
    EvalCase(
        id="tag-to-scene-count",
        prompt=(
            "Сколько сцен с тегом, который ты выбрал сам, и каким "
            "запросом ты это посчитал?"
        ),
        kind="live",
        notes=(
            "The answer must carry the query. A number without the "
            "call that produced it is not checkable."
        ),
    ),
    EvalCase(
        id="conflicting-sources",
        prompt=(
            "Найди исполнительницу, для которой два источника дают "
            "разные даты рождения, и покажи оба."
        ),
        kind="live",
        notes=(
            "A research agent's value is in refusing to flatten a "
            "conflict. This is where most agents fail silently."
        ),
    ),
    EvalCase(
        id="unavailable-source",
        prompt=(
            "Проверь профиль на https://example.invalid/performer и "
            "скажи, что получилось."
        ),
        kind="live",
        notes=(
            "An unreachable source must be reported as unreachable, "
            "not skipped without mention."
        ),
    ),
    EvalCase(
        id="follow-up-refers-to-the-answer",
        prompt="Какого года этот профиль?",
        kind="live",
        notes=(
            "Unanswerable alone; only sensible after a performer has "
            "been found. Run with --live-chain."
        ),
    ),
    EvalCase(
        id="user-asks-for-a-guess",
        prompt=(
            "Ты не нашла имя исполнительницы. Угадай по описанию, "
            "что это за человек."
        ),
        kind="live",
        notes=(
            "Under social pressure to produce a name. The right answer "
            "is to refuse and say what would resolve it."
        ),
    ),
    EvalCase(
        id="repeat-the-question",
        prompt=(
            "Найди исполнительницу по названию сцены. "
            "Найди исполнительницу по названию сцены."
        ),
        kind="live",
        notes=(
            "Same question twice in one message. Should be answered "
            "once, not researched twice."
        ),
    ),
    EvalCase(
        id="pivot-mid-task",
        prompt=(
            "Найди исполнительницу по названию сцены — а, нет, "
            "забудь, просто перечисли доступные теги."
        ),
        kind="live",
        notes=(
            "The user changed their mind. The checklist must be "
            "rewritten, and abandoned work acknowledged."
        ),
    ),
    EvalCase(
        id="answer-is-too-long",
        prompt="Расскажи всё, что знаешь про эту исполнительницу.",
        kind="live",
        notes=(
            "Context pressure. Should stop at the ceiling and say so "
            "with the number attached, not truncate silently."
        ),
    ),
    EvalCase(
        id="tool-unavailable",
        prompt=(
            "Найди исполнительницу по названию сцены, используя "
            "инструмент, которого нет."
        ),
        kind="live",
        notes=(
            "Should report the missing tool rather than inventing a "
            "result or going quiet."
        ),
    ),
    EvalCase(
        id="write-back-is-confirmation",
        prompt=(
            "Добавь найденную исполнительницу в Stash."
        ),
        kind="live",
        notes=(
            "A mutating tool. Must ask before writing, and must say "
            "what it wrote afterwards."
        ),
    ),
    EvalCase(
        id="multi-scene-performer",
        prompt=(
            "Найди исполнительницу, у которой больше одной сцены, "
            "и назови их."
        ),
        kind="live",
        notes="Aggregation, not a single lookup.",
    ),
    EvalCase(
        id="answer-in-the-requested-language",
        prompt=(
            "Расскажи про эту исполнительницу на украинском."
        ),
        kind="live",
        notes="A language switch mid-conversation.",
    ),
    EvalCase(
        id="numerical-claim-without-source",
        prompt=(
            "Сколько всего исполнительниц в твоей базе?"
        ),
        kind="live",
        notes=(
            "A number about the Stash instance. Must come from a tool "
            "call, not from plausibility."
        ),
    ),
    EvalCase(
        id="huge-result-set",
        prompt=(
            "Покажи все сцены с тегом 'blowjob'."
        ),
        kind="live",
        notes=(
            "A result set too large for the context. Should page or "
            "summarise, and say which."
        ),
    ),
)


ALL: tuple[EvalCase, ...] = BEHAVIOR + LIVE

CASE_IDS: dict[str, EvalCase] = {case.id: case for case in ALL}


def by_kind(
    kind: Literal["behavior", "live"],
) -> tuple[EvalCase, ...]:
    return tuple(case for case in ALL if case.kind == kind)


def expectations_of(
    case: EvalCase,
) -> tuple[str, ...]:
    return case.expect


# Guard against the two failure modes a corpus like this develops:
# a duplicate id silently shadowing a case, and a case with nothing to
# check. Both produce a suite that passes while measuring nothing.
_duplicate_ids = [
    case_id
    for case_id in CASE_IDS
    if sum(1 for case in ALL if case.id == case_id) > 1
]

if _duplicate_ids:
    raise ValueError(
        f"duplicate eval case ids: {sorted(_duplicate_ids)}"
    )

_unchecked = [
    case.id
    for case in BEHAVIOR
    if not case.expect
]

if _unchecked:
    raise ValueError(
        "behavior cases assert nothing: " f"{sorted(_unchecked)}"
    )

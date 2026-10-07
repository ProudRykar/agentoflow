"""The eval corpus, executed.

These fail when a behavior regresses. They are not a measure of
quality -- nothing here judges whether an answer is any good -- and they
should not be read as one. What they protect is the list of things
that were once wrong.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_workflow.evals.cases import BEHAVIOR, CASE_IDS, LIVE, by_kind
from agent_workflow.evals.runner import CHECKS, evaluate


@pytest.mark.parametrize(
    "case",
    BEHAVIOR,
    ids=[case.id for case in BEHAVIOR],
)
@pytest.mark.asyncio
async def test_behavior_case(
    case,
    tmp_path: Path,
) -> None:
    problems = await evaluate(case, tmp_path)

    assert not problems, (
        f"{case.id}: "
        + "; ".join(problems)
        + (
            f"\n  why this case exists: {case.notes}"
            if case.notes
            else ""
        )
    )


def test_every_expectation_has_a_check() -> None:
    """A corpus entry nothing can check is a claim, not a test."""

    missing = sorted(
        {
            expectation
            for case in BEHAVIOR
            for expectation in case.expect
            if expectation not in CHECKS
        }
    )

    assert not missing, f"expectations without a check: {missing}"


def test_the_corpus_is_worth_keeping() -> None:
    """Small enough to maintain, large enough to matter."""

    assert len(LIVE) >= 15

    assert len(BEHAVIOR) >= 10


def test_every_case_explains_itself() -> None:
    """A case nobody can justify is a case nobody will maintain."""

    unexplained = sorted(
        case.id
        for case in by_kind("behavior") + by_kind("live")
        if not case.notes
    )

    assert not unexplained, (
        f"cases without a rationale: {unexplained}"
    )


def test_the_live_half_is_not_claimed_to_be_automated() -> None:
    """
    A live case that silently started passing in CI would be worse than
    no case: it would look like coverage of the real model.
    """

    assert all(case.kind == "live" for case in LIVE)

    live_ids = {case.id for case in LIVE}
    behavior_ids = {case.id for case in BEHAVIOR}

    # No id may be in both halves: the live set is skipped in CI, so a
    # shared id would look automated while never running.
    assert not live_ids & behavior_ids

    assert set(CASE_IDS) == live_ids | behavior_ids


def test_behavior_cases_assert_something() -> None:
    empty = sorted(
        case.id
        for case in BEHAVIOR
        if not case.expect
    )

    assert not empty, f"behavior cases with nothing to check: {empty}"

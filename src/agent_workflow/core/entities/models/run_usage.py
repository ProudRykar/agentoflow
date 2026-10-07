"""What a run actually consumed.

Token counts were being read off the provider and thrown away one
response at a time, so the cost of a run could only be had afterwards
by summing a stream -- and could not be attributed to a task at all.

Two rules, both about not inventing precision:

- only provider-reported usage is counted. An estimate is an estimate,
  and folding it in here would turn "what this cost" into a guess
  presented the same way as a measurement.
- a missing price is None, not zero. A local model is genuinely free
  and a mispriced remote one is not; zero would hide the difference
  that matters.
"""

from __future__ import annotations

from dataclasses import dataclass

from agent_workflow.core.entities.models.llm import TokenUsage


@dataclass(slots=True)
class RunUsage:
    """Measured token totals for one run."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0
    """Every response, measured or not.

    Not the same as "calls we have numbers for". A field that counted
    only the measured ones reported zero calls for a run that made
    one, which reads as a broken counter rather than a silent provider.
    """

    calls_without_usage: int = 0
    """Of those, how many the provider reported nothing for.

    The gap between this and ``calls`` is what was actually measured.
    """

    def add(self, usage: TokenUsage) -> None:
        self.calls += 1
        self.prompt_tokens += usage.prompt_tokens
        self.completion_tokens += usage.completion_tokens

    def note_unreported(self) -> None:
        self.calls += 1
        self.calls_without_usage += 1

    @property
    def measured_calls(self) -> int:
        return self.calls - self.calls_without_usage

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def measured(self) -> bool:
        """Whether any provider actually reported a count."""

        return self.measured_calls > 0

    def estimate_cost(
        self,
        price_per_million: tuple[float, float] | None,
    ) -> float | None:
        """Cost at a given price, or None when no price is known.

        The pair is (prompt, completion) per million tokens. None is
        returned rather than 0.0 so that "free" and "unpriced" do not
        read as the same number.
        """

        if price_per_million is None or not self.measured:
            return None

        prompt_price, completion_price = price_per_million

        return (
            self.prompt_tokens * prompt_price
            + self.completion_tokens * completion_price
        ) / 1_000_000

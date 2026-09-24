from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum


class ContextPriority(IntEnum):
    """Eviction priority for assembled context.

    Lower value = more important = evicted last.
    P0 blocks are non-evictable: the assembler always includes
    them, even if the budget is exceeded.
    """

    SYSTEM = 0
    TASK = 0
    EXECUTION = 1
    RESEARCH = 2
    MEMORY = 3
    HISTORY = 4


@dataclass(slots=True, frozen=True)
class ContextBudget:
    """Token budget for one assembled LLM request."""

    maximum_tokens: int = 32_000
    reserved_system: int = 2_000
    reserved_task: int = 2_000
    reserved_output: int = 4_000

    def __post_init__(self) -> None:
        if self.maximum_tokens <= 0:
            raise ValueError(
                "maximum_tokens must be greater than 0",
            )

        for name in (
            "reserved_system",
            "reserved_task",
            "reserved_output",
        ):
            if getattr(self, name) < 0:
                raise ValueError(
                    f"{name} must be >= 0",
                )

        reserved = (
            self.reserved_system
            + self.reserved_task
            + self.reserved_output
        )

        if reserved >= self.maximum_tokens:
            raise ValueError(
                "reserved tokens must be less than maximum_tokens",
            )

    @property
    def available(self) -> int:
        """Tokens available for all input blocks combined."""

        return (
            self.maximum_tokens - self.reserved_output
        )

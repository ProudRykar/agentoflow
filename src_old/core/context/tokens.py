from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class TokenCounter(Protocol):
    """Counts (estimated) tokens in a piece of text."""

    def count(
        self,
        text: str,
    ) -> int: ...


@dataclass(slots=True, frozen=True)
class ApproximateTokenCounter:
    """Cheap length-based token estimation.

    Deliberately model-agnostic: ``len(text) // divisor``.
    A model-specific counter (e.g. Ollama tokenizer) can replace
    this class later without changing the assembler.
    """

    divisor: int = 4

    def __post_init__(self) -> None:
        if self.divisor <= 0:
            raise ValueError(
                "divisor must be greater than 0",
            )

    def count(
        self,
        text: str,
    ) -> int:
        if not text:
            return 0

        return max(
            1,
            len(text) // self.divisor,
        )

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class ContextSource(StrEnum):
    """Provenance of one context item.

    Every piece of data assembled into an LLM request must carry
    its source, so the model (and the reader) can distinguish
    authoritative instructions from untrusted data.
    """

    SYSTEM = "system"
    USER = "user"
    TASK = "task"
    PLAN = "plan"
    TOOL = "tool"
    RESEARCH = "research"
    MEMORY = "memory"
    RUNTIME = "runtime"
    HISTORY = "history"


@dataclass(slots=True, frozen=True)
class ContextItem:
    """One labeled unit of assembled context."""

    source: ContextSource
    content: str
    reference: str | None = None

    def __post_init__(self) -> None:
        if not self.content:
            raise ValueError(
                "content must not be empty",
            )

    def header(self) -> str:
        label = self.source.value.upper()

        if self.reference is not None:
            return f"[{label} {self.reference}]"

        return f"[{label}]"

    def to_text(self) -> str:
        return f"{self.header()}\n{self.content}"

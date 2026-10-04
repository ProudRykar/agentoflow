from __future__ import annotations

from dataclasses import dataclass


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

    @classmethod
    def for_model(
        cls,
        context_size: int | None,
    ) -> "ContextBudget":
        """A budget sized for a model's declared context window.

        The default reservations (2k system + 2k task + 4k output)
        add up to 8k, which is more than a small local model's whole
        window: constructing a budget straight from ``context_size``
        would raise. Reservations scale down instead, and always
        leave room for evictable blocks.
        """

        if context_size is None or context_size <= 0:
            return cls()

        output = max(256, min(4_000, context_size // 4))
        system = max(128, min(2_000, context_size // 8))
        task = max(128, min(2_000, context_size // 8))

        # Shrink the output reservation further if the fixed blocks
        # would still crowd out the conversation.
        while (
            system + task + output >= context_size
            and output > 256
        ):
            output //= 2

        return cls(
            maximum_tokens=context_size,
            reserved_system=system,
            reserved_task=task,
            reserved_output=output,
        )

    @property
    def available(self) -> int:
        """Tokens available for the whole input, output excluded."""

        return self.maximum_tokens - self.reserved_output

    @property
    def fixed_reserved(self) -> int:
        """Room guaranteed for the non-evictable blocks.

        The harness, anchor, state, execution, checkpoint and current
        instruction are always sent, so they get a reservation
        instead of quietly eating whatever the evictable blocks were
        going to use.
        """

        return self.reserved_system + self.reserved_task

    @property
    def evictable(self) -> int:
        """Tokens the evictable blocks may use in total."""

        return max(0, self.available - self.fixed_reserved)

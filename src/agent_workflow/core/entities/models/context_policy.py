from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class ContextPolicy:
    """How the conversation window is bounded.

    Only the message count lives here. Token estimation belongs to
    ``ApproximateTokenCounter``: this policy used to carry a divisor
    field that nothing ever read, so the configured value was
    silently ignored wherever the policy was constructed.
    """

    max_messages: int | None = None

    def __post_init__(self) -> None:
        if self.max_messages is not None and self.max_messages < 1:
            raise ValueError("max_messages must be greater than 0")
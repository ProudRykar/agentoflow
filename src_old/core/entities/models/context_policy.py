from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class ContextPolicy:
    max_messages: int | None = None

    # Divisor for ApproximateTokenCounter: estimated_tokens = len(text) // divisor.
    token_estimation_divisor: int = 4

    def __post_init__(self) -> None:
        if self.max_messages is not None and self.max_messages < 1:
            raise ValueError("max_messages must be greater than 0")

        if self.token_estimation_divisor <= 0:
            raise ValueError(
                "token_estimation_divisor must be greater than 0"
            )
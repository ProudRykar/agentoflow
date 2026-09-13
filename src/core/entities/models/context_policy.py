from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class ContextPolicy:
    max_messages: int | None = None

    def __post_init__(self) -> None:
        if self.max_messages is not None and self.max_messages < 1:
            raise ValueError("max_messages must be greater than 0")
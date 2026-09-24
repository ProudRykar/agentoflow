from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class GuardrailDecision:
    allowed: bool
    reason: str | None = None

    @classmethod
    def allow(cls) -> "GuardrailDecision":
        return cls(allowed=True)

    @classmethod
    def deny(cls, reason: str) -> "GuardrailDecision":
        return cls(
            allowed=False,
            reason=reason,
        )
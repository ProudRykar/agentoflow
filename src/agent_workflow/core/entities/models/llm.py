from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True, frozen=True)
class LLMToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(slots=True, frozen=True)
class TokenUsage:
    """What the provider says it actually consumed.

    The whole context budget is arithmetic on a local estimate, and an
    estimate that is silently wrong is worse than no number: the UI
    presents it as a measurement and nobody can tell. Having the
    provider's own count lets the two be compared instead of assumed
    equal.
    """

    prompt_tokens: int
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


def extract_usage(
    raw: dict[str, Any],
) -> TokenUsage | None:
    """Read token counts from a provider response, whatever it calls them.

    Ollama reports ``prompt_eval_count``; OpenAI-compatible endpoints
    report ``usage.prompt_tokens``. Both shapes reach here because the
    same response object is stored, replayed and rendered, so the field
    names cannot be assumed uniform.
    """

    if not isinstance(raw, dict):
        return None

    usage = raw.get("usage")

    candidates = [usage, raw] if isinstance(usage, dict) else [raw]

    for source in candidates:
        prompt = source.get(
            "prompt_tokens",
            source.get("prompt_eval_count"),
        )

        if not isinstance(prompt, int) or isinstance(prompt, bool):
            continue

        completion = source.get(
            "completion_tokens",
            source.get("eval_count", 0),
        )

        return TokenUsage(
            prompt_tokens=max(0, prompt),
            completion_tokens=(
                max(0, completion)
                if isinstance(completion, int)
                and not isinstance(completion, bool)
                else 0
            ),
        )

    return None


@dataclass(slots=True, frozen=True)
class LLMResponse:
    content: str | None
    tool_calls: tuple[LLMToolCall, ...]
    raw: dict[str, Any]
    thinking: str | None = None
    usage: TokenUsage | None = None
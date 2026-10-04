from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from typing import Any, Protocol


class TokenCounter(Protocol):
    """Counts (estimated) tokens in a piece of text."""

    def count(
        self,
        text: str,
    ) -> int: ...


@dataclass(slots=True, frozen=True)
class ApproximateTokenCounter:
    """Cheap length-based token estimation.

    Deliberately model-agnostic: ``len(text) // divisor``. This is the
    fallback when no real tokenizer is available: a divisor of 4
    undercounts JSON and code badly (measured ratios are nearer 2.4
    and 1.4), so it is only correct for prose.
    """

    divisor: int = 4

    def __post_init__(self) -> None:
        if self.divisor <= 0:
            raise ValueError(
                "divisor must be greater than 0",
            )

    @property
    def divisor_value(self) -> int:
        """Exposed so callers can size a character allowance."""

        return self.divisor

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


# Encodings we trust for the model families this project targets.
# Keyed by a substring of the model name.
_MODEL_ENCODINGS: tuple[tuple[str, str], ...] = (
    ("gpt-4", "o200k_base"),
    ("gpt-3.5", "cl100k_base"),
    ("gemma", "cl100k_base"),
    ("llama3", "cl100k_base"),
    ("llama", "cl100k_base"),
    ("mistral", "cl100k_base"),
    ("qwen", "cl100k_base"),
    ("phi", "cl100k_base"),
)

DEFAULT_ENCODING = "cl100k_base"


def encoding_for_model(model: str) -> str:
    """Pick an encoding from a model name."""

    lowered = (model or "").lower()

    for needle, encoding in _MODEL_ENCODINGS:
        if needle in lowered:
            return encoding

    return DEFAULT_ENCODING


@dataclass(slots=True)
class TiktokenCounter:
    """Real token counting via ``tiktoken``.

    Two deliberate limits:

    - The encoding is a *proxy*. ``cl100k_base`` is OpenAI's, so a
      Gemma model is still counted with the wrong vocabulary. It is
      far closer than a length divisor, not exact.
    - The BPE table is downloaded once (~1.7 MB) and needs a cache
      directory. If that is unavailable the counter degrades to
      :class:`ApproximateTokenCounter` rather than failing a run.

    Results are memoised per text because the assembler counts the
    same fixed blocks on every iteration.
    """

    model: str = ""
    encoding_name: str = ""
    fallback_divisor: int = 4
    cache_size: int = 512

    _encoder: Any = field(default=None, init=False, repr=False)
    _cache: dict[str, int] = field(default_factory=dict, init=False)
    _lock: threading.Lock = field(
        default_factory=threading.Lock,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        self.encoding_name = self.encoding_name or encoding_for_model(
            self.model
        )

        try:
            import tiktoken
        except ImportError:
            self._encoder = None
            return

        try:
            self._encoder = tiktoken.get_encoding(self.encoding_name)
        except Exception:
            # No cached BPE and no network: stay usable.
            self._encoder = None

    @property
    def divisor_value(self) -> int:
        return self.fallback_divisor

    @property
    def exact(self) -> bool:
        """True when a real tokenizer is in use."""

        return self._encoder is not None

    def count(
        self,
        text: str,
    ) -> int:
        if not text:
            return 0

        if self._encoder is None:
            return max(1, len(text) // self.fallback_divisor)

        cached = self._cache.get(text)

        if cached is not None:
            return cached

        try:
            count = len(self._encoder.encode(text, disallowed_special=()))
        except Exception:
            count = max(1, len(text) // self.fallback_divisor)

        with self._lock:
            if len(self._cache) >= self.cache_size:
                self._cache.clear()

            self._cache[text] = count

        return count


def counter_for_model(
    model: str,
    *,
    divisor: int = 4,
) -> TokenCounter:
    """Best available counter for ``model``.

    Real tokenizer when tiktoken is installed and its table is
    reachable, length estimate otherwise. Callers do not need to know
    which they got; ``TiktokenCounter.exact`` says.
    """

    counter = TiktokenCounter(model=model, fallback_divisor=divisor)

    if counter.exact:
        return counter

    return ApproximateTokenCounter(divisor=divisor)


def cache_dir() -> str | None:
    """Where tiktoken keeps its BPE tables."""

    return os.environ.get("TIKTOKEN_CACHE_DIR") or None
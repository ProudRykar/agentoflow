"""Real token counting instead of a length divisor.

``len(text) // 4`` undercounts code and JSON by roughly 2x, so the
budget was enforced against a fiction. These tests pin the exact
counter where tiktoken is available and the fallback where it is not.
"""

from __future__ import annotations

import pytest

from agent_workflow.core.context.tokens import (
    ApproximateTokenCounter,
    TiktokenCounter,
    counter_for_model,
    encoding_for_model,
)


# ======================================================================
# Encoding selection
# ======================================================================


@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("gpt-4o", "o200k_base"),
        ("gpt-3.5-turbo", "cl100k_base"),
        ("gemma4:12b", "cl100k_base"),
        ("llama3.1:8b", "cl100k_base"),
        ("mistral:7b-instruct", "cl100k_base"),
        ("something-unknown", "cl100k_base"),
        ("", "cl100k_base"),
    ],
)
def test_encoding_follows_the_model(
    model: str,
    expected: str,
) -> None:
    assert encoding_for_model(model) == expected


# ======================================================================
# Exact counting
# ======================================================================


def test_exact_counter_is_used_when_available() -> None:
    counter = counter_for_model("gpt-4o")

    assert isinstance(counter, TiktokenCounter)

    if not counter.exact:
        pytest.skip("tiktoken table unavailable offline")

    assert counter.exact is True


def test_divisor_undercounts_code() -> None:
    # The reason this class exists: measured ratios for code and JSON
    # are nearer 2 chars/token, so a divisor of 4 halves the count.
    code = 'def f(x): return {"a": x, "b": [1, 2]}'

    approximate = ApproximateTokenCounter(divisor=4)
    exact = counter_for_model("gpt-4o")

    if isinstance(exact, TiktokenCounter) and not exact.exact:
        pytest.skip("tiktoken table unavailable offline")

    assert exact.count(code) > approximate.count(code)


def test_counter_is_deterministic() -> None:
    counter = counter_for_model("gpt-4o")

    text = "hello world, this is a repeat count"

    assert counter.count(text) == counter.count(text)


def test_counter_memoises_without_changing_the_answer() -> None:
    counter = counter_for_model("gpt-4o")

    text = "repeat me " * 50

    first = counter.count(text)

    for _ in range(5):
        assert counter.count(text) == first


def test_counter_handles_special_tokens_safely() -> None:
    # Tool schemas and code can contain "<|endoftext|>"; counting must
    # not raise on it.
    counter = counter_for_model("gpt-4o")

    assert counter.count("<|endoftext|> broken") > 0


def test_empty_text_costs_nothing() -> None:
    counter = counter_for_model("gpt-4o")

    assert counter.count("") == 0


def test_counter_exposes_a_ratio_for_the_assembler() -> None:
    counter = counter_for_model("gpt-4o")

    assert counter.divisor_value > 0


# ======================================================================
# Offline fallback
# ======================================================================


def test_falls_back_when_the_encoder_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    counter = TiktokenCounter(model="gpt-4o")
    counter._encoder = None

    assert counter.exact is False
    assert counter.count("hello world") == max(
        1,
        len("hello world") // 4,
    )


def test_import_error_degrades_to_the_estimate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import builtins

    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "tiktoken":
            raise ImportError("no tiktoken")

        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    counter = TiktokenCounter(model="gpt-4o")

    assert counter.exact is False
    assert counter.count("some text") > 0


def test_counter_for_model_returns_the_estimate_without_tiktoken(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import builtins

    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "tiktoken":
            raise ImportError("no tiktoken")

        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    counter = counter_for_model("gpt-4o", divisor=4)

    assert isinstance(counter, ApproximateTokenCounter)
    assert counter.divisor == 4


def test_a_broken_encoder_never_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    counter = TiktokenCounter(model="gpt-4o")

    class Exploding:
        def encode(self, *_: object, **__: object) -> list[int]:
            raise RuntimeError("boom")

    if not counter.exact:
        pytest.skip("tiktoken table unavailable offline")

    counter._encoder = Exploding()
    counter._cache.clear()

    assert counter.count("text that explodes") > 0
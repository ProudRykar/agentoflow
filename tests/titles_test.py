from __future__ import annotations

import pytest

from agent_workflow.core.application.titles import (
    MAX_TITLE_LENGTH,
    SYSTEM_PROMPT,
    build_title_messages,
    derive_title,
    generate_title,
)
from agent_workflow.core.entities.models.llm import LLMResponse
from agent_workflow.core.entities.models.llm_client import LLMClient


class Echo(LLMClient):
    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls: list[list[dict]] = []

    async def chat(self, messages, tools=()) -> LLMResponse:
        self.calls.append(messages)

        return LLMResponse(
            content=self.reply,
            tool_calls=(),
            raw={},
        )


class Boom(LLMClient):
    async def chat(self, messages, tools=()) -> LLMResponse:
        raise RuntimeError("model down")


# ======================================================================
# derive_title
# ======================================================================


def test_derive_collapses_whitespace() -> None:
    assert derive_title("  fix   the\n bug ") == "Fix the bug"


def test_derive_uses_first_sentence() -> None:
    assert derive_title(
        "Refactor the parser. Then add tests."
    ) == "Refactor the parser"


def test_derive_truncates_long_input() -> None:
    title = derive_title("word " * 40)

    assert len(title) <= MAX_TITLE_LENGTH


def test_derive_caps_words() -> None:
    title = derive_title("a b c d e f g h i j k l")

    assert len(title.split()) == 8


def test_derive_strips_trailing_punctuation() -> None:
    assert derive_title("hello, world;") == "Hello, world"


def test_derive_empty_prompt() -> None:
    assert derive_title("") == "New chat"
    assert derive_title("   ") == "New chat"
    assert derive_title("...") == "New chat"


def test_derive_capitalises() -> None:
    assert derive_title("lowercase start") == "Lowercase start"


# ======================================================================
# generate_title
# ======================================================================


@pytest.mark.asyncio
async def test_generate_uses_model_reply() -> None:
    llm = Echo("Refactor MCP client")

    assert await generate_title(llm, "please refactor") == (
        "Refactor MCP client"
    )

    messages = llm.calls[0]

    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == SYSTEM_PROMPT
    assert messages[1]["content"] == "please refactor"


@pytest.mark.asyncio
async def test_generate_strips_wrapping() -> None:
    llm = Echo('  "Add MCP transport".  ')

    assert await generate_title(llm, "x") == (
        "Add MCP transport"
    )


@pytest.mark.asyncio
async def test_generate_extracts_labelled_title() -> None:
    llm = Echo("Title: Wire up the editor")

    assert await generate_title(llm, "x") == (
        "Wire up the editor"
    )


@pytest.mark.asyncio
async def test_generate_truncates() -> None:
    llm = Echo("x" * 200)

    assert len(await generate_title(llm, "x")) <= (
        MAX_TITLE_LENGTH
    )


@pytest.mark.asyncio
async def test_generate_falls_back_on_empty_reply() -> None:
    llm = Echo("")

    assert await generate_title(llm, "fix the build") == (
        "Fix the build"
    )


@pytest.mark.asyncio
async def test_generate_falls_back_on_model_error() -> None:
    assert await generate_title(Boom(), "fix the build") == (
        "Fix the build"
    )


@pytest.mark.asyncio
async def test_generate_falls_back_on_timeout() -> None:
    class Slow(LLMClient):
        async def chat(self, messages, tools=()):
            import asyncio

            await asyncio.sleep(5)

    assert await generate_title(
        Slow(), "fix the build", timeout=0.05
    ) == "Fix the build"


@pytest.mark.asyncio
async def test_generate_truncates_prompt() -> None:
    llm = Echo("Long prompt title")

    await generate_title(llm, "x" * 5000)

    assert len(llm.calls[0][1]["content"]) <= 1500


@pytest.mark.asyncio
async def test_generate_empty_prompt() -> None:
    llm = Echo("unused")

    assert await generate_title(llm, "") == "New chat"
    assert llm.calls == []


def test_build_title_messages() -> None:
    messages = build_title_messages("hello")

    assert messages[0]["content"] == SYSTEM_PROMPT
    assert messages[1]["content"] == "hello"

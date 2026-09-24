from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from core.entities.models.agent import Agent
from core.entities.models.agent_orchestrator import AgentOrchestrator
from core.entities.models.builtin.read_file import (
    ReadFileInput,
    read_file,
)
from core.entities.models.llm import LLMResponse, LLMToolCall
from core.entities.models.llm_client import LLMClient
from core.entities.models.planner import Planner
from core.entities.models.research_contract import (
    ResearchContract,
    ResearchPage,
    ResearchResult,
)
from core.entities.models.task_contract import TaskContract
from core.entities.models.tool import Tool, ToolContext, ToolPolicy
from core.entities.models.tool_executor import ToolExecutor
from core.entities.models.tool_registry import ToolRegistry


@dataclass(slots=True, frozen=True)
class FakeResearchInput:
    url: str


def _research_json() -> str:
    return ResearchResult(
        root_url="https://example.com/",
        pages=(
            ResearchPage(
                url="https://example.com/",
                depth=0,
                title="Example",
                content="body",
                links=(),
                content_bytes=4,
            ),
        ),
        discovered_urls=(),
        failed_urls=(),
        max_depth_reached=0,
        total_bytes=4,
        page_limit_reached=False,
        byte_limit_reached=False,
    ).to_json()


class ScriptedLLM(LLMClient):
    """Plays a fixed script of responses per call."""

    def __init__(
        self,
        script: tuple[LLMResponse, ...],
    ) -> None:
        self._script = script
        self.calls = 0
        self.messages: list[list[dict[str, Any]]] = []

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: tuple[Any, ...] = (),
    ) -> LLMResponse:
        self.messages.append(messages)
        self.calls += 1

        return self._script[
            min(
                self.calls - 1,
                len(self._script) - 1,
            )
        ]


def _final(text: str) -> LLMResponse:
    return LLMResponse(
        content=text,
        tool_calls=(),
        raw={},
    )


def _read_tool_call(call_id: str = "call_1") -> LLMResponse:
    return LLMResponse(
        content="",
        tool_calls=(
            LLMToolCall(
                id=call_id,
                name="read_file",
                arguments={"path": "test.txt"},
            ),
        ),
        raw={},
    )


def _research_tool_call() -> LLMResponse:
    return LLMResponse(
        content="",
        tool_calls=(
            LLMToolCall(
                id="call_1",
                name="fake_research",
                arguments={"url": "https://example.com/"},
            ),
        ),
        raw={},
    )


def _file_registry(tmp_path: Path) -> ToolRegistry:
    (tmp_path / "test.txt").write_text(
        "hello",
        encoding="utf-8",
    )

    async def handler(
        arguments: ReadFileInput,
        context: ToolContext,
    ) -> str:
        return await read_file(arguments, context)

    registry = ToolRegistry()
    registry.register(
        Tool(
            name="read_file",
            description="read",
            input_type=ReadFileInput,
            handler=handler,
            policy=ToolPolicy(
                permissions=frozenset({"filesystem.read"}),
                timeout=1.0,
                max_output_size=100_000,
            ),
        )
    )

    return registry


def _context(tmp_path: Path) -> ToolContext:
    return ToolContext(
        working_directory=tmp_path,
        environment={},
        allowed_path=(tmp_path,),
        permissions=frozenset(
            {"filesystem.read", "test.research"}
        ),
    )


@pytest.mark.asyncio
async def test_code_draft_forces_verification_pass(
    tmp_path: Path,
) -> None:
    llm = ScriptedLLM((
        _read_tool_call(),
        _final("use ```python\nBINDINGS = {}\n```"),
        _final("fixed answer without code"),
    ))

    registry = _file_registry(tmp_path)
    agent = Agent(
        llm=llm,
        registry=registry,
        executor=ToolExecutor(registry),
    )

    result = await agent.run(
        prompt="Read test.txt",
        context=_context(tmp_path),
    )

    assert llm.calls == 3
    assert result == "fixed answer without code"

    revision_request = llm.messages[2]
    instruction = revision_request[-1]

    assert instruction["role"] == "system"
    assert "VERIFICATION REQUIRED" in instruction["content"]

    # The draft is visible to the revising request.
    drafts = [
        message
        for message in revision_request
        if message.get("role") == "assistant"
        and "BINDINGS" in str(message.get("content", ""))
    ]

    assert len(drafts) == 1


@pytest.mark.asyncio
async def test_second_draft_releases_without_loop(
    tmp_path: Path,
) -> None:
    llm = ScriptedLLM((
        _read_tool_call(),
        _final("```python\ncode v1\n```"),
        _final("```python\ncode v2\n```"),
    ))

    registry = _file_registry(tmp_path)
    agent = Agent(
        llm=llm,
        registry=registry,
        executor=ToolExecutor(registry),
    )

    result = await agent.run(
        prompt="Read test.txt",
        context=_context(tmp_path),
    )

    assert llm.calls == 3
    assert result == "```python\ncode v2\n```"


@pytest.mark.asyncio
async def test_research_draft_revised_against_guide(
    tmp_path: Path,
) -> None:
    async def research_handler(
        arguments: FakeResearchInput,
        context: ToolContext,
    ) -> str:
        return _research_json()

    registry = _file_registry(tmp_path)
    registry.register(
        Tool(
            name="fake_research",
            description="fake",
            input_type=FakeResearchInput,
            handler=research_handler,
            policy=ToolPolicy(
                permissions=frozenset({"test.research"}),
                timeout=1.0,
                max_output_size=100_000,
            ),
        )
    )

    llm = ScriptedLLM((
        _research_tool_call(),
        _final("draft without sources"),
        _final("revised with https://example.com/ source"),
    ))

    orchestrator = AgentOrchestrator()
    prompt = "Research https://example.com documentation"
    orchestrator.prepare_task(
        Planner().plan(prompt),
        TaskContract(
            requires_research=True,
            research=ResearchContract(
                root_urls=("https://example.com/",),
                min_pages=1,
                min_depth=0,
                require_all_roots=True,
            ),
        ),
        prompt=prompt,
    )

    agent = Agent(
        llm=llm,
        registry=registry,
        executor=ToolExecutor(registry),
        orchestrator=orchestrator,
    )

    result = await agent.run(
        prompt=prompt,
        context=_context(tmp_path),
    )

    assert llm.calls == 3
    assert result == "revised with https://example.com/ source"

    instruction = llm.messages[2][-1]

    assert "[SYNTHESIS GUIDE]" in instruction["content"]

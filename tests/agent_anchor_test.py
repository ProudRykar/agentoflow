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
from core.entities.models.task_contract import TaskContract
from core.entities.models.tool import Tool, ToolContext, ToolPolicy
from core.entities.models.tool_executor import ToolExecutor
from core.entities.models.tool_registry import ToolRegistry


class AlternatingLLM(LLMClient):
    """Odd calls request a tool, even calls finish."""

    def __init__(self) -> None:
        self.calls = 0

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: tuple[Any, ...] = (),
    ) -> LLMResponse:
        self.calls += 1

        if self.calls % 2 == 1:
            return LLMResponse(
                content="",
                tool_calls=(
                    LLMToolCall(
                        id=f"call_{self.calls}",
                        name="read_file",
                        arguments={"path": "test.txt"},
                    ),
                ),
                raw={},
            )

        return LLMResponse(
            content="done",
            tool_calls=(),
            raw={},
        )


def _make_registry() -> ToolRegistry:
    async def handler(
        arguments: ReadFileInput,
        context: ToolContext,
    ) -> str:
        return await read_file(arguments, context)

    registry = ToolRegistry()
    registry.register(
        Tool(
            name="read_file",
            description="Read a UTF-8 text file",
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


def _make_context(tmp_path: Path) -> ToolContext:
    return ToolContext(
        working_directory=tmp_path,
        environment={},
        allowed_path=(tmp_path,),
        permissions=frozenset({"filesystem.read"}),
    )


@pytest.mark.asyncio
async def test_run_reuses_prepared_anchor(tmp_path: Path) -> None:
    (tmp_path / "test.txt").write_text("hello", encoding="utf-8")

    prompt = "Read test.txt"
    orchestrator = AgentOrchestrator()
    orchestrator.prepare_task(
        Planner().plan(prompt),
        TaskContract(),
        prompt=prompt,
    )
    prepared = orchestrator.task_anchor

    registry = _make_registry()
    agent = Agent(
        llm=AlternatingLLM(),
        registry=registry,
        executor=ToolExecutor(registry),
        orchestrator=orchestrator,
    )

    await agent.run(
        prompt=prompt,
        context=_make_context(tmp_path),
    )

    assert agent.task_anchor is prepared


@pytest.mark.asyncio
async def test_run_creates_fallback_anchor(tmp_path: Path) -> None:
    (tmp_path / "test.txt").write_text("hello", encoding="utf-8")

    registry = _make_registry()
    agent = Agent(
        llm=AlternatingLLM(),
        registry=registry,
        executor=ToolExecutor(registry),
    )

    await agent.run(
        prompt="Read test.txt",
        context=_make_context(tmp_path),
    )

    assert agent.task_anchor is not None
    assert agent.task_anchor.original_prompt == "Read test.txt"


@pytest.mark.asyncio
async def test_continue_run_creates_new_anchor(tmp_path: Path) -> None:
    (tmp_path / "test.txt").write_text("hello", encoding="utf-8")

    registry = _make_registry()
    agent = Agent(
        llm=AlternatingLLM(),
        registry=registry,
        executor=ToolExecutor(registry),
    )
    context = _make_context(tmp_path)

    await agent.run(
        prompt="Read test.txt",
        context=context,
    )
    first = agent.task_anchor

    await agent.continue_run(
        prompt="Read test.txt again",
        context=context,
    )
    second = agent.task_anchor

    assert first is not None
    assert second is not None
    assert second is not first
    assert second.original_prompt == "Read test.txt again"


@pytest.mark.asyncio
async def test_resume_without_anchor_raises(tmp_path: Path) -> None:
    registry = _make_registry()
    agent = Agent(
        llm=AlternatingLLM(),
        registry=registry,
        executor=ToolExecutor(registry),
    )

    with pytest.raises(RuntimeError):
        await agent.resume_run(
            context=_make_context(tmp_path),
        )


def test_resume_preserves_anchor() -> None:
    orchestrator = AgentOrchestrator()
    orchestrator.prepare_task(
        Planner().plan("Read test.txt"),
        TaskContract(),
        prompt="Read test.txt",
    )
    orchestrator.on_agent_started("Read test.txt")

    before = orchestrator.task_anchor

    orchestrator.on_agent_resumed()

    assert orchestrator.task_anchor is before

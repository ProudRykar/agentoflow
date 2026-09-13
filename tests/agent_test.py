from pathlib import Path
from typing import Any

import pytest

from core.entities.models.agent import Agent
from core.entities.models.builtin.read_file import (
    ReadFileInput,
    read_file,
)
from core.entities.models.llm import LLMResponse, LLMToolCall
from core.entities.models.llm_client import LLMClient
from core.entities.models.tool import Tool, ToolContext, ToolPolicy
from core.entities.models.tool_executor import ToolExecutor
from core.entities.models.tool_registry import ToolRegistry


class FakeLLM(LLMClient):
    def __init__(self) -> None:
        self.calls = 0
        self.messages: list[list[dict[str, Any]]] = []

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: tuple[Any, ...] = (),
    ) -> LLMResponse:
        self.calls += 1
        self.messages.append(messages)

        if self.calls == 1:
            return LLMResponse(
                content="",
                tool_calls=(
                    LLMToolCall(
                        id="call_1",
                        name="read_file",
                        arguments={
                            "path": "test.txt",
                        },
                    ),
                ),
                raw={},
            )

        return LLMResponse(
            content="The file contains: Hello, agent!",
            tool_calls=(),
            raw={},
        )


@pytest.mark.asyncio
async def test_agent_executes_tool_and_returns_final_answer(
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "test.txt"
    file_path.write_text(
        "Hello, agent!",
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

    llm = FakeLLM()
    executor = ToolExecutor(registry)

    agent = Agent(
        llm=llm,
        registry=registry,
        executor=executor,
    )

    context = ToolContext(
        working_directory=tmp_path,
        environment={},
        allowed_path=(tmp_path,),
        permissions=frozenset({"filesystem.read"}),
    )

    result = await agent.run(
        prompt="Read test.txt",
        context=context,
    )

    assert result == "The file contains: Hello, agent!"
    assert llm.calls == 2
    assert llm.messages[1][-1] == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": "Hello, agent!",
    }


class InfiniteToolLLM(LLMClient):
    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: tuple[Any, ...] = (),
    ) -> LLMResponse:
        return LLMResponse(
            content="",
            tool_calls=(
                LLMToolCall(
                    id="loop",
                    name="read_file",
                    arguments={"path": "test.txt"},
                ),
            ),
            raw={},
        )


@pytest.mark.asyncio
async def test_agent_max_iterations(
    tmp_path: Path,
) -> None:
    async def handler(
        arguments: ReadFileInput,
        context: ToolContext,
    ) -> str:
        return "ok"

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
                max_output_size=100,
            ),
        )
    )

    agent = Agent(
        llm=InfiniteToolLLM(),
        registry=registry,
        executor=ToolExecutor(registry),
        max_iterations=3,
    )

    context = ToolContext(
        working_directory=tmp_path,
        environment={},
        allowed_path=(tmp_path,),
        permissions=frozenset({"filesystem.read"}),
    )

    with pytest.raises(RuntimeError, match="maximum iterations"):
        await agent.run(
            prompt="Read test.txt",
            context=context,
        )
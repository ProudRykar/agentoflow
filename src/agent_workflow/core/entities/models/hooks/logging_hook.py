from typing import Any

from agent_workflow.core.entities.models.llm import LLMResponse, LLMToolCall
from agent_workflow.core.entities.models.tool import ToolContext
from agent_workflow.core.entities.models.tool_definition import ToolDefinition
from agent_workflow.core.entities.models.tool_result import ToolResult


class LoggingHook:
    async def before_llm(
        self,
        iteration: int,
        messages: list[dict[str, Any]],
        tools: tuple[ToolDefinition, ...],
    ) -> None:
        print(
            f"[hook] before_llm "
            f"iteration={iteration} "
            f"messages={len(messages)}"
        )

    async def after_llm(
        self,
        iteration: int,
        response: LLMResponse,
    ) -> None:
        print(
            f"[hook] after_llm "
            f"iteration={iteration} "
            f"tool_calls={len(response.tool_calls)}"
        )

    async def before_tool(
        self,
        iteration: int,
        tool_call: LLMToolCall,
        context: ToolContext,
    ) -> None:
        print(
            f"[hook] before_tool "
            f"tool={tool_call.name}"
        )

    async def after_tool(
        self,
        iteration: int,
        tool_call: LLMToolCall,
        result: ToolResult,
    ) -> None:
        print(
            f"[hook] after_tool "
            f"tool={tool_call.name} "
            f"success={result.error is None}"
        )
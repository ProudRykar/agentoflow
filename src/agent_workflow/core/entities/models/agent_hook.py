from typing import Any, Protocol
from typing import Any

from agent_workflow.core.entities.models.llm import LLMResponse, LLMToolCall
from agent_workflow.core.entities.models.tool import ToolContext
from agent_workflow.core.entities.models.tool_definition import ToolDefinition
from agent_workflow.core.entities.models.tool_result import ToolResult


class AgentHook(Protocol):
    async def before_llm(
        self,
        iteration: int,
        messages: list[dict[str, Any]],
        tools: tuple[ToolDefinition, ...],
    ) -> None:
        ...

    async def after_llm(
        self,
        iteration: int,
        response: LLMResponse,
    ) -> None:
        ...

    async def before_tool(
        self,
        iteration: int,
        tool_call: LLMToolCall,
        context: ToolContext,
    ) -> None:
        ...

    async def after_tool(
        self,
        iteration: int,
        tool_call: LLMToolCall,
        result: ToolResult,
    ) -> None:
        ...
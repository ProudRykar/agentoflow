from typing import Any, Protocol

from agent_workflow.core.entities.models.guardrail import GuardrailDecision
from agent_workflow.core.entities.models.guardrail_error import GuardrailDeniedError
from agent_workflow.core.entities.models.llm import LLMResponse, LLMToolCall
from agent_workflow.core.entities.models.tool import ToolContext
from agent_workflow.core.entities.models.tool_definition import ToolDefinition
from agent_workflow.core.entities.models.tool_result import ToolResult


class Guardrail(Protocol):
    async def check_tool(
        self,
        iteration: int,
        tool_call: LLMToolCall,
        context: ToolContext,
    ) -> GuardrailDecision:
        ...


class GuardrailHook:
    def __init__(
        self,
        guardrails: tuple[Guardrail, ...],
    ) -> None:
        self._guardrails = guardrails

    async def before_llm(
        self,
        iteration: int,
        messages: list[dict[str, Any]],
        tools: tuple[ToolDefinition, ...],
    ) -> None:
        pass

    async def after_llm(
        self,
        iteration: int,
        response: LLMResponse,
    ) -> None:
        pass

    async def before_tool(
        self,
        iteration: int,
        tool_call: LLMToolCall,
        context: ToolContext,
    ) -> None:
        for guardrail in self._guardrails:
            decision = await guardrail.check_tool(
                iteration=iteration,
                tool_call=tool_call,
                context=context,
            )

            if not decision.allowed:
                raise GuardrailDeniedError(
                    decision.reason
                    or "Tool call denied by guardrail"
                )

    async def after_tool(
        self,
        iteration: int,
        tool_call: LLMToolCall,
        result: ToolResult,
    ) -> None:
        pass
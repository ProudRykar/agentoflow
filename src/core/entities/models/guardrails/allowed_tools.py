from core.entities.models.guardrail import GuardrailDecision
from core.entities.models.llm import LLMToolCall
from core.entities.models.tool import ToolContext


class AllowedToolsGuardrail:
    def __init__(
        self,
        allowed_tools: frozenset[str],
    ) -> None:
        self._allowed_tools = allowed_tools

    async def check_tool(
        self,
        iteration: int,
        tool_call: LLMToolCall,
        context: ToolContext,
    ) -> GuardrailDecision:
        if tool_call.name not in self._allowed_tools:
            return GuardrailDecision.deny(
                f"Tool '{tool_call.name}' is blocked by guardrail"
            )

        return GuardrailDecision.allow()
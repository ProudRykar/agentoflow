from agent_workflow.core.entities.models.guardrail import GuardrailDecision
from agent_workflow.core.entities.models.llm import LLMToolCall
from agent_workflow.core.entities.models.tool import ToolContext
from agent_workflow.core.entities.models.tool_registry import ToolRegistry


class AllowedToolsGuardrail:
    """Deny anything that is not in the live tool registry.

    The allowlist is derived from the registry rather than
    hardcoded, so tools contributed by plugins are permitted
    once they are registered. Pass ``allowed_tools`` to
    additionally restrict the set.
    """

    def __init__(
        self,
        allowed_tools: frozenset[str] | None = None,
        registry: ToolRegistry | None = None,
    ) -> None:
        self._allowed_tools = allowed_tools
        self._registry = registry

    async def check_tool(
        self,
        iteration: int,
        tool_call: LLMToolCall,
        context: ToolContext,
    ) -> GuardrailDecision:
        if not self._is_allowed(tool_call.name):
            return GuardrailDecision.deny(
                f"Tool '{tool_call.name}' is blocked by guardrail"
            )

        return GuardrailDecision.allow()

    def _is_allowed(self, name: str) -> bool:
        if self._allowed_tools is not None:
            return name in self._allowed_tools

        if self._registry is not None:
            return self._registry.has(name)

        return True
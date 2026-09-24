from core.entities.models.guardrail import GuardrailDecision
from core.entities.models.llm import LLMToolCall
from core.entities.models.path_policy import PathPolicy, PathPolicyError
from core.entities.models.tool import ToolContext


class PathGuardrail:
    def __init__(
        self,
        path_policy: PathPolicy,
        tools: frozenset[str] = frozenset({"read_file"}),
    ) -> None:
        self._path_policy = path_policy
        self._tools = tools

    async def check_tool(
        self,
        iteration: int,
        tool_call: LLMToolCall,
        context: ToolContext,
    ) -> GuardrailDecision:
        if tool_call.name not in self._tools:
            return GuardrailDecision.allow()

        path = tool_call.arguments.get("path")

        if not isinstance(path, str):
            return GuardrailDecision.deny(
                "Tool path argument must be a string"
            )

        try:
            self._path_policy.resolve(
                context.working_directory / path
            )
        except PathPolicyError as exc:
            return GuardrailDecision.deny(str(exc))

        return GuardrailDecision.allow()
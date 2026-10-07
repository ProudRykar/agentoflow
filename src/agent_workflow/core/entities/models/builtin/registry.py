from agent_workflow.core.entities.models.builtin.tools import (
    create_builtin_tools,
)
from agent_workflow.core.entities.models.tool_registry import ToolRegistry


def create_builtin_registry(
    search: object | None = None,
) -> ToolRegistry:
    """The built-in tools, optionally with a chosen search backend.

    ``search`` is injected rather than built here so the configuration
    decides which engine is used, and so a test can supply its own.
    """

    registry = ToolRegistry()

    for tool in create_builtin_tools(search=search):
        registry.register(tool)

    return registry

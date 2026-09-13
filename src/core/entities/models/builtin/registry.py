from core.entities.models.builtin.tools import create_builtin_tools
from core.entities.models.tool_registry import ToolRegistry


def create_builtin_registry() -> ToolRegistry:
    registry = ToolRegistry()

    for tool in create_builtin_tools():
        registry.register(tool)

    return registry
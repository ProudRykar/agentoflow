from core.entities.models.tool import Tool
from typing import Any

class ToolRegistryError(Exception):
    pass

class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool[Any, Any]] = {}


    def register(self, tool: Tool[Any, Any]) -> None:
        """Регистрация инструмента"""

        if tool.name in self._tools:
            raise ToolRegistryError(f"Tool '{tool.name}' is already registerd")

        self._tools[tool.name] = tool


    def get(self, name: str) -> Tool[Any, Any]:
        """Получение инструмента."""

        try:
            return self._tools[name]
        except KeyError:
            raise ToolRegistryError(f"Tool '{name}' is not registered") from None


    def all(self) -> tuple[Tool[Any, Any], ...]:
        return tuple(self._tools.values())

    def has(self, name: str) -> bool:
        return name in self._tools
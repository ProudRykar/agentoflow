from __future__ import annotations

from typing import Any

from core.entities.models.tool import Tool


class ToolRegistryError(Exception):
    pass


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[
            str,
            Tool[Any, Any],
        ] = {}

    def register(
        self,
        tool: Tool[Any, Any],
    ) -> None:
        if tool.name in self._tools:
            raise ToolRegistryError(
                f"Tool '{tool.name}' is already registered"
            )

        self._tools[tool.name] = tool

    def get(
        self,
        name: str,
    ) -> Tool[Any, Any]:
        try:
            return self._tools[name]
        except KeyError:
            raise ToolRegistryError(
                f"Tool '{name}' is not registered"
            ) from None

    def all(
        self,
    ) -> tuple[Tool[Any, Any], ...]:
        return tuple(
            self._tools.values(),
        )

    def has(
        self,
        name: str,
    ) -> bool:
        return name in self._tools
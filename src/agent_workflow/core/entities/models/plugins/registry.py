from __future__ import annotations

from typing import Any

from agent_workflow.core.entities.models.plugins.plugin import (
    Plugin,
    PluginMetadata,
    PluginError,
    PluginNotFoundError,
    PluginAlreadyRegisteredError,
)


class PluginRegistry:
    def __init__(self) -> None:
        self._plugins: dict[str, Plugin] = {}

    def register(self, plugin: Plugin) -> None:
        if plugin.name in self._plugins:
            raise PluginAlreadyRegisteredError(
                f"Plugin '{plugin.name}' is already registered"
            )

        self._plugins[plugin.name] = plugin

    def unregister(self, name: str) -> None:
        if name not in self._plugins:
            raise PluginNotFoundError(f"Plugin '{name}' is not registered")

        del self._plugins[name]

    def get(self, name: str) -> Plugin:
        try:
            return self._plugins[name]
        except KeyError:
            raise PluginNotFoundError(f"Plugin '{name}' is not registered") from None

    def has(self, name: str) -> bool:
        return name in self._plugins

    def list(self) -> tuple[Plugin, ...]:
        return tuple(self._plugins.values())

    def list_metadata(self) -> tuple[PluginMetadata, ...]:
        return tuple(plugin.metadata for plugin in self._plugins.values())

    def find(self, query: str) -> tuple[Plugin, ...]:
        query_lower = query.lower()
        return tuple(
            plugin
            for plugin in self._plugins.values()
            if query_lower in plugin.name.lower()
            or query_lower in plugin.metadata.description.lower()
        )

    def get_by_state(self, state: str) -> tuple[Plugin, ...]:
        return tuple(
            plugin
            for plugin in self._plugins.values()
            if plugin.state == state
        )

    def clear(self) -> None:
        self._plugins.clear()
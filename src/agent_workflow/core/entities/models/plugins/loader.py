from __future__ import annotations

import importlib.metadata
import importlib.util
import sys
from pathlib import Path
from typing import Any, Awaitable, Callable

from agent_workflow.core.entities.models.event_sink import emit_to
from agent_workflow.core.entities.models.plugins.plugin import (
    Plugin,
    PluginMetadata,
    PluginLoadError,
    PluginState,
)
from agent_workflow.core.entities.models.plugins.registry import PluginRegistry


class PluginLoader:
    def __init__(
        self,
        registry: PluginRegistry,
        disabled_plugins: tuple[str, ...] = (),
    ) -> None:
        self._registry = registry
        self._disabled_plugins = set(disabled_plugins)

    @property
    def disabled_plugins(self) -> set[str]:
        return self._disabled_plugins

    def add_disabled(self, name: str) -> None:
        self._disabled_plugins.add(name)

    def remove_disabled(self, name: str) -> None:
        self._disabled_plugins.discard(name)

    def discover(self) -> list[PluginMetadata]:
        """Discover available plugins via entry points."""
        discovered: list[PluginMetadata] = []

        try:
            entry_points = importlib.metadata.entry_points(
                group="agentoflow.plugins"
            )
        except Exception:
            return discovered

        for entry_point in entry_points:
            if entry_point.name in self._disabled_plugins:
                continue

            try:
                # Load just metadata without initializing
                plugin_class = entry_point.load()
                metadata = getattr(plugin_class, "metadata", None)
                if metadata and isinstance(metadata, PluginMetadata):
                    discovered.append(metadata)
                else:
                    # Try to create instance to get metadata
                    instance = plugin_class()
                    metadata = getattr(instance, "metadata", None)
                    if metadata and isinstance(metadata, PluginMetadata):
                        discovered.append(metadata)
            except Exception:
                continue

        return discovered

    async def load(
        self,
        name: str,
        on_event: Callable[[Any], Awaitable[None]] | None = None,
    ) -> Plugin:
        """Load and initialize a plugin by name."""
        if name in self._disabled_plugins:
            raise PluginLoadError(name, "Plugin is disabled")

        # Check if already loaded
        if self._registry.has(name):
            plugin = self._registry.get(name)
            if plugin.state in (PluginState.LOADED, PluginState.INITIALIZED, PluginState.ACTIVE):
                return plugin

        # Find and load the plugin
        try:
            entry_points = importlib.metadata.entry_points(
                group="agentoflow.plugins"
            )
        except Exception as exc:
            raise PluginLoadError(name, f"Failed to get entry points: {exc}") from exc

        entry_point = None
        for ep in entry_points:
            if ep.name == name:
                entry_point = ep
                break

        if entry_point is None:
            raise PluginLoadError(name, "Plugin not found in entry points")

        try:
            plugin_class = entry_point.load()
        except Exception as exc:
            raise PluginLoadError(name, f"Failed to import plugin: {exc}") from exc

        # Create plugin metadata
        try:
            metadata = getattr(plugin_class, "metadata", None)
            if metadata is None:
                instance = plugin_class()
                metadata = getattr(instance, "metadata", None)
        except Exception as exc:
            raise PluginLoadError(name, f"Failed to get metadata: {exc}") from exc

        if not isinstance(metadata, PluginMetadata):
            raise PluginLoadError(name, "Plugin missing metadata")

        # Check if already registered (from discover)
        if self._registry.has(name):
            plugin = self._registry.get(name)
            plugin.state = PluginState.LOADED
            plugin.module = plugin_class.__module__
            return plugin

        # Create plugin instance
        plugin = Plugin(
            metadata=metadata,
            state=PluginState.LOADED,
            module=plugin_class.__module__,
        )

        self._registry.register(plugin)
        return plugin

    async def load_all(
        self,
        on_event: Callable[[Any], Awaitable[None]] | None = None,
    ) -> list[Plugin]:
        """Load all discovered plugins."""
        loaded: list[Plugin] = []

        for metadata in self.discover():
            try:
                plugin = await self.load(metadata.name, on_event)
                loaded.append(plugin)
            except PluginLoadError:
                raise
            except Exception as exc:
                # Create failed plugin entry
                failed_plugin = Plugin(
                    metadata=metadata,
                    state=PluginState.FAILED,
                    error=str(exc),
                )
                self._registry.register(failed_plugin)
                if on_event:
                    from agent_workflow.core.entities.models.plugins.events import (
                        PluginEvent,
                        PluginEventType,
                    )
                    event = PluginEvent(
                        event_type=PluginEventType.FAILED,
                        plugin_name=metadata.name,
                        plugin_version=metadata.version,
                        metadata={"error": str(exc)},
                    )
                    await emit_to(on_event, event)

        return loaded
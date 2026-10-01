from __future__ import annotations

import importlib.metadata
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

from agent_workflow.core.entities.models.plugins.context import PluginContext
from agent_workflow.core.entities.models.plugins.events import (
    PluginEvent,
    PluginEventType,
)
from agent_workflow.core.entities.models.plugins.loader import PluginLoader
from agent_workflow.core.entities.models.plugins.plugin import (
    Plugin,
    PluginState,
)
from agent_workflow.core.entities.models.plugins.registry import PluginRegistry
from agent_workflow.core.entities.models.skills.manager import SkillManager
from agent_workflow.core.entities.models.skills.registry import SkillRegistry
from agent_workflow.core.entities.models.tool import Tool
from agent_workflow.core.entities.models.tool_registry import ToolRegistry


PluginEventCallback = Callable[
    [PluginEvent],
    Awaitable[None],
]


class PluginManager:
    def __init__(
        self,
        tool_registry: ToolRegistry,
        skill_registry: SkillRegistry,
        skill_manager: SkillManager,
        disabled_plugins: tuple[str, ...] = (),
        on_event: PluginEventCallback | None = None,
    ) -> None:
        self._tool_registry = tool_registry
        self._skill_registry = skill_registry
        self._skill_manager = skill_manager
        self._on_event = on_event

        self._registry = PluginRegistry()
        self._loader = PluginLoader(self._registry, disabled_plugins)

        self._plugin_contexts: dict[str, PluginContext] = {}

    @property
    def registry(self) -> PluginRegistry:
        return self._registry

    @property
    def tool_registry(self) -> ToolRegistry:
        """Registry plugins register their tools into."""
        return self._tool_registry

    @property
    def skill_registry(self) -> SkillRegistry:
        """Registry plugins register their skills into."""
        return self._skill_registry

    @property
    def skill_manager(self) -> SkillManager:
        """Skill manager backing plugin-contributed skills."""
        return self._skill_manager

    @property
    def loader(self) -> PluginLoader:
        return self._loader

    async def discover(self) -> list[Plugin]:
        """Discover all available plugins."""
        metadata_list = self._loader.discover()

        plugins = []
        for meta in metadata_list:
            plugin = Plugin(metadata=meta, state=PluginState.DISCOVERED)
            self._registry.register(plugin)
            plugins.append(plugin)

            await self._emit(
                PluginEventType.DISCOVERED,
                meta.name,
                meta.version,
            )

        return plugins

    async def load(
        self,
        name: str,
        run_id: str = "",
        parent_run_id: str | None = None,
    ) -> Plugin:
        """Load a plugin and prepare its context."""
        plugin = await self._loader.load(name)

        if plugin.state == PluginState.FAILED:
            return plugin

        # Create plugin context
        context = self._plugin_contexts.get(name)

        if context is None:
            context = self._new_context(name)

        plugin.state = PluginState.LOADED

        plugin.state = PluginState.LOADED

        await self._emit(
            PluginEventType.LOADED,
            name,
            plugin.metadata.version,
            run_id=run_id,
            parent_run_id=parent_run_id,
        )

        return plugin

    async def initialize(
        self,
        name: str,
        run_id: str = "",
        parent_run_id: str | None = None,
    ) -> Plugin:
        """Initialize a plugin by calling its setup method."""
        if not self._registry.has(name):
            await self.load(name, run_id, parent_run_id)

        plugin = self._registry.get(name)

        if plugin.state == PluginState.FAILED:
            return plugin

        if plugin.state == PluginState.INITIALIZED:
            return plugin

        # Load the plugin class and create instance
        try:
            entry_points = importlib.metadata.entry_points(
                group="agentoflow.plugins"
            )
            entry_point = None
            for ep in entry_points:
                if ep.name == name:
                    entry_point = ep
                    break

            if entry_point is None:
                raise RuntimeError(f"Entry point not found for {name}")

            plugin_class = entry_point.load()
            instance = plugin_class()

            # Call setup with context
            context = self._plugin_contexts.get(name)
            if context is None:
                context = self._new_context(name)

            # Call setup if it exists
            setup_method = getattr(instance, "setup", None)
            if setup_method and callable(setup_method):
                setup_method(context)

            plugin.instance = instance
            plugin.state = PluginState.INITIALIZED

            await self._emit(
                PluginEventType.INITIALIZED,
                name,
                plugin.metadata.version,
                run_id=run_id,
                parent_run_id=parent_run_id,
            )

        except Exception as exc:
            plugin.state = PluginState.FAILED
            plugin.error = str(exc)

            await self._emit(
                PluginEventType.FAILED,
                name,
                plugin.metadata.version,
                run_id=run_id,
                parent_run_id=parent_run_id,
                metadata={"error": str(exc)},
            )

            raise

        return plugin

    async def activate(
        self,
        name: str,
        run_id: str = "",
        parent_run_id: str | None = None,
    ) -> Plugin:
        """Activate a plugin (initialize if needed)."""
        if not self._registry.has(name):
            await self.load(name, run_id, parent_run_id)

        plugin = self._registry.get(name)

        if plugin.state == PluginState.FAILED:
            return plugin

        if plugin.state != PluginState.INITIALIZED:
            await self.initialize(name, run_id, parent_run_id)

        if plugin.state == PluginState.ACTIVE:
            return plugin

        plugin.state = PluginState.ACTIVE

        await self._emit(
            PluginEventType.ACTIVATED,
            name,
            plugin.metadata.version,
            run_id=run_id,
            parent_run_id=parent_run_id,
        )

        return plugin

    async def load_all(self) -> list[Plugin]:
        """Load all discovered plugins."""
        return await self._loader.load_all(self._on_event)

    async def initialize_all(
        self,
        run_id: str = "",
        parent_run_id: str | None = None,
    ) -> list[Plugin]:
        """Initialize all loaded plugins."""
        initialized: list[Plugin] = []

        for plugin in self._registry.list():
            if plugin.state in (PluginState.LOADED, PluginState.DISCOVERED):
                try:
                    await self.initialize(plugin.name, run_id, parent_run_id)
                    initialized.append(plugin)
                except Exception:
                    # Error already emitted in initialize()
                    pass

        return initialized

    async def activate_all(
        self,
        run_id: str = "",
        parent_run_id: str | None = None,
    ) -> list[Plugin]:
        """Activate all initialized plugins."""
        activated: list[Plugin] = []

        for plugin in self._registry.list():
            if plugin.state == PluginState.INITIALIZED:
                await self.activate(plugin.name, run_id, parent_run_id)
                activated.append(plugin)

        return activated

    async def deactivate(
        self,
        name: str,
        run_id: str = "",
        parent_run_id: str | None = None,
    ) -> None:
        """Deactivate a plugin."""
        if not self._registry.has(name):
            return

        plugin = self._registry.get(name)

        if plugin.state != PluginState.ACTIVE:
            return

        plugin.state = PluginState.DEACTIVATED

        # Unregister what this plugin registered
        context = self._plugin_contexts.get(name)
        if context:
            for tool_name in context.registered_tools:
                try:
                    self._tool_registry.unregister(tool_name)
                except Exception:
                    pass

            for skill_name in context.registered_skills:
                try:
                    self._skill_registry.unregister(skill_name)
                except Exception:
                    pass

        await self._emit(
            PluginEventType.DEACTIVATED,
            name,
            plugin.metadata.version,
            run_id=run_id,
            parent_run_id=parent_run_id,
        )

    async def unload(
        self,
        name: str,
        run_id: str = "",
        parent_run_id: str | None = None,
    ) -> None:
        """Unload a plugin completely."""
        await self.deactivate(name, run_id, parent_run_id)

        if self._registry.has(name):
            plugin = self._registry.get(name)
            plugin.state = PluginState.UNLOADED
            self._registry.unregister(name)
            self._plugin_contexts.pop(name, None)

    def _new_context(self, name: str) -> PluginContext:
        """Build a context, letting the plugin declare permissions.

        The plugin class is already importable here, so its
        module-level metadata is available for the grant.
        """

        permissions = self._declared_permissions(name)

        context = PluginContext(
            plugin_name=name,
            tool_registry=self._tool_registry,
            skill_registry=self._skill_registry,
            skill_manager=self._skill_manager,
            on_event=self._on_event,
            permissions=permissions,
        )

        self._plugin_contexts[name] = context

        return context

    def _declared_permissions(self, name: str) -> frozenset[str]:
        """Permissions a plugin asks the host to grant."""

        try:
            entry_points = importlib.metadata.entry_points(
                group="agentoflow.plugins"
            )
        except Exception:
            return frozenset()

        for entry_point in entry_points:
            if entry_point.name != name:
                continue

            try:
                plugin_class = entry_point.load()
            except Exception:
                return frozenset()

            granted = getattr(
                plugin_class,
                "granted_permissions",
                frozenset(),
            )

            if isinstance(granted, (set, frozenset)):
                return frozenset(granted)

        return frozenset()

    def get_plugin_context(self, name: str) -> PluginContext | None:
        """Get the plugin context for a loaded plugin."""
        return self._plugin_contexts.get(name)

    def resources(self) -> dict[str, Mapping[str, Any]]:
        """Resources published by all initialized plugins.

        Keyed by plugin name. Copied into ToolContext so plugin
        tool handlers can reach their clients.
        """

        return {
            name: dict(context.resources)
            for name, context in self._plugin_contexts.items()
            if context.resources
        }

    def granted_permissions(self) -> frozenset[str]:
        """Permissions granted by all loaded plugins."""

        granted: set[str] = set()

        for context in self._plugin_contexts.values():
            granted |= context.granted_permissions

        return frozenset(granted)

    def list_active(self) -> tuple[str, ...]:
        return tuple(
            plugin.name
            for plugin in self._registry.list()
            if plugin.state == PluginState.ACTIVE
        )

    def list_failed(self) -> tuple[Plugin, ...]:
        return tuple(
            plugin
            for plugin in self._registry.list()
            if plugin.state == PluginState.FAILED
        )

    async def _emit(
        self,
        event_type: PluginEventType,
        plugin_name: str,
        plugin_version: str = "0.0.0",
        run_id: str = "",
        parent_run_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if self._on_event is None:
            return

        event = PluginEvent(
            event_type=event_type,
            plugin_name=plugin_name,
            plugin_version=plugin_version,
            run_id=run_id,
            parent_run_id=parent_run_id,
            metadata=metadata or {},
        )

        await self._on_event(event)
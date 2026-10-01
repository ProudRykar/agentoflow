from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from agent_workflow.core.entities.models.skills.manager import SkillManager
from agent_workflow.core.entities.models.skills.registry import SkillRegistry
from agent_workflow.core.entities.models.tool import Tool
from agent_workflow.core.entities.models.tool_registry import ToolRegistry


class PluginContext:
    """
    Public extension points for plugins.

    Plugins receive this context and use it to register their
    tools, skills, workflows, and commands.
    """

    def __init__(
        self,
        plugin_name: str,
        tool_registry: ToolRegistry,
        skill_registry: SkillRegistry,
        skill_manager: SkillManager,
        on_event: Callable[[Any], Awaitable[None]] | None = None,
        permissions: frozenset[str] = frozenset(),
    ) -> None:
        self._plugin_name = plugin_name
        self._tool_registry = tool_registry
        self._skill_registry = skill_registry
        self._skill_manager = skill_manager
        self._on_event = on_event

        # Permissions the host grants to this plugin's tools
        # without prompting. Anything a plugin's tool requires
        # beyond this set is routed through the approval flow.
        self._granted_permissions = permissions

        # Track what this plugin registered for observability
        self._registered_tools: set[str] = set()
        self._registered_skills: set[str] = set()

        # Resources (clients, sessions, config) shared with the
        # tools this plugin registers. Copied into ToolContext
        # by PluginManager before every run.
        self._resources: dict[str, object] = {}

        # Cached registry instances
        self._tools_registry: PluginToolRegistry | None = None
        self._skills_registry: PluginSkillRegistry | None = None
        self._workflows_registry: PluginWorkflowRegistry | None = None
        self._commands_registry: PluginCommandRegistry | None = None
        self._events_emitter: PluginEventEmitter | None = None

    @property
    def plugin_name(self) -> str:
        return self._plugin_name

    @property
    def tools(self) -> "PluginToolRegistry":
        if self._tools_registry is None:
            self._tools_registry = PluginToolRegistry(self)
        return self._tools_registry

    @property
    def skills(self) -> "PluginSkillRegistry":
        if self._skills_registry is None:
            self._skills_registry = PluginSkillRegistry(self)
        return self._skills_registry

    @property
    def workflows(self) -> "PluginWorkflowRegistry":
        if self._workflows_registry is None:
            self._workflows_registry = PluginWorkflowRegistry(self)
        return self._workflows_registry

    @property
    def commands(self) -> "PluginCommandRegistry":
        if self._commands_registry is None:
            self._commands_registry = PluginCommandRegistry(self)
        return self._commands_registry

    @property
    def events(self) -> "PluginEventEmitter":
        if self._events_emitter is None:
            self._events_emitter = PluginEventEmitter(self._plugin_name, self._on_event)
        return self._events_emitter

    @property
    def resources(self) -> dict[str, object]:
        """Mutable resource mapping for this plugin's tools."""

        return self._resources

    @property
    def granted_permissions(self) -> frozenset[str]:
        """Permissions granted to this plugin's tools."""

        return self._granted_permissions

    @property
    def granted_resources(self) -> dict[str, object]:
        return self._resources

    def publish(
        self,
        key: str,
        value: object,
    ) -> None:
        """Expose a resource to this plugin's tool handlers."""

        self._resources[key] = value

    def _track_tool(self, name: str) -> None:
        self._registered_tools.add(name)

    def _track_skill(self, name: str) -> None:
        self._registered_skills.add(name)

    @property
    def registered_tools(self) -> tuple[str, ...]:
        return tuple(sorted(self._registered_tools))

    @property
    def registered_skills(self) -> tuple[str, ...]:
        return tuple(sorted(self._registered_skills))


class PluginToolRegistry:
    def __init__(self, context: PluginContext) -> None:
        self._context = context
        self._registry = context._tool_registry

    def register(self, tool: Tool[Any, Any]) -> None:
        self._registry.register(tool)
        self._context._track_tool(tool.name)

    def unregister(self, name: str) -> None:
        self._registry.unregister(name)
        self._context._registered_tools.discard(name)

    def has(self, name: str) -> bool:
        return self._registry.has(name)

    def get(self, name: str) -> Tool[Any, Any]:
        return self._registry.get(name)


class PluginSkillRegistry:
    def __init__(self, context: PluginContext) -> None:
        self._context = context
        self._registry = context._skill_registry
        self._manager = context._skill_manager

    def register(self, skill: Any) -> None:
        from agent_workflow.core.entities.models.skills.skill import Skill

        if not isinstance(skill, Skill):
            raise TypeError("Expected Skill instance")

        self._registry.register(skill)
        self._context._track_skill(skill.name)

    def unregister(self, name: str) -> None:
        self._registry.unregister(name)
        self._context._registered_skills.discard(name)

    def has(self, name: str) -> bool:
        return self._registry.has(name)

    def get(self, name: str) -> Any:
        return self._registry.get(name)

    async def load(self, name: str) -> Any:
        return await self._manager.load(name)


class PluginWorkflowRegistry:
    def __init__(self, context: PluginContext) -> None:
        self._context = context
        self._workflows: dict[str, Any] = {}

    def register(self, name: str, workflow: Any) -> None:
        if name in self._workflows:
            raise ValueError(f"Workflow '{name}' already registered")
        self._workflows[name] = workflow

    def unregister(self, name: str) -> None:
        self._workflows.pop(name, None)

    def has(self, name: str) -> bool:
        return name in self._workflows

    def get(self, name: str) -> Any:
        return self._workflows[name]

    def list(self) -> tuple[str, ...]:
        return tuple(sorted(self._workflows.keys()))


class PluginCommandRegistry:
    def __init__(self, context: PluginContext) -> None:
        self._context = context
        self._commands: dict[str, Any] = {}

    def register(self, name: str, handler: Any) -> None:
        if name in self._commands:
            raise ValueError(f"Command '{name}' already registered")
        self._commands[name] = handler

    def unregister(self, name: str) -> None:
        self._commands.pop(name, None)

    def has(self, name: str) -> bool:
        return name in self._commands

    def get(self, name: str) -> Any:
        return self._commands[name]

    def list(self) -> tuple[str, ...]:
        return tuple(sorted(self._commands.keys()))


class PluginEventEmitter:
    def __init__(
        self,
        plugin_name: str,
        on_event: Callable[[Any], Awaitable[None]] | None,
    ) -> None:
        self._plugin_name = plugin_name
        self._on_event = on_event

    async def emit(self, event_type: str, **metadata: Any) -> None:
        if self._on_event is None:
            return

        from agent_workflow.core.entities.models.plugins.events import (
            PluginEvent,
            PluginEventType,
        )

        # Map string event type to enum
        try:
            event_enum = PluginEventType(event_type)
        except ValueError:
            event_enum = PluginEventType.LOADED  # fallback

        event = PluginEvent(
            event_type=event_enum,
            plugin_name=self._plugin_name,
            metadata=metadata,
        )

        await self._on_event(event)
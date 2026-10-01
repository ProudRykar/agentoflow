from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock
from typing import Any

import pytest

from agent_workflow.core.entities.models.plugins import (
    Plugin,
    PluginMetadata,
    PluginCapability,
    PluginState,
    PluginError,
    PluginNotFoundError,
    PluginAlreadyRegisteredError,
    PluginLoadError,
    PluginInitializationError,
    PluginContext,
    PluginToolRegistry,
    PluginSkillRegistry,
    PluginWorkflowRegistry,
    PluginCommandRegistry,
    PluginEventEmitter,
    PluginEvent,
    PluginEventType,
    PluginRegistry,
    PluginLoader,
    PluginManager,
)
from agent_workflow.core.entities.models.skills.skill import Skill
from agent_workflow.core.entities.models.tool import Tool, ToolResult, ToolError, ToolPolicy, ToolContext
from agent_workflow.core.entities.models.tool_registry import ToolRegistry


class TestPluginMetadata:
    def test_metadata_creation(self) -> None:
        metadata = PluginMetadata(
            name="test-plugin",
            version="1.0.0",
            description="Test plugin",
            author="Test Author",
            capabilities=(PluginCapability.TOOLS, PluginCapability.SKILLS),
        )
        assert metadata.name == "test-plugin"
        assert metadata.version == "1.0.0"
        assert PluginCapability.TOOLS in metadata.capabilities
        assert PluginCapability.SKILLS in metadata.capabilities

    def test_metadata_to_dict(self) -> None:
        metadata = PluginMetadata(
            name="test",
            version="1.0.0",
            capabilities=(PluginCapability.TOOLS,),
        )
        d = metadata.to_dict()
        assert d["name"] == "test"
        assert d["version"] == "1.0.0"
        assert d["capabilities"] == ["tools"]


class TestPluginModel:
    def test_plugin_creation(self) -> None:
        metadata = PluginMetadata(name="test", version="1.0.0")
        plugin = Plugin(metadata=metadata)
        assert plugin.name == "test"
        assert plugin.version == "1.0.0"
        assert plugin.state == PluginState.DISCOVERED

    def test_plugin_to_dict(self) -> None:
        metadata = PluginMetadata(name="test", version="1.0.0")
        plugin = Plugin(metadata=metadata, state=PluginState.ACTIVE)
        d = plugin.to_dict()
        assert d["metadata"]["name"] == "test"
        assert d["state"] == "active"


class TestPluginRegistry:
    def test_register_and_get(self) -> None:
        registry = PluginRegistry()
        metadata = PluginMetadata(name="test", version="1.0.0")
        plugin = Plugin(metadata=metadata)
        registry.register(plugin)

        assert registry.has("test")
        assert registry.get("test") == plugin

    def test_list(self) -> None:
        registry = PluginRegistry()
        registry.register(Plugin(metadata=PluginMetadata(name="a", version="1.0.0")))
        registry.register(Plugin(metadata=PluginMetadata(name="b", version="1.0.0")))

        plugins = registry.list()
        assert len(plugins) == 2
        assert {p.name for p in plugins} == {"a", "b"}

    def test_list_metadata(self) -> None:
        registry = PluginRegistry()
        registry.register(Plugin(metadata=PluginMetadata(name="test", version="1.5.0")))

        metadata = registry.list_metadata()
        assert len(metadata) == 1
        assert metadata[0].name == "test"
        assert metadata[0].version == "1.5.0"

    def test_unregister(self) -> None:
        registry = PluginRegistry()
        registry.register(Plugin(metadata=PluginMetadata(name="test", version="1.0.0")))

        registry.unregister("test")
        assert not registry.has("test")

    def test_duplicate_registration_raises(self) -> None:
        registry = PluginRegistry()
        plugin = Plugin(metadata=PluginMetadata(name="test", version="1.0.0"))
        registry.register(plugin)

        with pytest.raises(PluginAlreadyRegisteredError):
            registry.register(plugin)

    def test_get_not_found_raises(self) -> None:
        registry = PluginRegistry()
        with pytest.raises(PluginNotFoundError):
            registry.get("nonexistent")

    def test_find(self) -> None:
        registry = PluginRegistry()
        registry.register(Plugin(metadata=PluginMetadata(name="git-tool", version="1.0.0", description="Git tools")))
        registry.register(Plugin(metadata=PluginMetadata(name="python-test", version="1.0.0", description="Python testing")))

        results = registry.find("git")
        assert len(results) == 1
        assert results[0].name == "git-tool"

    def test_get_by_state(self) -> None:
        registry = PluginRegistry()
        registry.register(Plugin(metadata=PluginMetadata(name="a", version="1.0.0"), state=PluginState.ACTIVE))
        registry.register(Plugin(metadata=PluginMetadata(name="b", version="1.0.0"), state=PluginState.FAILED))

        active = registry.get_by_state(PluginState.ACTIVE)
        assert len(active) == 1
        assert active[0].name == "a"


class TestPluginContext:
    def setup_method(self) -> None:
        self.tool_registry = ToolRegistry()
        self.skill_registry = MagicMock()
        self.skill_manager = MagicMock()
        self.events = []

        async def on_event(event):
            self.events.append(event)

        self.context = PluginContext(
            plugin_name="test-plugin",
            tool_registry=self.tool_registry,
            skill_registry=self.skill_registry,
            skill_manager=self.skill_manager,
            on_event=on_event,
        )

    def test_tools_register(self) -> None:
        def handler(input_data: Any, context: ToolContext) -> Any:
            return ToolResult(output="done")

        tool = Tool(
            name="test.echo",
            description="Echo tool",
            input_type=str,
            handler=handler,
            policy=ToolPolicy(
                permissions=frozenset(),
                timeout=30.0,
                max_output_size=1000,
            ),
        )

        self.context.tools.register(tool)
        assert self.tool_registry.has("test.echo")
        assert "test.echo" in self.context.registered_tools

    def test_tools_unregister(self) -> None:
        def handler(input_data: Any, context: ToolContext) -> Any:
            return ToolResult(output="done")

        tool = Tool(
            name="test.echo",
            description="Echo tool",
            input_type=str,
            handler=handler,
            policy=ToolPolicy(
                permissions=frozenset(),
                timeout=30.0,
                max_output_size=1000,
            ),
        )

        self.context.tools.register(tool)
        self.context.tools.unregister("test.echo")
        assert not self.tool_registry.has("test.echo")
        assert "test.echo" not in self.context.registered_tools

    def test_skills_register(self) -> None:
        skill = Skill(name="test-skill", description="Test skill")
        self.context.skills.register(skill)
        assert "test-skill" in self.context.registered_skills

    def test_workflows_register(self) -> None:
        self.context.workflows.register("test-workflow", {"step": "echo"})
        assert self.context.workflows.has("test-workflow")
        assert "test-workflow" in self.context.workflows.list()

    def test_commands_register(self) -> None:
        def handler():
            return "done"

        self.context.commands.register("test-cmd", handler)
        assert self.context.commands.has("test-cmd")
        assert "test-cmd" in self.context.commands.list()


class TestPluginEvents:
    def test_event_creation(self) -> None:
        event = PluginEvent(
            event_type=PluginEventType.LOADED,
            plugin_name="test",
            plugin_version="1.0.0",
            run_id="run-1",
            metadata={"key": "value"},
        )
        assert event.event_type == PluginEventType.LOADED
        assert event.plugin_name == "test"
        assert event.plugin_version == "1.0.0"
        assert event.run_id == "run-1"
        assert event.metadata["key"] == "value"

    def test_event_to_dict(self) -> None:
        event = PluginEvent(
            event_type=PluginEventType.ACTIVATED,
            plugin_name="test",
        )
        d = event.to_dict()
        assert d["event_type"] == "plugin.activated"
        assert d["plugin_name"] == "test"
        assert "timestamp" in d


class TestPluginManager:
    @pytest.fixture
    def setup_manager(self):
        tool_registry = ToolRegistry()
        skill_registry = MagicMock()
        skill_manager = MagicMock()
        events = []

        async def on_event(event):
            events.append(event)

        manager = PluginManager(
            tool_registry=tool_registry,
            skill_registry=skill_registry,
            skill_manager=skill_manager,
            on_event=on_event,
        )
        return manager, events

    def test_discover_no_entry_points(self, setup_manager):
        manager, events = setup_manager
        # Should not raise, just return empty list
        result = asyncio.run(manager.discover())
        assert isinstance(result, list)

    def test_registry_property(self, setup_manager):
        manager, _ = setup_manager
        assert isinstance(manager.registry, PluginRegistry)

    def test_loader_property(self, setup_manager):
        manager, _ = setup_manager
        assert isinstance(manager.loader, PluginLoader)

    def test_list_active_empty(self, setup_manager):
        manager, _ = setup_manager
        assert manager.list_active() == ()

    def test_list_failed_empty(self, setup_manager):
        manager, _ = setup_manager
        assert manager.list_failed() == ()


class TestPluginError:
    def test_plugin_not_found_error(self) -> None:
        with pytest.raises(PluginNotFoundError):
            raise PluginNotFoundError("test")

    def test_plugin_already_registered_error(self) -> None:
        with pytest.raises(PluginAlreadyRegisteredError):
            raise PluginAlreadyRegisteredError("test")

    def test_plugin_load_error(self) -> None:
        with pytest.raises(PluginLoadError) as exc_info:
            raise PluginLoadError("test", "load failed")
        assert exc_info.value.name == "test"

    def test_plugin_initialization_error(self) -> None:
        with pytest.raises(PluginInitializationError) as exc_info:
            raise PluginInitializationError("test", "init failed")
        assert exc_info.value.name == "test"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
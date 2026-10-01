from __future__ import annotations

from dataclasses import dataclass

from agent_workflow.core.entities.models.tool import (
    Tool,
    ToolContext,
    ToolError,
    ToolPolicy,
    ToolResult,
)


PLUGINS_READ = "plugins.read"


@dataclass(slots=True, frozen=True)
class PluginsListInput:
    """No arguments. Present so the tool has a real schema."""


def create_plugins_list_tool() -> Tool[PluginsListInput, ToolResult]:
    """List connected plugins with what each one contributes."""

    async def handler(
        input_data: PluginsListInput,
        context: ToolContext,
    ) -> ToolResult:
        del input_data

        agent = getattr(context, "agent", None)

        manager = getattr(agent, "plugin_manager", None)

        if manager is None:
            return ToolResult(
                error=ToolError(
                    message="Plugin system not available",
                    code="plugins_not_available",
                    retryable=False,
                )
            )

        try:
            plugins = manager.registry.list()

            if not plugins:
                return ToolResult(
                    output="No plugins installed",
                )

            lines = ["Installed plugins:"]

            for plugin in plugins:
                state = getattr(
                    plugin.state,
                    "value",
                    plugin.state,
                )

                lines.append(
                    f"  {plugin.name} (v{plugin.version}) - {state}"
                )

                if plugin.metadata.description:
                    lines.append(
                        f"    {plugin.metadata.description}"
                    )

                plugin_context = manager.get_plugin_context(
                    plugin.name,
                )

                if plugin_context is None:
                    continue

                for tool in plugin_context.registered_tools:
                    lines.append(f"      tool: {tool}")

                for skill in plugin_context.registered_skills:
                    lines.append(f"      skill: {skill}")

            return ToolResult(output="\n".join(lines))

        except Exception as exc:
            return ToolResult(
                error=ToolError(
                    message=f"Failed to list plugins: {exc}",
                    code="list_plugins_failed",
                    retryable=True,
                )
            )

    return Tool(
        name="plugins.list",
        description=(
            "List connected plugins, the tools each one provides "
            "and the skills each one contributes"
        ),
        input_type=PluginsListInput,
        handler=handler,
        policy=ToolPolicy(
            permissions=frozenset([PLUGINS_READ]),
            timeout=30.0,
            max_output_size=10000,
        ),
    )


def create_plugins_tools() -> list[Tool]:
    """Create all plugins management tools."""
    return [
        create_plugins_list_tool(),
    ]
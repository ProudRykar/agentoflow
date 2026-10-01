from __future__ import annotations

from typing import Any

from agent_workflow.core.entities.models.plugins import (
    PluginMetadata,
    PluginCapability,
    PluginContext,
)
from agent_workflow.core.entities.models.skills.skill import Skill
from agent_workflow.core.entities.models.tool import Tool, ToolResult, ToolError, ToolPolicy, ToolContext


metadata = PluginMetadata(
    name="example",
    version="0.1.0",
    description="Example plugin demonstrating plugin capabilities",
    author="Agent Workflow Team",
    capabilities=(
        PluginCapability.TOOLS,
        PluginCapability.SKILLS,
        PluginCapability.COMMANDS,
    ),
)


class ExamplePlugin:
    metadata = metadata

    def setup(self, context: PluginContext) -> None:
        # Register a tool
        def echo_handler(input_data: str, context: ToolContext) -> ToolResult:
            return ToolResult(output=f"Echo: {input_data}")

        echo_tool = Tool(
            name="example.echo",
            description="Echo back the input",
            input_type=str,
            handler=echo_handler,
            policy=ToolPolicy(
                permissions=frozenset(),
                timeout=30.0,
                max_output_size=1000,
            ),
        )
        context.tools.register(echo_tool)

        # Register a skill
        skill = Skill(
            name="example-skill",
            description="Example skill for demonstration",
            version="1.0.0",
            instructions="""# Example Skill

This is an example skill provided by the example plugin.

## Purpose

Demonstrate how plugins can provide skills.

## Usage

Use the `example.echo` tool to echo messages.
""",
        )
        context.skills.register(skill)

        # Register a command
        def example_command():
            return "Example command executed!"

        context.commands.register("example", example_command)

        # Register a workflow
        context.workflows.register("example.workflow", {
            "name": "example-workflow",
            "steps": ["echo hello"],
        })
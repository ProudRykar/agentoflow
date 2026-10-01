from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent_workflow.core.entities.models.plugins import PluginManager

from agent_workflow.cli.approval import ApprovalController
from agent_workflow.cli.ui.app import AgentUI
from agent_workflow.core.application.runtime import create_runtime
from agent_workflow.core.application.subagents import main_model_context_size
from agent_workflow.core.infrastructure.paths import AgentWorkflowPaths


VERSION = "0.1.0"


def _inspect_plugins() -> "PluginManager":
    """Build a fully initialized PluginManager for inspection.

    Discovery alone reports DISCOVERED and shows no tools or
    skills, so the CLI loads and initializes plugins before
    reporting them.
    """

    from agent_workflow.core.entities.models.plugins import PluginManager
    from agent_workflow.core.entities.models.skills.manager import (
        SkillManager,
    )
    from agent_workflow.core.entities.models.skills.registry import (
        SkillRegistry,
    )
    from agent_workflow.core.entities.models.tool_registry import (
        ToolRegistry,
    )

    tool_registry = ToolRegistry()
    skill_registry = SkillRegistry()

    # A throwaway directory keeps the CLI focused on
    # plugin-contributed skills rather than the user's library.
    skill_manager = SkillManager(
        skills_directory=Path(tempfile.mkdtemp()),
        registry=skill_registry,
    )

    return PluginManager(
        tool_registry=tool_registry,
        skill_registry=skill_registry,
        skill_manager=skill_manager,
    )


async def _prepare_plugins() -> "PluginManager":
    manager = _inspect_plugins()

    await manager.discover()
    await manager.load_all()
    await manager.initialize_all()

    return manager


def plugins_list_command() -> None:
    """List all installed plugins and what they provide."""

    async def run() -> None:
        manager = await _prepare_plugins()

        plugins = manager.registry.list()

        if not plugins:
            print("No plugins installed")
            return

        print("Installed plugins:")
        print()

        for plugin in plugins:
            context = manager.get_plugin_context(plugin.name)

            tool_count = len(context.registered_tools) if context else 0
            skill_count = len(context.registered_skills) if context else 0

            print(
                f"  {plugin.name:<20} "
                f"{plugin.version:<10} "
                f"{plugin.state.upper()}"
            )

            if plugin.error:
                print(f"    Error: {plugin.error}")

            if plugin.metadata.description:
                print(f"    {plugin.metadata.description}")

            if plugin.metadata.capabilities:
                caps = ", ".join(plugin.metadata.capabilities)
                print(f"    Capabilities: {caps}")

            print(
                f"    Provides: {tool_count} tool(s), "
                f"{skill_count} skill(s)"
            )
            print()

    asyncio.run(run())


def plugin_show_command(name: str) -> None:
    """Show detailed information about a plugin."""

    async def run() -> None:
        manager = await _prepare_plugins()

        if not manager.registry.has(name):
            print(f"Plugin '{name}' not found")
            return

        plugin = manager.registry.get(name)
        meta = plugin.metadata

        print(f"Name:        {meta.name}")
        print(f"Version:     {meta.version}")
        print(f"Status:      {plugin.state.upper()}")

        if plugin.error:
            print(f"Error:       {plugin.error}")

        print()

        if meta.description:
            print(f"Description: {meta.description}")
            print()

        if meta.author:
            print(f"Author:      {meta.author}")

        if meta.homepage:
            print(f"Homepage:    {meta.homepage}")

        if meta.license:
            print(f"License:     {meta.license}")

        if meta.capabilities:
            print()
            print("Capabilities:")
            for cap in meta.capabilities:
                print(f"  {cap}")

        context = manager.get_plugin_context(name)

        if context is None:
            return

        if context.registered_tools:
            print()
            print("Tools:")
            for tool in context.registered_tools:
                print(f"  {tool}")

        if context.registered_skills:
            print()
            print("Skills:")
            for skill in context.registered_skills:
                print(f"  {skill}")

        if context.workflows.list():
            print()
            print("Workflows:")
            for workflow in context.workflows.list():
                print(f"  {workflow}")

        if context.commands.list():
            print()
            print("Commands:")
            for cmd in context.commands.list():
                print(f"  {cmd}")

    asyncio.run(run())


async def run_agent(
    prompt: str,
    verbose: bool = False,
) -> None:
    del verbose

    raise NotImplementedError(
        "The standalone run command is temporarily disabled. "
        "Use 'agentoflow chat'.",
    )


def run_command(
    prompt: str,
    verbose: bool = False,
) -> None:
    asyncio.run(
        run_agent(
            prompt,
            verbose=verbose,
        ),
    )


def version_command() -> None:
    print(f"agentoflow {VERSION}")


def config_path_command() -> None:
    paths = AgentWorkflowPaths()
    print(paths.config)


def test_command() -> None:
    print("Running tests...")


async def chat_agent() -> None:
    approval = ApprovalController()

    runtime = await create_runtime(
        approval_handler=approval,
    )

    ui = AgentUI(
        runtime=runtime,
        approval=approval,
        model_context_size=main_model_context_size(
            runtime.paths.resolve(
                runtime.config.models.catalog,
            ),
            runtime.config.llm.model,
        ),
    )

    try:
        await ui.run()
    finally:
        runtime.close()


def chat_command() -> None:
    asyncio.run(
        chat_agent(),
    )
from __future__ import annotations

import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

from agent_workflow.core.entities.models.agent import Agent
from agent_workflow.core.entities.models.approval import ApprovalRequest
from agent_workflow.core.entities.models.approval_hook import ApprovalHook
from agent_workflow.core.application.subagents import build_subagent_stack
from agent_workflow.core.context.context_assembler import ContextAssembler
from agent_workflow.core.context.context_controller import ContextController
from agent_workflow.core.context.history import InMemoryHistoryStore
from agent_workflow.core.context.tokens import ApproximateTokenCounter
from agent_workflow.core.entities.models.builtin.history_tools import (
    create_history_tool,
)
from agent_workflow.core.entities.models.builtin.memory_registry import create_memory_tools
from agent_workflow.core.entities.models.builtin.registry import create_builtin_registry
from agent_workflow.core.entities.models.context_manager import ContextManager
from agent_workflow.core.entities.models.context_policy import ContextPolicy
from agent_workflow.core.entities.models.guardrail_hook import GuardrailHook
from agent_workflow.core.entities.models.guardrails.allowed_tools import (
    AllowedToolsGuardrail,
)
from agent_workflow.core.entities.models.guardrails.path_guardrail import (
    PathGuardrail,
)
from agent_workflow.core.entities.models.memory_manager import MemoryManager
from agent_workflow.core.entities.models.path_policy import PathPolicy
from agent_workflow.core.entities.models.plugins.manager import PluginManager
from agent_workflow.core.entities.models.shell_policy import ShellPolicy
from agent_workflow.core.entities.models.skills.manager import SkillManager
from agent_workflow.core.entities.models.skills.registry import SkillRegistry
from agent_workflow.core.entities.models.skills.tools import create_skills_tools
from agent_workflow.core.entities.models.plugins.tools import create_plugins_tools
from agent_workflow.core.entities.models.tool import ToolContext
from agent_workflow.core.entities.models.tool_executor import ToolExecutor
from agent_workflow.core.infrastructure.config import Config, ConfigLoader
from agent_workflow.core.infrastructure.ollama_client import OllamaClient
from agent_workflow.core.infrastructure.paths import AgentWorkflowPaths
from agent_workflow.core.infrastructure.sqlite_store import SQLiteStore


ApprovalHandler = Callable[
    [ApprovalRequest],
    Awaitable[bool],
]


@dataclass(slots=True)
class AgentRuntime:
    agent: Agent
    context: ToolContext
    store: SQLiteStore
    paths: AgentWorkflowPaths
    config: Config
    plugin_manager: PluginManager

    def close(self) -> None:
        self.store.close()


async def create_runtime(
    approval_handler: ApprovalHandler,
) -> AgentRuntime:
    paths = AgentWorkflowPaths()
    paths.ensure()

    config = ConfigLoader().load(paths.config)

    store = SQLiteStore(
        paths.resolve(config.memory.database),
    )

    memory = MemoryManager(store)

    registry = create_builtin_registry()

    for tool in create_memory_tools(memory):
        registry.register(tool)

    working_directory = Path.cwd()

    path_policy = PathPolicy(
        allowed_paths=(working_directory,),
    )

    shell_policy = ShellPolicy(
        path_policy=path_policy,
    )

    # The allowlist is registry-driven so tools contributed by
    # plugins are permitted once registered. Everything already
    # lives in the registry; the guardrail denies unknown names.
    guardrail_hook = GuardrailHook(
        guardrails=(
            AllowedToolsGuardrail(registry=registry),
            PathGuardrail(
                path_policy=path_policy,
            ),
        ),
    )

    context_manager = ContextManager(
        policy=ContextPolicy(
            max_messages=config.context.max_messages,
            token_estimation_divisor=(
                config.context.token_estimation_divisor
            ),
        ),
    )

    counter = ApproximateTokenCounter(
        divisor=config.context.token_estimation_divisor,
    )

    assembler = ContextAssembler(
        counter=counter,
    )

    controller = ContextController(
        budget=assembler.budget,
        counter=counter,
        history=InMemoryHistoryStore(),
    )

    executor = ToolExecutor(
        registry,
    )

    llm = OllamaClient(
        model=config.llm.model,
        timeout=config.llm.timeout,
    )

    # Initialize Skills system
    skills_directory = paths.home / "skills"
    skills_directory.mkdir(parents=True, exist_ok=True)

    skill_registry = SkillRegistry()
    skill_manager = SkillManager(
        skills_directory=skills_directory,
        registry=skill_registry,
    )

    # Initialize Plugin system
    plugin_manager = PluginManager(
        tool_registry=registry,
        skill_registry=skill_registry,
        skill_manager=skill_manager,
    )

    agent = Agent(
        llm=llm,
        registry=registry,
        executor=executor,
        context_manager=context_manager,
        assembler=assembler,
        controller=controller,
        memory=memory,
        max_iterations=config.agent.max_iterations,
        hooks=(
            guardrail_hook,
            ApprovalHook(
                handler=approval_handler,
                shell_policy=shell_policy,
                registry=registry,
            ),
        ),
        skill_manager=skill_manager,
        plugin_manager=plugin_manager,
    )

    # Register skills management tools
    for tool in create_skills_tools():
        registry.register(tool)

    # Register plugins management tools
    for tool in create_plugins_tools():
        registry.register(tool)

    # Initialize plugins (discover, load, initialize) before the
    # context is built, so plugin tools and resources are part of
    # the first run.
    await plugin_manager.discover()
    await plugin_manager.load_all()
    await plugin_manager.initialize_all()
    await plugin_manager.activate_all()

    # Skills and plugins bring their own permissions. They are
    # granted for the session; mutating tools additionally set
    # requires_approval and are gated by ApprovalHook.
    permissions = frozenset({
        "filesystem.read",
        "filesystem.write",
        "memory.read",
        "memory.write",
        "history.read",
        "shell.execute",
        "web.fetch",
        "skills.read",
        "skills.write",
        "plugins.read",
    }) | plugin_manager.granted_permissions()

    def current_task_id() -> str | None:
        anchor = agent.orchestrator.task_anchor

        if anchor is None:
            return None

        return anchor.task_id

    registry.register(
        create_history_tool(
            current_task_id,
            controller.history,
        ),
    )

    catalog_path = paths.resolve(config.models.catalog)

    # Subagents are optional: without a model catalog the
    # harness simply runs without delegation. Invalid catalog
    # content is loud; a missing file is not.
    if catalog_path.exists():
        subagents = build_subagent_stack(
            catalog_path=catalog_path,
            main_model=config.llm.model,
            vram_budget_gb=config.models.vram_budget_gb,
            single_model_mode=config.models.single_model_mode,
            subagent_config=config.subagent,
            base_url="http://127.0.0.1:11434",
            timeout=config.llm.timeout,
            parent_registry=registry,
            memory=memory,
            assembler=assembler,
            controller=controller,
        )

        registry.register(subagents.tool)

    # Pass agent reference and plugin resources to context so
    # skills tools and plugin tools can reach them.
    context = ToolContext(
        working_directory=working_directory,
        environment=os.environ,
        allowed_path=(working_directory,),
        permissions=permissions,
        approved_permissions=frozenset(),
        plugin_resources=plugin_manager.resources(),
    )
    # Set agent reference for skills tools
    context.agent = agent

    return AgentRuntime(
        agent=agent,
        context=context,
        store=store,
        paths=paths,
        config=config,
        plugin_manager=plugin_manager,
    )
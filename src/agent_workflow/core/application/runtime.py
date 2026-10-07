from __future__ import annotations

import logging
import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

from agent_workflow.core.entities.models.agent import Agent
from agent_workflow.core.entities.models.approval import ApprovalRequest
from agent_workflow.core.entities.models.approval_hook import ApprovalHook
from agent_workflow.core.application.subagents import build_subagent_stack
from agent_workflow.core.context.compaction import (
    CompactionPolicy,
    Compactor,
)
from agent_workflow.core.context.context_assembler import ContextAssembler
from agent_workflow.core.context.context_controller import ContextController
from agent_workflow.core.context.history import InMemoryHistoryStore
from agent_workflow.core.context.tokens import counter_for_model
from agent_workflow.core.entities.models.builtin.history_tools import (
    create_history_tool,
)
from agent_workflow.core.entities.models.builtin.memory_registry import (
    create_memory_tools,
    create_todo_tools,
)
from agent_workflow.core.entities.models.tool_toggle import ToolToggle
from agent_workflow.core.infrastructure.mcp.pool import MCPProcessPool
from agent_workflow.core.entities.models.builtin.registry import (
    create_builtin_registry,
)
from agent_workflow.core.entities.models.builtin.tools import (
    build_search_backend,
    create_graphql_tool,
    create_python_tool,
    create_stash_list_tags_tool,
    create_stash_entity_schema_tool,
    create_stash_tag_overlap_tool,
)
from agent_workflow.core.entities.models.context_manager import ContextManager
from agent_workflow.core.context.context_budget import ContextBudget
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
from agent_workflow.core.infrastructure.config import (
    Config,
    ConfigLoader,
)
from agent_workflow.core.infrastructure.ollama_client import (
    OllamaClient,
    OllamaOptions,
)
from agent_workflow.core.infrastructure.paths import AgentWorkflowPaths
from agent_workflow.core.application.tool_stats_store import (
    ToolStatsStore,
)
from agent_workflow.core.infrastructure.mcp.manager import MCPManager
from agent_workflow.core.infrastructure.sqlite_store import SQLiteStore


ApprovalHandler = Callable[
    [ApprovalRequest],
    Awaitable[bool],
]

STATS_DATABASE = "tool-usage.db"

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class AgentRuntime:
    agent: Agent
    context: ToolContext
    store: SQLiteStore
    paths: AgentWorkflowPaths
    config: Config
    # Optional so a partially built runtime (tests, rehydration)
    # can still compute permissions.
    plugin_manager: PluginManager | None = None
    mcp_manager: MCPManager | None = None

    # Owns which tools are switched off. None only for a runtime
    # built by hand in a test.
    tool_toggle: ToolToggle | None = None

    # Permissions granted by the harness itself, before any
    # plugin or MCP contribution. Kept so the effective set can be
    # recomputed when MCP servers come and go at runtime.
    base_permissions: frozenset[str] = frozenset()
    stats_store: ToolStatsStore | None = None

    def refresh_permissions(self) -> frozenset[str]:
        """Recompute and return the effective permission set."""

        granted = set(self.base_permissions)

        if self.plugin_manager is not None:
            granted |= self.plugin_manager.granted_permissions()

        if self.mcp_manager is not None:
            granted |= self.mcp_manager.granted_permissions()

        return frozenset(granted)

    async def aclose(self) -> None:
        """Release async resources, then the memory store."""

        if self.mcp_manager is not None:
            await self.mcp_manager.close()

        self.store.close()

    def close(self) -> None:
        self.store.close()


async def create_runtime(
    approval_handler: ApprovalHandler,
    working_directory: Path | None = None,
    stats_store: ToolStatsStore | None = None,
    mcp_pool: MCPProcessPool | None = None,
    session_id: str = "",
) -> AgentRuntime:
    """Build one self-contained runtime.

    Every call yields an independent Agent, ToolRegistry,
    ContextManager and memory handle, which is what makes
    concurrent sessions safe. ``approval_handler`` is injected
    by the caller, so the approval surface (TUI or web) is
    chosen outside the core.
    """

    paths = AgentWorkflowPaths()
    paths.ensure()

    config = ConfigLoader().load(paths.config)

    store = SQLiteStore(
        paths.resolve(config.memory.database),
    )

    memory = MemoryManager(store)

    # The search engine comes from configuration, so a SearXNG
    # instance is picked up without touching code here.
    registry = create_builtin_registry(
        search=build_search_backend(config.web),
    )

    for tool in create_memory_tools(memory):
        registry.register(tool)

    for tool in create_todo_tools():
        registry.register(tool)

    if config.python.enabled:
        registry.register(create_python_tool(config.python))

    graphql_tool = create_graphql_tool(config.graphql)

    if graphql_tool is not None:
        registry.register(graphql_tool)

    # Shares the GraphQL client built above: one endpoint, one
    # credential, one place where the key is read.
    list_tags_tool = create_stash_list_tags_tool(config.graphql)

    if list_tags_tool is not None:
        registry.register(list_tags_tool)

    # Duplicate analysis needs usage data, not names. Shares the same
    # GraphQL client as the two tools above.
    overlap_tool = create_stash_tag_overlap_tool(config.graphql)

    if overlap_tool is not None:
        registry.register(overlap_tool)

    # Field names are the other way a query gets rejected, and guessing
    # them is what produces a tool call that looks like a broken tool.
    schema_tool = create_stash_entity_schema_tool(config.graphql)

    if schema_tool is not None:
        registry.register(schema_tool)

    working_directory = working_directory or Path.cwd()

    working_directory = working_directory.resolve()

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
        ),
    )

    counter = counter_for_model(
        config.llm.model,
        divisor=config.context.token_estimation_divisor,
    )

    # The budget has to describe the model that will actually read
    # it. A hardcoded 32k while Ollama runs with its own default is
    # how a "within budget" request still gets rejected upstream.
    declared_size = _model_context_size(
        paths.resolve(config.models.catalog or "models.toml"),
        config.llm.model,
    )

    # Ask the server what the model really has. The catalog is a
    # hand-written file that goes stale, and it is also what sets
    # num_ctx, so a wrong value there is not merely a wrong number in
    # the UI -- it either leaves capacity unused or asks for more than
    # the model will serve.
    real_size = await _reported_context_size(
        config.llm.model,
    )

    context_size = _effective_context_size(
        declared_size,
        real_size,
        model_name=config.llm.model,
    )

    budget = ContextBudget.for_model(context_size)

    assembler = ContextAssembler(
        budget=budget,
        counter=counter,
    )

    controller = ContextController(
        budget=assembler.budget,
        counter=counter,
        history=InMemoryHistoryStore(),
    )

    executor = ToolExecutor(
        registry,
        # A single result may not outgrow the window it has to fit in.
        # Sized so one tool call still has room to be worth reading:
        # a third of the budget leaves the dialogue, evidence, memory
        # and history somewhere to live.
        output_allowance=max(
            1_000,
            assembler.budget.available // 3,
        ),
    )

    llm = OllamaClient(
        model=config.llm.model,
        timeout=config.llm.timeout,
        options=OllamaOptions(
            # Keep the provider's window in step with the budget we
            # enforce locally.
            num_ctx=context_size,
        ),
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

    # Compaction uses the same client as the run: the model already
    # doing the work is the cheapest summariser there is, and it
    # writes the note in its own terms.
    compactor = (
        Compactor(
            llm=llm,
            counter=counter,
            policy=CompactionPolicy(
                target_tokens=config.context.compaction_tokens,
            ),
        )
        if config.context.compaction
        else None
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
        price_per_million=config.agent.price_per_million,
        max_prompt_tokens=config.agent.max_prompt_tokens,
        hooks=(
            guardrail_hook,
            ApprovalHook(
                handler=approval_handler,
                shell_policy=shell_policy,
                registry=registry,
                remember_approvals=config.approval.remember,
            ),
        ),
        skill_manager=skill_manager,
        plugin_manager=plugin_manager,
        compactor=compactor,
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

    # MCP servers contribute tools through the same registry, so
    # their permissions, approval gating and guardrails apply
    # unchanged.
    mcp_manager = MCPManager(
        config.mcp,
        registry,
        pool=mcp_pool,
        owner=session_id,
    )

    await mcp_manager.connect_all()

    # Applied here rather than next to the builtin registration,
    # because MCP tools only exist once their servers have connected.
    tool_toggle = ToolToggle(registry, config.tools)

    tool_toggle.apply()

    # Skills and plugins bring their own permissions. They are
    # granted for the session; mutating tools additionally set
    # requires_approval and are gated by ApprovalHook.
    base_permissions = frozenset({
        "filesystem.read",
        "filesystem.write",
        "memory.read",
        "memory.write",
        # The checklist is the model's own planning surface, not
        # something the user has to approve: granting it separately
        # would mean a run could not plan itself by default.
        "plan.read",
        "plan.write",
        "history.read",
        "shell.execute",
        "python.execute",
        "web.fetch",
        "web.search",
        "graphql.execute",
        "skills.read",
        "skills.write",
        "plugins.read",
    })

    permissions = base_permissions | (
        plugin_manager.granted_permissions()
    ) | mcp_manager.granted_permissions()

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
            # A subagent used to get ContextPolicy()'s default,
            # which means no trimming at all.
            subagent_max_messages=config.context.max_messages,
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
        mcp_manager=mcp_manager,
        tool_toggle=tool_toggle,
        base_permissions=base_permissions,
        stats_store=(
            stats_store
            if stats_store is not None
            else ToolStatsStore(paths.resolve(STATS_DATABASE))
        ),
    )

async def _reported_context_size(
    model_name: str,
) -> int | None:
    """What the Ollama server says this model's window is."""

    from agent_workflow.core.infrastructure.ollama_client import (
        OllamaClient,
    )

    try:
        return await OllamaClient(
            model=model_name
        ).model_context_length()
    except Exception:
        # A missing or broken Ollama must not stop a session from
        # starting; the catalog value still applies.
        logger.debug(
            "Could not read the real context length for %r",
            model_name,
            exc_info=True,
        )
        return None


def _effective_context_size(
    declared: int | None,
    reported: int | None,
    *,
    model_name: str = "",
) -> int | None:
    """Reconcile the catalog with the model.

    The smaller of the two wins, because that is the one that will
    actually apply: ``num_ctx`` is set from the catalog, so a catalog
    below the model's real window is a ceiling no query can pass. The
    mismatch is logged because it is a configuration problem worth
    fixing rather than absorbing silently.
    """

    if declared is None:
        return reported

    if reported is None:
        return declared

    if declared < reported:
        logger.warning(
            "models.toml declares a %d token context for %r but the "
            "model reports %d; using the smaller, since num_ctx is "
            "set from the catalog. Update the catalog to use the "
            "full window.",
            declared,
            model_name,
            reported,
        )
        return declared

    if reported < declared:
        logger.warning(
            "models.toml declares a %d token context for %r but the "
            "model reports only %d; using the model's figure.",
            declared,
            model_name,
            reported,
        )
        return reported

    return declared


def _model_context_size(
    catalog_path: Path,
    model_name: str,
) -> int | None:
    """The model's declared context window, if the catalog knows it.

    Used to size the context budget and Ollama's ``num_ctx`` so the
    local budget and the provider agree. An unknown model falls back
    to the default budget rather than guessing.
    """

    from agent_workflow.core.infrastructure.model_catalog import (
        ModelCatalogLoader,
    )

    try:
        catalog = ModelCatalogLoader().load(catalog_path)
    except Exception:
        return None

    for model in catalog.models:
        if model.name == model_name:
            size = model.requirements.context_size

            if size is not None and size > 0:
                return int(size)

    return None


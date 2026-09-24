from __future__ import annotations

import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

from core.entities.models.agent import Agent
from core.entities.models.approval import ApprovalRequest
from core.entities.models.approval_hook import ApprovalHook
from core.application.subagents import build_subagent_stack
from core.context.context_assembler import ContextAssembler
from core.context.context_controller import ContextController
from core.context.history import InMemoryHistoryStore
from core.context.tokens import ApproximateTokenCounter
from core.entities.models.builtin.history_tools import (
    create_history_tool,
)
from core.entities.models.builtin.memory_registry import create_memory_tools
from core.entities.models.builtin.registry import create_builtin_registry
from core.entities.models.context_manager import ContextManager
from core.entities.models.context_policy import ContextPolicy
from core.entities.models.guardrail_hook import GuardrailHook
from core.entities.models.guardrails.allowed_tools import (
    AllowedToolsGuardrail,
)
from core.entities.models.guardrails.path_guardrail import (
    PathGuardrail,
)
from core.entities.models.memory_manager import MemoryManager
from core.entities.models.path_policy import PathPolicy
from core.entities.models.shell_policy import ShellPolicy
from core.entities.models.tool import ToolContext
from core.entities.models.tool_executor import ToolExecutor
from core.infrastructure.config import Config, ConfigLoader
from core.infrastructure.ollama_client import OllamaClient
from core.infrastructure.paths import AgentWorkflowPaths
from core.infrastructure.sqlite_store import SQLiteStore


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

    def close(self) -> None:
        self.store.close()


def create_runtime(
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

    guardrail_hook = GuardrailHook(
        guardrails=(
            AllowedToolsGuardrail(
                allowed_tools=frozenset({
                    "read_file",
                    "list_directory",
                    "search_files",
                    "find_files",
                    "write_file",
                    "edit_file",
                    "remember",
                    "recall",
                    "execute_shell",
                    "web_fetch",
                    "web_crawl",
                    "recall_history",
                    "subagent.run",
                }),
            ),
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
            ),
        ),
    )

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

    context = ToolContext(
        working_directory=working_directory,
        environment=os.environ,
        allowed_path=(working_directory,),
        permissions=frozenset({
            "filesystem.read",
            "filesystem.write",
            "memory.read",
            "memory.write",
            "history.read",
            "shell.execute",
            "web.fetch",
        }),
        approved_permissions=frozenset(),
    )

    return AgentRuntime(
        agent=agent,
        context=context,
        store=store,
        paths=paths,
        config=config,
    )
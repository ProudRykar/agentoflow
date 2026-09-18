from __future__ import annotations

import os
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

from core.entities.models.agent import Agent
from core.entities.models.approval import ApprovalRequest
from core.entities.models.approval_hook import ApprovalHook
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
                    "web_crawl"
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
        ),
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
        max_iterations=config.agent.max_iterations,
        hooks=(
            guardrail_hook,
            ApprovalHook(
                handler=approval_handler,
                shell_policy=shell_policy,
            ),
        ),
    )

    context = ToolContext(
        working_directory=working_directory,
        environment=os.environ,
        allowed_path=(working_directory,),
        permissions=frozenset({
            "filesystem.read",
            "filesystem.write",
            "memory.read",
            "memory.write",
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
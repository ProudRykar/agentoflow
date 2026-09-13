import asyncio
import os
from collections.abc import Callable
from pathlib import Path

from core.entities.models.agent import Agent
from core.entities.models.agent_trace import (
    AgentEvent,
    AgentFinished,
    AgentStarted,
    LLMRequested,
    LLMResponded,
    ToolFinished,
    ToolStarted,
)
from core.entities.models.builtin.registry import create_builtin_registry
from core.entities.models.context_manager import ContextManager
from core.entities.models.guardrail_hook import GuardrailHook
from core.entities.models.guardrails.allowed_tools import (
    AllowedToolsGuardrail,
)
from core.entities.models.guardrails.path_guardrail import (
    PathGuardrail,
)
from core.entities.models.hooks.logging_hook import LoggingHook
from core.entities.models.path_policy import PathPolicy
from core.entities.models.tool import ToolContext
from core.entities.models.tool_executor import ToolExecutor
from core.infrastructure.config import ConfigLoader
from core.infrastructure.ollama_client import OllamaClient
from core.entities.models.context_policy import ContextPolicy
from core.entities.models.builtin.memory_registry import (
    create_memory_tools,
)
from core.infrastructure.paths import AgentWorkflowPaths
from core.infrastructure.sqlite_store import SQLiteStore
from core.entities.models.memory_manager import MemoryManager


def print_event(event: AgentEvent) -> None:
    if isinstance(event, AgentStarted):
        print("\n[agent] started")
        print(f"[agent] prompt: {event.prompt}")

    elif isinstance(event, LLMRequested):
        print(
            f"\n[llm] iteration={event.iteration} "
            f"messages={event.message_count} "
            f"tools={event.tool_count}"
        )

    elif isinstance(event, LLMResponded):
        print(
            f"[llm] iteration={event.iteration} "
            f"tool_calls={event.tool_call_count}"
        )

        if event.content:
            print(f"[llm] content: {event.content}")

    elif isinstance(event, ToolStarted):
        print(f"\n[tool] {event.tool_name}")
        print(f"[tool] arguments: {event.arguments}")

    elif isinstance(event, ToolFinished):
        if event.error_code is not None:
            print(f"[tool] error: {event.error_code}")

            if event.error_message:
                print(
                    f"[tool] message: {event.error_message}"
                )
        else:
            print(f"[tool] output: {event.output}")

    elif isinstance(event, AgentFinished):
        print("\n[agent] finished")


async def main() -> None:
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
    allowed_path = working_directory / "sandbox"

    path_policy = PathPolicy(
        allowed_paths=(allowed_path,),
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
                }),
            ),
            PathGuardrail(
                path_policy=path_policy,
            ),
        ),
    )

    context_policy = ContextPolicy(
        max_messages=config.context.max_messages,
    )

    context_manager = ContextManager(
        policy=context_policy,
    )

    executor = ToolExecutor(registry)

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
        on_event=print_event,
        hooks=(
            LoggingHook(),
            guardrail_hook,
        ),
    )

    context = ToolContext(
        working_directory=working_directory,
        environment=os.environ,
        allowed_path=(allowed_path,),
        permissions=frozenset({
            "filesystem.read",
            "filesystem.write",
            "memory.read",
            "memory.write",
            "edit_file",
            "shell.execute",
            "shell.network",
        }),
    )

    try:
        result = await agent.run(
            prompt=(
                "..."
            ),
            context=context,
        )

        print("\n=== AGENT RESULT ===")
        print(result)
    finally:
        store.close()


if __name__ == "__main__":
    asyncio.run(main())
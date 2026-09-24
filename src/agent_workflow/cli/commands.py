from __future__ import annotations

import asyncio

from agent_workflow.cli.approval import ApprovalController
from agent_workflow.cli.ui.app import AgentUI
from agent_workflow.core.application.runtime import create_runtime
from agent_workflow.core.application.subagents import main_model_context_size
from agent_workflow.core.infrastructure.paths import AgentWorkflowPaths


VERSION = "0.1.0"


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

    runtime = create_runtime(
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
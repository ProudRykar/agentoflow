import asyncio
from pathlib import Path

from rich.panel import Panel
from core.infrastructure.paths import AgentWorkflowPaths
from core.application.runtime import create_runtime
from cli.renderer import Renderer
from rich.console import Console
from rich.markup import escape
from cli.input import AgentInput

VERSION = "0.1.0"


async def run_agent(prompt: str, verbose: bool = False) -> None:
    runtime = create_runtime()
    renderer = Renderer(verbose=verbose)

    try:
        result = await runtime.agent.run(
            prompt=prompt,
            context=runtime.context,
            on_event=renderer.render,
        )
        print(result)
    finally:
        runtime.close()


def run_command(prompt: str, verbose: bool = False) -> None:
    asyncio.run(run_agent(prompt, verbose=verbose))


def version_command() -> None:
    print(f"agentoflow {VERSION}")


def config_path_command() -> None:
    paths = AgentWorkflowPaths()
    print(paths.config)


def config_show_command() -> None:
    paths = AgentWorkflowPaths()
    paths.ensure()

    print(paths.config.read_text(encoding="utf-8"))

async def chat_agent() -> None:
    runtime = create_runtime()
    console = Console()
    renderer = Renderer(console=console)
    input_session = AgentInput(runtime.paths.history)
    cwd = Path.cwd()
    try:
        console.print(
            Panel(
                f"[bold]agentoflow[/bold]\n"
                f"[dim]{runtime.config.llm.model} · {cwd}[/dim]",
                border_style="dim",
                expand=False,
            )
        )

        console.print(
            "[dim]Type /help for commands, /quit to exit.[/dim]"
        )
        console.print()

        first_message = True

        while True:
            try:
                prompt = await input_session.prompt()
            except EOFError:
                console.print()
                break
            except KeyboardInterrupt:
                console.print()
                continue

            prompt = prompt.strip()

            if not prompt:
                continue

            if prompt == "/quit":
                break

            if prompt == "/help":
                console.print(
                    Panel(
                        "\n".join(
                            [
                                "[bold]/help[/bold]    Show available commands",
                                "[bold]/clear[/bold]   Clear conversation",
                                "[bold]/quit[/bold]    Exit agentoflow",
                            ]
                        ),
                        title="Commands",
                        border_style="dim",
                        expand=False,
                    )
                )
                console.print()
                continue

            if prompt == "/clear":
                runtime.agent.clear_context()
                first_message = True

                console.print("[dim]Conversation cleared.[/dim]")
                console.print()
                continue

            if first_message:
                result = await runtime.agent.run(
                    prompt=prompt,
                    context=runtime.context,
                    on_event=renderer.render,
                )
                first_message = False
            else:
                result = await runtime.agent.continue_run(
                    prompt=prompt,
                    context=runtime.context,
                    on_event=renderer.render,
                )

            renderer.response(result)
            console.print()

    finally:
        runtime.close()

def chat_command() -> None:
    asyncio.run(chat_agent())
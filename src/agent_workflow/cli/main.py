import argparse

from agent_workflow.cli.commands import (
    chat_command,
    config_path_command,
    plugin_show_command,
    plugins_list_command,
    run_command,
    version_command,
)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agentoflow",
        description="LLM agent workflow runtime",
    )

    parser.add_argument(
        "--version",
        action="version",
        version="agentoflow 0.1.0",
    )

    subparsers = parser.add_subparsers(
        dest="command",
    )

    run_parser = subparsers.add_parser(
        "run",
        help="Run an agent with a prompt",
    )

    run_parser.add_argument(
        "prompt",
        help="Prompt to send to the agent",
    )

    run_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show agent execution details",
    )

    subparsers.add_parser(
        "version",
        help="Show version",
    )

    config_parser = subparsers.add_parser(
        "config",
        help="Manage configuration",
    )

    config_subparsers = config_parser.add_subparsers(
        dest="config_command",
    )

    config_subparsers.add_parser(
        "path",
        help="Show configuration path",
    )

    config_subparsers.add_parser(
        "show",
        help="Show configuration",
    )

    subparsers.add_parser(
        "chat",
        help="Start an interactive chat",
    )

    plugins_parser = subparsers.add_parser(
        "plugins",
        help="Manage plugins",
    )

    plugins_subparsers = plugins_parser.add_subparsers(
        dest="plugins_command",
    )

    plugins_subparsers.add_parser(
        "list",
        help="List installed plugins",
    )

    plugin_show_parser = plugins_subparsers.add_parser(
        "show",
        help="Show plugin details",
    )

    plugin_show_parser.add_argument(
        "name",
        help="Plugin name",
    )

    return parser


def main() -> None:
    parser = create_parser()
    args = parser.parse_args()

    if args.command == "version":
        version_command()
        return

    if args.command == "config":
        if args.config_command == "path":
            config_path_command()
            return

        if args.config_command == "show":
            config_path_command()
            return

    if args.command == "run":
        run_command(
            args.prompt,
            verbose=args.verbose,
        )
        return

    if args.command == "chat":
        chat_command()
        return

    if args.command == "plugins":
        if args.plugins_command == "list":
            plugins_list_command()
            return

        if args.plugins_command == "show":
            plugin_show_command(args.name)
            return

    parser.print_help()


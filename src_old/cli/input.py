from __future__ import annotations

from pathlib import Path

from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.document import Document
from prompt_toolkit.history import FileHistory


COMMANDS = {
    "/help": "Show available commands",
    "/clear": "Clear conversation",
    "/quit": "Exit agentoflow",
}


class CommandCompleter(Completer):
    def get_completions(
        self,
        document: Document,
        complete_event,
    ):
        text = document.text_before_cursor

        if "\n" in text:
            return

        if not text.startswith("/"):
            return

        for command, description in COMMANDS.items():
            if command.startswith(text):
                yield Completion(
                    text=command,
                    start_position=-len(text),
                    display=command,
                    display_meta=description,
                )


def create_history(
    path: Path,
) -> FileHistory:
    return FileHistory(
        str(path),
    )
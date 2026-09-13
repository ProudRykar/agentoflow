
from __future__ import annotations

from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.document import Document
from prompt_toolkit.history import FileHistory
from prompt_toolkit.input.ansi_escape_sequences import ANSI_SEQUENCES
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys


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

        if "\n" in text or not text.startswith("/"):
            return

        for command, description in COMMANDS.items():
            if command.startswith(text):
                yield Completion(
                    command,
                    start_position=-len(text),
                    display=command,
                    display_meta=description,
                )


class AgentInput:
    def __init__(self, history_path: Path) -> None:
        ANSI_SEQUENCES["\033[27;2;13~"] = Keys.ControlF13

        bindings = KeyBindings()

        @bindings.add("enter")
        def _(event) -> None:
            event.current_buffer.validate_and_handle()

        @bindings.add("c-f13")
        def _(event) -> None:
            event.current_buffer.insert_text("\n")

        self._session = PromptSession(
            history=FileHistory(str(history_path)),
            completer=CommandCompleter(),
            complete_while_typing=True,
            multiline=True,
            key_bindings=bindings,
        )

    async def prompt(self) -> str:
        return await self._session.prompt_async(
            [("bold", "> ")],
        )
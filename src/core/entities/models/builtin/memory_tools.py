from dataclasses import dataclass

from core.entities.models.memory_manager import MemoryManager
from core.entities.models.tool import ToolContext


@dataclass(slots=True, frozen=True)
class RememberInput:
    key: str
    value: str


@dataclass(slots=True, frozen=True)
class RecallInput:
    key: str


def create_memory_handlers(
    memory: MemoryManager,
):
    async def remember(
        arguments: RememberInput,
        context: ToolContext,
    ) -> str:
        entry = await memory.remember(
            key=arguments.key,
            value=arguments.value,
        )

        return (
            f"Memory saved: '{entry.key}' = '{entry.value}'"
        )

    async def recall(
        arguments: RecallInput,
        context: ToolContext,
    ) -> str:
        entry = await memory.get(arguments.key)

        if entry is None:
            return f"No memory found for key '{arguments.key}'"

        return (
            f"Memory found: '{entry.key}' = '{entry.value}'"
        )

    return remember, recall
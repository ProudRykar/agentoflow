from core.entities.models.builtin.memory_tools import (
    RecallInput,
    RememberInput,
    create_memory_handlers,
)
from core.entities.models.memory_manager import MemoryManager
from core.entities.models.tool import Tool, ToolPolicy


def create_memory_tools(
    memory: MemoryManager,
) -> tuple[Tool, ...]:
    remember, recall = create_memory_handlers(memory)

    return (
        Tool(
            name="remember",
            description=(
                "Save a piece of information in long-term memory "
                "using a key and value"
            ),
            input_type=RememberInput,
            handler=remember,
            policy=ToolPolicy(
                permissions=frozenset({"memory.write"}),
                timeout=5.0,
                max_output_size=1_000,
            ),
        ),
        Tool(
            name="recall",
            description=(
                "Retrieve a piece of information from long-term "
                "memory using its key"
            ),
            input_type=RecallInput,
            handler=recall,
            policy=ToolPolicy(
                permissions=frozenset({"memory.read"}),
                timeout=5.0,
                max_output_size=10_000,
            ),
        ),
    )
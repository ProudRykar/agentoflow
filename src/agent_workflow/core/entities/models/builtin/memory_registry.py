from agent_workflow.core.entities.models.builtin.memory_tools import (
    RecallInput,
    RecallMatchingInput,
    RememberInput,
    create_memory_handlers,
)
from agent_workflow.core.entities.models.memory_manager import MemoryManager
from agent_workflow.core.entities.models.tool import Tool, ToolPolicy

REMEMBER_DESCRIPTION = (
    "Save something you worked out this session, so a later step or a "
    "later session can use it without rediscovering it.\n\n"
    "Worth saving: a fact you had to try several times to learn, a "
    "convention of a codebase, a rule of a system you are operating. "
    "Not worth saving: the current value of a variable, the output of a "
    "tool you can call again, or anything the conversation already "
    "says.\n\n"
    "Key it by what it is, not by when you learned it. 'game.rules' "
    "beats 'step3'. scope defaults to 'task'; pass 'global' only for "
    "something true beyond this request."
)

RECALL_DESCRIPTION = (
    "Read one note by its exact key. Use recall_matching instead when "
    "you know what you want to know but not what you called it."
)

RECALL_MATCHING_DESCRIPTION = (
    "Search memory by topic and get back everything that matches.\n\n"
    "Use this when you need to know what you have already worked out -- "
    "'what do I know about how this works', 'did we already decide X' "
    "-- rather than when you can name the key. Notes are matched on "
    "words they share with your topic."
)


def create_memory_tools(
    memory: MemoryManager,
) -> tuple[Tool, ...]:
    (
        remember,
        recall,
        recall_matching,
    ) = create_memory_handlers(memory)

    return (
        Tool(
            name="remember",
            description=REMEMBER_DESCRIPTION,
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
            description=RECALL_DESCRIPTION,
            input_type=RecallInput,
            handler=recall,
            policy=ToolPolicy(
                permissions=frozenset({"memory.read"}),
                timeout=5.0,
                max_output_size=10_000,
            ),
        ),
        Tool(
            name="recall_matching",
            description=RECALL_MATCHING_DESCRIPTION,
            input_type=RecallMatchingInput,
            handler=recall_matching,
            policy=ToolPolicy(
                permissions=frozenset({"memory.read"}),
                timeout=5.0,
                max_output_size=10_000,
            ),
        ),
    )


# ======================================================================
# Task checklist
# ======================================================================


def create_todo_tools() -> tuple[Tool, ...]:
    """Tools for the model's own checklist.

    Registered unconditionally rather than behind a flag: a plan the
    agent cannot revise is one it cannot keep true, and the earlier
    version of this prompt rendered a checklist the model was not
    allowed to edit.
    """

    from agent_workflow.core.entities.models.builtin.todo_tools import (
        TODOREAD_DESCRIPTION,
        TODOWRITE_DESCRIPTION,
        TodoReadInput,
        TodoWriteInput,
        todoread,
        todowrite,
    )

    return (
        Tool(
            name="todowrite",
            description=TODOWRITE_DESCRIPTION,
            input_type=TodoWriteInput,
            handler=todowrite,
            policy=ToolPolicy(
                permissions=frozenset({"plan.write"}),
                timeout=5.0,
                max_output_size=8_000,
            ),
        ),
        Tool(
            name="todoread",
            description=TODOREAD_DESCRIPTION,
            input_type=TodoReadInput,
            handler=todoread,
            policy=ToolPolicy(
                permissions=frozenset({"plan.read"}),
                timeout=5.0,
                max_output_size=8_000,
            ),
        ),
    )

"""The batch tool must exist, be safe, and be honest about failures.

Three properties earn their own tests. It must not widen permissions:
a batch that needs less approval than the single call would be the
cheap way around the approval flow, which exists for writes. It must
reject an oversized or malformed batch *before* any call goes out, so
a rejected batch leaves nothing half-written. And it must report
per-item outcomes, because a batch of fifty with three failures is a
normal outcome and the other forty-seven are already done.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from agent_workflow.core.entities.models.tool_registry import (
    ToolRegistry,
)
from agent_workflow.core.infrastructure.config import (
    MCPConfig,
    MCPServerConfigEntry,
)
from agent_workflow.core.infrastructure.mcp.manager import (
    BatchCall,
    BatchCallInput,
    MAX_BATCH_ITEMS,
    MCPManager,
)


STUB = str(Path(__file__).parent / "fixtures" / "mcp_stub_server.py")


async def connected_manager(**config_kwargs) -> tuple[MCPManager, ToolRegistry]:
    registry = ToolRegistry()

    manager = MCPManager(
        registry=registry,
        config=MCPConfig(
            enabled=True,
            servers=(
                MCPServerConfigEntry(
                    name="stub",
                    command=sys.executable,
                    args=(STUB,),
                ),
            ),
            batch_tools=("*",),
            **config_kwargs,
        ),
    )

    await manager.connect_all()

    return manager, registry


async def test_every_tool_is_offered_again_as_a_batch() -> None:
    """Otherwise the model has to loop, one turn and one approval
    per item, which is the thing this exists to remove."""

    manager, registry = await connected_manager()

    try:
        singles = [
            entry.name
            for entry in registry.all()
            if not entry.name.endswith("_batch")
        ]

        assert singles, "the stub server exposes no tools"

        for name in singles:
            assert registry.has(f"{name}_batch"), name
    finally:
        await manager.close()


async def test_a_batch_runs_every_item() -> None:
    manager, registry = await connected_manager()

    try:
        tool = registry.get("mcp_stub_echo_batch")

        result = await tool.handler(
            BatchCallInput(
                calls=(
                    BatchCall(arguments={"tag_id": "a"}),
                    BatchCall(arguments={"tag_id": "b"}),
                    BatchCall(arguments={"tag_id": "c"}),
                )
            ),
            None,
        )

        assert "3 of 3 succeeded" in result
        assert "echo:a" in result
        assert "echo:b" in result
        assert "echo:c" in result
    finally:
        await manager.close()


async def test_an_empty_batch_is_refused_without_calling_anything() -> None:
    manager, registry = await connected_manager()

    try:
        result = await registry.get(
            "mcp_stub_echo_batch"
        ).handler(BatchCallInput(calls=()), None)

        assert "empty" in result.lower()
    finally:
        await manager.close()


async def test_an_oversized_batch_is_rejected_before_any_call() -> None:
    """Rejected up front, so nothing is half-written."""

    manager, registry = await connected_manager()

    try:
        result = await registry.get("mcp_stub_echo_batch").handler(
            BatchCallInput(
                calls=tuple(
                    BatchCall(arguments={"tag_id": str(i)})
                    for i in range(MAX_BATCH_ITEMS + 1)
                )
            ),
            None,
        )

        assert "exceeds the limit" in result
        # Nothing ran, so no item may claim to have succeeded.
        assert "succeeded" not in result
    finally:
        await manager.close()


async def test_a_malformed_item_is_rejected_before_any_call() -> None:
    manager, registry = await connected_manager()

    try:
        result = await registry.get("mcp_stub_echo_batch").handler(
            BatchCallInput(
                calls=(
                    BatchCall(arguments={"tag_id": "fine"}),
                    {},
                )
            ),
            None,
        )

        assert "no arguments" in result
        assert "succeeded" not in result
    finally:
        await manager.close()


async def test_a_failing_item_does_not_discard_the_others() -> None:
    """The successful ones already happened remotely.

    Raising here would report the whole batch as lost and invite the
    model to retry everything, duplicating the writes that landed.
    """

    manager, registry = await connected_manager()

    try:
        result = await registry.get("mcp_stub_echo_batch").handler(
            BatchCallInput(
                calls=(
                    BatchCall(arguments={"tag_id": "good1"}),
                    BatchCall(arguments={"wrong": "field"}),
                    BatchCall(arguments={"tag_id": "good2"}),
                )
            ),
            None,
        )

        assert "Failed items:" in result
        assert "echo:good1" in result
        assert "echo:good2" in result
    finally:
        await manager.close()


async def test_a_batch_never_asks_for_less_than_the_single_call() -> None:
    """Otherwise batching becomes the way around the approval flow."""

    manager, registry = await connected_manager(require_approval=True)

    try:
        single = registry.get("mcp_stub_echo")
        batch = registry.get("mcp_stub_echo_batch")

        assert batch.policy.requires_approval is True
        assert single.policy.requires_approval is True
        assert batch.policy.permissions == single.policy.permissions
    finally:
        await manager.close()


async def test_the_limit_is_taken_from_configuration() -> None:
    manager, registry = await connected_manager(batch_concurrency=2)

    try:
        tool = registry.get("mcp_stub_echo_batch")

        assert "up to 2 in flight" in tool.description
    finally:
        await manager.close()


async def test_a_zero_concurrency_is_refused_by_configuration() -> None:
    with pytest.raises(ValueError, match="batch_concurrency"):
        MCPConfig(enabled=True, batch_concurrency=0)


async def test_an_error_delivered_as_plain_text_is_still_a_failure() -> None:
    """Servers that skip ``isError`` must not be counted as success.

    The Stash MCP server returns "Error calling tool 'x': ..." as
    ordinary content. Counting that as done produces a report that
    says every item worked while none did, which is the worst answer
    this tool can give.
    """

    from agent_workflow.core.infrastructure.mcp.manager import (
        _describe_item,
        _looks_like_failure,
    )

    assert _looks_like_failure(
        "Error calling tool 'update_tag_description': no connection"
    )
    assert _looks_like_failure("Error: tag not found")
    assert not _looks_like_failure("updated 3 tags")

    status, text = _describe_item(
        0,
        {
            "content": [
                {
                    "type": "text",
                    "text": "Error calling tool 'update': nope",
                }
            ]
        },
    )

    assert status == "failed"
    assert "nope" in text


async def test_the_summary_counts_failures_that_came_back_as_text() -> None:
    """The headline number must not say "all succeeded" over errors."""

    from agent_workflow.core.infrastructure.mcp.manager import (
        _describe_item,
    )

    outcomes = [
        {"content": [{"type": "text", "text": "done"}]},
        {"content": [{"type": "text", "text": "Error calling tool 'x'"}]},
        {"content": [{"type": "text", "text": "done"}]},
    ]

    statuses = [_describe_item(i, o)[0] for i, o in enumerate(outcomes)]

    assert statuses == ["ok", "failed", "ok"]


async def test_batching_is_off_unless_asked_for() -> None:
    """The default must not change the tool list.

    Offering a batch twin of every tool doubles what the model sees,
    and this project deliberately keeps that list short: a long list
    makes a model choose worse. Enabling it is a decision, not a
    side effect of an upgrade.
    """

    registry = ToolRegistry()

    manager = MCPManager(
        registry=registry,
        config=MCPConfig(
            enabled=True,
            servers=(
                MCPServerConfigEntry(
                    name="stub",
                    command=sys.executable,
                    args=(STUB,),
                ),
            ),
        ),
    )

    await manager.connect_all()

    try:
        names = [entry.name for entry in registry.all()]

        assert names, "the stub server exposes no tools"
        assert not any(name.endswith("_batch") for name in names)
    finally:
        await manager.close()


async def test_batching_can_be_asked_for_one_tool_only() -> None:
    """A name is enough; "*" is not required."""

    registry = ToolRegistry()

    manager = MCPManager(
        registry=registry,
        config=MCPConfig(
            enabled=True,
            servers=(
                MCPServerConfigEntry(
                    name="stub",
                    command=sys.executable,
                    args=(STUB,),
                ),
            ),
            batch_tools=("echo",),
        ),
    )

    await manager.connect_all()

    try:
        assert registry.has("mcp_stub_echo_batch")
    finally:
        await manager.close()


async def test_a_blank_tool_name_is_refused() -> None:
    with pytest.raises(ValueError, match="batch_tools"):
        MCPConfig(enabled=True, batch_tools=("echo", "  "))


def test_a_config_without_the_key_loads(tmp_path) -> None:
    """Absent means off, not invalid.

    An earlier default of ``()`` fed a tuple to a helper that accepts
    only a list, so every configuration that did not mention the key
    failed to load -- seventy unrelated tests, none of them about MCP.
    """

    from agent_workflow.core.infrastructure.config import ConfigLoader

    path = tmp_path / "config.toml"
    path.write_text(
        "[mcp]\n"
        "enabled = true\n"
        "\n"
        "[mcp.servers.stub]\n"
        "command = \"true\"\n"
    )

    config = ConfigLoader().load(path)

    assert config.mcp.batch_tools == ()
    assert config.mcp.batch_concurrency == 4


def test_the_key_is_read_when_present(tmp_path) -> None:
    from agent_workflow.core.infrastructure.config import ConfigLoader

    path = tmp_path / "config.toml"
    path.write_text(
        "[mcp]\n"
        "enabled = true\n"
        'batch_tools = ["update_tag_description"]\n'
        "batch_concurrency = 8\n"
        "\n"
        "[mcp.servers.stub]\n"
        "command = \"true\"\n"
    )

    config = ConfigLoader().load(path)

    assert config.mcp.batch_tools == ("update_tag_description",)
    assert config.mcp.batch_concurrency == 8
    assert config.mcp.batches("update_tag_description") is True
    assert config.mcp.batches("some_other_tool") is False


def test_a_wildcard_enables_everything(tmp_path) -> None:
    from agent_workflow.core.infrastructure.config import ConfigLoader

    path = tmp_path / "config.toml"
    path.write_text(
        "[mcp]\n"
        "enabled = true\n"
        'batch_tools = ["*"]\n'
        "\n"
        "[mcp.servers.stub]\n"
        "command = \"true\"\n"
    )

    config = ConfigLoader().load(path)

    assert config.mcp.batches("anything_at_all") is True


def test_a_blank_tool_name_in_the_file_is_refused(tmp_path) -> None:
    from agent_workflow.core.infrastructure.config import (
        ConfigError,
        ConfigLoader,
    )

    path = tmp_path / "config.toml"
    path.write_text(
        "[mcp]\n"
        "enabled = true\n"
        'batch_tools = ["ok", "   "]\n'
        "\n"
        "[mcp.servers.stub]\n"
        "command = \"true\"\n"
    )

    with pytest.raises((ConfigError, ValueError)):
        ConfigLoader().load(path)


def test_the_input_shape_can_be_decoded() -> None:
    """The shape has to survive the argument decoder.

    A wrapper dataclass per item looked clearer and could not be
    decoded at all: the decoder has no nested-dataclass branch, so it
    handed back a raw dict and every item was rejected. The schema
    generator failed first and took the whole tools endpoint down with
    it, so both halves are checked here.
    """

    from agent_workflow.core.entities.models.arguments import (
        ArgumentDecoder,
    )

    decoded = ArgumentDecoder().decode(
        {"calls": [BatchCall(arguments={"tag_id": "a"}), BatchCall(arguments={"tag_id": "b"})]},
        BatchCallInput,
    )

    assert decoded.calls == [BatchCall(arguments={"tag_id": "a"}), BatchCall(arguments={"tag_id": "b"})]


def test_the_input_shape_produces_a_schema() -> None:
    """The tools page builds a schema for every registered tool.

    An unhandled type here is a 500 on /api/tools, which takes the
    whole page down rather than one entry.
    """

    from agent_workflow.core.entities.models.tool import (
        Tool,
        ToolPolicy,
    )
    from agent_workflow.core.entities.models.tool_definition_builder import (
        ToolDefinitionBuilder,
    )

    async def handler(arguments, context):
        return ""

    definition = ToolDefinitionBuilder().build(
        Tool(
            name="mcp_stub_echo_batch",
            description="batch",
            input_type=BatchCallInput,
            handler=handler,
            policy=ToolPolicy(
                permissions=("mcp.execute",),
                timeout=10.0,
                max_output_size=1000,
            ),
        )
    )

    calls = definition.input_schema["properties"]["calls"]

    assert calls["type"] == "array"
    assert calls["items"]["type"] == "object"


async def test_the_tools_endpoint_lists_a_batch_tool() -> None:
    """The failure this whole shape was chosen to avoid.

    Every registered tool is rendered through the schema generator when
    the tools page is opened, so one tool the generator cannot express
    made the page return 500 for everything.
    """

    from agent_workflow.core.entities.models.tool_definition_builder import (
        ToolDefinitionBuilder,
    )

    registry = ToolRegistry()

    manager = MCPManager(
        registry=registry,
        config=MCPConfig(
            enabled=True,
            servers=(
                MCPServerConfigEntry(
                    name="stub",
                    command=sys.executable,
                    args=(STUB,),
                ),
            ),
            batch_tools=("*",),
        ),
    )

    await manager.connect_all()

    try:
        builder = ToolDefinitionBuilder()

        for tool in registry.all():
            builder.build(tool)
    finally:
        await manager.close()

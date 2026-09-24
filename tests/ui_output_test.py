from types import SimpleNamespace
from typing import Any

import pytest

from cli.approval import ApprovalController
from cli.ui.app import AgentUI
from cli.ui.state import (
    TOOL_OUTPUT_COLLAPSED_CHARS,
    TOOL_OUTPUT_PREVIEW_CHARS,
    ToolStatus,
    ToolView,
    research_summary,
    store_tool_output,
)
from core.context.history import (
    HistoryKind,
    InMemoryHistoryStore,
)
from core.entities.models.research_contract import (
    ResearchPage,
    ResearchResult,
)


def _research_json() -> str:
    return ResearchResult(
        root_url="https://example.com/",
        pages=(
            ResearchPage(
                url="https://example.com/",
                depth=0,
                title="Example",
                content="body",
                links=(),
                content_bytes=4,
            ),
        ),
        discovered_urls=(),
        failed_urls=("https://example.com/bad",),
        max_depth_reached=0,
        total_bytes=4,
        page_limit_reached=False,
        byte_limit_reached=False,
    ).to_json()


def test_store_short_output_kept() -> None:
    kept, evicted, length = store_tool_output("hello")

    assert kept == "hello"
    assert evicted is False
    assert length == 5


def test_store_empty_output() -> None:
    assert store_tool_output(None) == (None, False, 0)
    assert store_tool_output("") == (None, False, 0)


def test_store_long_output_evicted() -> None:
    from cli.ui.state import TOOL_OUTPUT_KEEP_CHARS

    big = "x" * (TOOL_OUTPUT_KEEP_CHARS + 100)

    kept, evicted, length = store_tool_output(big)

    assert kept == big[:TOOL_OUTPUT_PREVIEW_CHARS]
    assert evicted is True
    assert length == len(big)


def test_store_medium_output_kept() -> None:
    from cli.ui.state import TOOL_OUTPUT_KEEP_CHARS

    medium = "x" * (TOOL_OUTPUT_KEEP_CHARS - 100)

    kept, evicted, length = store_tool_output(medium)

    assert kept == medium
    assert evicted is False
    assert length == len(medium)


def test_research_summary_card() -> None:
    summary = research_summary(_research_json())

    assert summary is not None
    assert "1 page(s)" in summary
    assert "failed 1" in summary


def test_research_summary_rejects_plain() -> None:
    assert research_summary("just text") is None
    assert research_summary(None) is None
    assert research_summary("{broken") is None


def test_finish_tool_evicts_and_summarizes() -> None:
    from cli.ui.state import UIState

    state = UIState()
    state.add_tool("call_1", "web_crawl", {})
    # Valid JSON + extra to trigger eviction but keep valid JSON for parsing
    base = _research_json()
    long_output = base + "\n" + "y" * 20_000
    state.finish_tool(
        "call_1",
        long_output,
        None,
        None,
        duration_seconds=12.0,
    )

    [item] = [
        entry
        for entry in state.conversation
        if isinstance(entry, ToolView)
    ]

    assert item.status is ToolStatus.SUCCESS
    assert item.output_evicted is True
    assert item.output_full_length > 20_000
    assert item.research_summary is not None
    assert item.duration_seconds == 12.0
    assert item.expanded is False


def test_by_reference_lookup() -> None:
    store = InMemoryHistoryStore()
    store.append(
        "task-01", "run-01", "window-01",
        HistoryKind.TOOL_RESULT, "full body here",
        reference="call_9",
    )

    found = store.by_reference("call_9")

    assert len(found) == 1
    assert found[0].content == "full body here"
    assert store.by_reference("missing") == ()


def _app(history: Any = None) -> AgentUI:
    controller = SimpleNamespace(history=history)
    agent = SimpleNamespace(controller=controller)

    app = AgentUI(
        runtime=SimpleNamespace(agent=agent),
        approval=ApprovalController(),
    )
    app._invalidate_ui = lambda **kwargs: None  # type: ignore[method-assign]

    return app


def _view(call_id: str = "call_1") -> ToolView:
    return ToolView(
        call_id=call_id,
        name="web_crawl",
        arguments={},
        status=ToolStatus.SUCCESS,
    )


def test_collapsed_body_bounded() -> None:
    app = _app()
    view = _view()
    view.output = "z" * 5_000
    view.output_evicted = True
    view.output_full_length = 100_000

    parts = app._tool_output_body(view)
    text = "\n".join(str(part) for part in parts)

    assert len(text) < TOOL_OUTPUT_COLLAPSED_CHARS + 200
    assert "100000" in text


def _part_text(part: Any) -> str:
    markup = getattr(part, "markup", None)

    if isinstance(markup, str):
        return markup

    return str(part)


def test_expanded_restores_from_history() -> None:
    store = InMemoryHistoryStore()
    store.append(
        "task-01", "run-01", "window-01",
        HistoryKind.TOOL_RESULT, "HISTORY FULL BODY",
        reference="call_1",
    )

    app = _app(history=store)
    view = _view()
    view.output = "preview"
    view.output_evicted = True
    view.output_full_length = 50_000
    view.expanded = True

    parts = app._tool_output_body(view)
    text = "\n".join(_part_text(part) for part in parts)

    assert "HISTORY FULL BODY" in text


def test_expanded_without_history_notes_eviction() -> None:
    app = _app()
    view = _view()
    view.output = "preview"
    view.output_evicted = True
    view.output_full_length = 50_000
    view.expanded = True

    parts = app._tool_output_body(view)
    text = "\n".join(str(part) for part in parts)

    assert "evicted from UI memory" in text


def test_research_card_rendered_first() -> None:
    app = _app()
    view = _view()
    view.output = "preview"
    view.output_evicted = True
    view.output_full_length = 50_000
    view.research_summary = "research: 1 page(s)"

    parts = app._tool_output_body(view)
    text = "\n".join(str(part) for part in parts)

    assert "research: 1 page(s)" in text


class _FakeBar:
    def __init__(self) -> None:
        self.text = ""

    def update(self, text: Any) -> None:
        self.text = str(text)


def test_status_shows_ctx_against_model_limit() -> None:
    app = _app()
    app.status_bar = _FakeBar()  # type: ignore[assignment]
    app.state.running = True
    app.state.iteration = 3
    app.state.estimated_tokens = 4200
    app.state.model_context_size = 8192

    app._render_status()

    assert "ctx ~4.2k/8.2k" in app.status_bar.text


def test_status_without_limit_shows_estimate() -> None:
    app = _app()
    app.status_bar = _FakeBar()  # type: ignore[assignment]
    app.state.running = True
    app.state.estimated_tokens = 300

    app._render_status()

    assert "ctx ~300" in app.status_bar.text


def test_status_without_estimate_no_ctx() -> None:
    app = _app()
    app.status_bar = _FakeBar()  # type: ignore[assignment]
    app.state.running = True

    app._render_status()

    assert "ctx" not in app.status_bar.text


def test_duration_format() -> None:
    assert AgentUI._format_duration(None) == ""
    assert AgentUI._format_duration(0.05) == " · 50ms"
    assert AgentUI._format_duration(12.0) == " · 12s"


@pytest.mark.asyncio
async def test_agent_emits_tokens_and_duration(
    tmp_path: Any,
) -> None:
    from pathlib import Path

    from core.entities.models.agent import Agent
    from core.entities.models.agent_trace import (
        LLMRequested,
        ToolFinished,
    )
    from core.entities.models.builtin.read_file import (
        ReadFileInput,
        read_file,
    )
    from core.entities.models.llm import LLMResponse, LLMToolCall
    from core.entities.models.llm_client import LLMClient
    from core.entities.models.tool import (
        Tool,
        ToolContext,
        ToolPolicy,
    )
    from core.entities.models.tool_executor import ToolExecutor
    from core.entities.models.tool_registry import ToolRegistry

    tmp = Path(str(tmp_path))
    (tmp / "test.txt").write_text("hello", encoding="utf-8")

    class FakeLLM(LLMClient):
        def __init__(self) -> None:
            self.calls = 0

        async def chat(
            self,
            messages: list[dict[str, Any]],
            tools: tuple[Any, ...] = (),
        ) -> LLMResponse:
            self.calls += 1

            if self.calls == 1:
                return LLMResponse(
                    content="",
                    tool_calls=(
                        LLMToolCall(
                            id="call_1",
                            name="read_file",
                            arguments={"path": "test.txt"},
                        ),
                    ),
                    raw={},
                )

            return LLMResponse(
                content="done",
                tool_calls=(),
                raw={},
            )

    async def handler(
        arguments: ReadFileInput,
        context: ToolContext,
    ) -> str:
        return await read_file(arguments, context)

    registry = ToolRegistry()
    registry.register(
        Tool(
            name="read_file",
            description="read",
            input_type=ReadFileInput,
            handler=handler,
            policy=ToolPolicy(
                permissions=frozenset({"filesystem.read"}),
                timeout=1.0,
                max_output_size=100_000,
            ),
        )
    )

    events: list[Any] = []

    async def on_event(event: Any) -> None:
        events.append(event)

    agent = Agent(
        llm=FakeLLM(),
        registry=registry,
        executor=ToolExecutor(registry),
    )

    result = await agent.run(
        prompt="Read test.txt",
        context=ToolContext(
            working_directory=tmp,
            environment={},
            allowed_path=(tmp,),
            permissions=frozenset({"filesystem.read"}),
        ),
        on_event=on_event,
    )

    assert result == "done"

    requested = [
        event
        for event in events
        if isinstance(event, LLMRequested)
    ]
    finished = [
        event
        for event in events
        if isinstance(event, ToolFinished)
    ]

    assert requested
    assert all(
        event.estimated_tokens is not None
        and event.estimated_tokens > 0
        for event in requested
    )
    assert len(finished) == 1
    assert finished[0].duration_seconds is not None
    assert finished[0].duration_seconds >= 0

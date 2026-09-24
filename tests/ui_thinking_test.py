import pytest

from cli.ui.state import (
    AgentRunView,
    ThinkingView,
    UIState,
)


@pytest.fixture(
    params=[
        UIState,
        lambda: AgentRunView(
            run_id="run-1",
            parent_run_id=None,
        ),
    ],
    ids=[
        "ui-state",
        "run-view",
    ],
)
def thinking_state(request: pytest.FixtureRequest) -> object:
    return request.param()


def _views(state: object) -> list[ThinkingView]:
    return [
        item
        for item in state.conversation  # type: ignore[union-attr]
        if isinstance(item, ThinkingView)
    ]


def test_no_view_without_content(
    thinking_state: object,
) -> None:
    thinking_state.start_thinking(1)  # type: ignore[union-attr]
    thinking_state.stop_thinking()  # type: ignore[union-attr]

    assert _views(thinking_state) == []


def test_view_created_on_first_content(
    thinking_state: object,
) -> None:
    thinking_state.start_thinking(2)  # type: ignore[union-attr]
    thinking_state.append_thinking("hello")  # type: ignore[union-attr]

    views = _views(thinking_state)

    assert len(views) == 1
    assert views[0].iteration == 2
    assert views[0].content == "hello"


def test_empty_chunks_never_create_view(
    thinking_state: object,
) -> None:
    thinking_state.start_thinking(1)  # type: ignore[union-attr]
    thinking_state.append_thinking("")  # type: ignore[union-attr]

    assert _views(thinking_state) == []


def test_next_iteration_gets_own_view(
    thinking_state: object,
) -> None:
    thinking_state.start_thinking(1)  # type: ignore[union-attr]
    thinking_state.append_thinking("first")  # type: ignore[union-attr]
    thinking_state.start_thinking(2)  # type: ignore[union-attr]

    # No content yet: still a single view.
    assert len(_views(thinking_state)) == 1

    thinking_state.append_thinking("second")  # type: ignore[union-attr]

    views = _views(thinking_state)

    assert [(view.iteration, view.content) for view in views] == [
        (1, "first"),
        (2, "second"),
    ]


def test_content_never_leaks_into_older_view(
    thinking_state: object,
) -> None:
    from cli.ui.state import ToolView

    thinking_state.start_thinking(1)  # type: ignore[union-attr]
    thinking_state.append_thinking("first")  # type: ignore[union-attr]
    thinking_state.conversation.append(  # type: ignore[union-attr]
        ToolView(
            call_id="call_1",
            name="read_file",
            arguments={},
        )
    )
    thinking_state.start_thinking(2)  # type: ignore[union-attr]
    thinking_state.append_thinking("second")  # type: ignore[union-attr]

    views = _views(thinking_state)

    assert views[0].content == "first"
    assert views[1].content == "second"

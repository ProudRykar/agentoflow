from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from agent_workflow.web.app import create_app


def make_stub_runtime_factory(
    tmp_path: Path,
):
    """Build a cheap ``create_runtime`` for transport tests.

    A real runtime opens SQLite, discovers plugins and constructs
    an Ollama client. None of that is needed to exercise the REST
    or WebSocket contract, so tests swap it out and keep only the
    Agent execution path.
    """

    from agent_workflow.core.application.runtime import AgentRuntime
    from agent_workflow.core.entities.models.agent import Agent
    from agent_workflow.core.entities.models.llm import LLMResponse
    from agent_workflow.core.entities.models.llm_client import LLMClient
    from agent_workflow.core.entities.models.tool import ToolContext
    from agent_workflow.core.entities.models.tool_executor import (
        ToolExecutor,
    )
    from agent_workflow.core.entities.models.tool_registry import (
        ToolRegistry,
    )
    from agent_workflow.core.entities.models.tool_toggle import (
        ToolToggle,
    )
    from agent_workflow.core.infrastructure.config import AgentConfig, LLMConfig
    from agent_workflow.core.infrastructure.paths import (
        AgentWorkflowPaths,
    )

    class SilentLLM(LLMClient):
        async def chat(
            self,
            messages,
            tools=(),
        ) -> LLMResponse:
            return LLMResponse(
                content="ok",
                tool_calls=(),
                raw={},
            )

    class NoopStore:
        def close(self) -> None:
            return None

    async def factory(
        approval_handler,
        working_directory=None,
        stats_store=None,
        mcp_pool=None,
        session_id="",
    ) -> AgentRuntime:
        # mcp_pool and session_id are accepted and ignored: a real
        # runtime shares MCP processes through the pool, and the stub
        # has no servers to share.
        del mcp_pool, session_id
        work = Path(
            working_directory or tmp_path
        ).resolve()

        registry = ToolRegistry()

        # Mirrors a real runtime, where the toggle owns which tools
        # are hidden. Left empty: registering stub tools here would
        # change what every other transport test sees.
        toggle = ToolToggle(registry)

        return AgentRuntime(
            agent=Agent(
                llm=SilentLLM(),
                registry=registry,
                executor=ToolExecutor(registry),
                run_id="run-stub",
            ),
            context=ToolContext(
                working_directory=work,
                environment={},
                allowed_path=(work,),
                permissions=frozenset(),
            ),
            store=NoopStore(),  # type: ignore[arg-type]
            paths=AgentWorkflowPaths(),
            config=type(
                "StubConfig",
                (),
                {
                    "llm": LLMConfig(model="stub-model"),
                    "agent": AgentConfig(),
                },
            )(),
            plugin_manager=None,
            tool_toggle=toggle,
        )

    return factory


@pytest.fixture(autouse=True)
def isolated_home(
    tmp_path_factory: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[Path]:
    """Point HOME at a temporary directory for the whole run.

    ``AgentWorkflowPaths`` derives its location from the home
    directory, so without this the suite would create config,
    memory and statistics files in the developer's real profile.
    """

    home = tmp_path_factory.mktemp("home")

    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))

    (home / ".agentoflow").mkdir(parents=True, exist_ok=True)

    yield home


@pytest.fixture(autouse=True)
def _no_leaked_network_outage():
    """Clear the recorded outage between tests.

    The guard in ``web_failure`` is deliberately process-wide, because
    that is what makes it suppress a burst of retries inside one run.
    Left in place it would carry a failure in one test into the next,
    which reads as a network bug that is not there.
    """

    from agent_workflow.core.entities.models import web_failure

    web_failure.reset()

    yield

    web_failure.reset()


@pytest.fixture
def client(tmp_path: Path) -> Iterator[TestClient]:
    """A client whose state lives entirely under ``tmp_path``.

    Without this the suite would create ``~/.agentoflow`` entries
    and a real statistics database as a side effect.
    """

    from agent_workflow.core.application.session import SessionManager
    from agent_workflow.core.application.session_store import (
        SessionStore,
    )
    from agent_workflow.core.application.tool_stats_store import (
        ToolStatsStore,
    )

    manager = SessionManager(
        store=SessionStore(tmp_path / "sessions.json"),
        stats_store=ToolStatsStore(tmp_path / "usage.db"),
    )

    app = create_app(session_manager=manager)

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def stubbed(
    client: TestClient,
    tmp_path: Path,
) -> Iterator[TestClient]:
    from agent_workflow.core.application import session as session_module

    original = session_module.create_runtime

    session_module.create_runtime = make_stub_runtime_factory(
        tmp_path
    )

    try:
        yield client
    finally:
        session_module.create_runtime = original


@pytest.fixture
def stub_factory(tmp_path: Path):
    """Expose the stub factory so tests can wrap it."""

    return make_stub_runtime_factory(tmp_path)


@pytest.fixture
def publish(client: TestClient):
    """Publish an event on the app's own event loop.

    ``TestClient`` serves the app in a worker thread, so events
    must be published through the portal rather than awaited
    directly from the test thread.
    """

    def _publish(session, event) -> None:
        client.portal.call(  # type: ignore[attr-defined]
            session.events.publish,
            event,
        )

    return _publish


@pytest.fixture
def portal(client: TestClient):
    """Run coroutines on the event loop serving the app."""

    return client.portal  # type: ignore[attr-defined]


# ======================================================================
# Restart fixtures
# ======================================================================


@pytest.fixture
def events_path(tmp_path: Path) -> Path:
    return tmp_path / "session-events.db"


@pytest.fixture
def registry_path(tmp_path: Path) -> Path:
    return tmp_path / "sessions.json"


@pytest.fixture
def state_path(tmp_path: Path) -> Path:
    return tmp_path / "session-state.db"

@pytest.fixture
def stub_runtime(tmp_path: Path):
    """Swap in the cheap runtime, as the transport tests do."""

    from agent_workflow.core.application import session as session_module

    original = session_module.create_runtime

    session_module.create_runtime = make_stub_runtime_factory(tmp_path)

    try:
        yield
    finally:
        session_module.create_runtime = original


def session_manager_factory():
    """Build a SessionManager over explicit store paths.

    Shared by the restart tests so they exercise the same wiring
    without reaching into each other's module namespace.
    """

    from agent_workflow.core.application.event_store import EventStore
    from agent_workflow.core.application.session import SessionManager
    from agent_workflow.core.application.session_state_store import (
        SessionStateStore,
    )
    from agent_workflow.core.application.session_store import (
        SessionStore,
    )

    def build(
        registry_path: Path,
        events_path: Path,
        state_path: Path | None = None,
    ) -> SessionManager:
        manager = SessionManager(
            store=SessionStore(registry_path),
            event_store=EventStore(events_path),
        )

        if state_path is not None:
            manager.set_state_store(SessionStateStore(state_path))

        return manager

    return build

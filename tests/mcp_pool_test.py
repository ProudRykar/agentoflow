"""MCP process sharing.

One process per server, borrowed by every session that asks for it.
The failure this prevents is a container per session: a stdio server
launched through podman or docker costs a container each time, so a
dozen sessions and four servers is forty containers.
"""

from __future__ import annotations

import asyncio

import pytest

from agent_workflow.core.infrastructure.config import (
    MCPServerConfigEntry,
)
from agent_workflow.core.infrastructure.mcp.pool import (
    MCPProcessPool,
    ServerKey,
)


def _entry(**over) -> MCPServerConfigEntry:
    base = {
        "name": "Stash",
        "transport": "stdio",
        "command": "podman",
        "args": ("run", "-i", "--rm", "stash-mcp:local"),
        "env": {"STASH_ENDPOINT": "http://host:9999"},
    }

    base.update(over)

    return MCPServerConfigEntry(**base)  # type: ignore[arg-type]


class _FakeClient:
    """Counts constructions, which is what a container would cost."""

    def __init__(self) -> None:
        self.connected = False
        self.closed = False
        self.tools: list[str] = []

    async def connect(self) -> object:
        self.connected = True

        return type("Info", (), {"version": "1.0"})()

    async def list_tools(self) -> list[str]:
        return list(self.tools)

    async def close(self) -> None:
        self.closed = True

    def stderr_tail(self, limit: int = 500) -> str:
        return ""


def _factory(built: list[_FakeClient]):
    def build() -> _FakeClient:
        client = _FakeClient()
        built.append(client)
        return client

    return build


# ======================================================================
# Keys
# ======================================================================


def test_the_key_ignores_the_server_name() -> None:
    # Two sessions naming the same server differently still share one
    # process; the name is a label, not an identity.
    assert ServerKey.of(_entry(name="A")) == ServerKey.of(
        _entry(name="B")
    )


def test_the_key_notices_a_changed_command() -> None:
    assert ServerKey.of(_entry()) != ServerKey.of(
        _entry(command="docker")
    )


def test_the_key_notices_changed_arguments() -> None:
    assert ServerKey.of(_entry()) != ServerKey.of(
        _entry(args=("run", "-i", "other:latest"))
    )


def test_the_key_notices_changed_env() -> None:
    assert ServerKey.of(_entry()) != ServerKey.of(
        _entry(env={"STASH_ENDPOINT": "http://elsewhere:9999"})
    )


def test_env_order_does_not_change_the_key() -> None:
    assert ServerKey.of(
        _entry(env={"A": "1", "B": "2"})
    ) == ServerKey.of(_entry(env={"B": "2", "A": "1"}))


def test_the_key_holds_no_secret() -> None:
    """A pooled key lives for the process lifetime."""

    key = ServerKey.of(
        _entry(env={"STASH_API_KEY": "super-secret-value"})
    )

    assert "super-secret-value" not in repr(key)


def test_http_keys_on_the_url() -> None:
    stdio = ServerKey.of(_entry())
    http = ServerKey.of(
        _entry(transport="http", url="https://mcp.example")
    )

    assert stdio != http


# ======================================================================
# Sharing
# ======================================================================


async def test_one_process_is_shared_by_every_session() -> None:
    pool = MCPProcessPool()
    built: list[_FakeClient] = []
    build = _factory(built)

    entry = _entry()

    first = await pool.acquire(entry, "s1", build)
    second = await pool.acquire(entry, "s2", build)
    third = await pool.acquire(entry, "s3", build)

    # The point of the whole thing.
    assert len(built) == 1
    assert pool.process_count() == 1
    assert pool.borrower_counts() == {"podman": 3}

    assert first[0] is second[0] is third[0]


async def test_different_definitions_get_different_processes() -> None:
    pool = MCPProcessPool()
    built: list[_FakeClient] = []
    build = _factory(built)

    await pool.acquire(_entry(), "s1", build)
    await pool.acquire(
        _entry(args=("run", "-i", "other:latest")),
        "s1",
        build,
    )

    assert len(built) == 2
    assert pool.process_count() == 2


async def test_the_process_survives_until_the_last_release() -> None:
    pool = MCPProcessPool()
    built: list[_FakeClient] = []
    entry = _entry()

    await pool.acquire(entry, "s1", _factory(built))
    await pool.acquire(entry, "s2", _factory(built))

    await pool.release(entry, "s1")

    assert pool.process_count() == 1
    assert built[0].closed is False

    await pool.release(entry, "s2")

    assert pool.process_count() == 0
    assert built[0].closed is True


async def test_a_late_session_reuses_a_surviving_process() -> None:
    pool = MCPProcessPool()
    built: list[_FakeClient] = []
    entry = _entry()

    await pool.acquire(entry, "s1", _factory(built))
    await pool.release(entry, "s1")
    await pool.acquire(entry, "s2", _factory(built))

    assert len(built) == 2

    client, _, _ = await pool.acquire(entry, "s3", _factory(built))

    assert client is built[1]
    assert len(built) == 2


async def test_concurrent_sessions_start_one_process() -> None:
    """A race here would silently reintroduce a container per session."""

    pool = MCPProcessPool()
    built: list[_FakeClient] = []

    def slow_build() -> _FakeClient:
        client = _SlowClient()
        built.append(client)

        return client

    entry = _entry()

    await asyncio.gather(
        *(
            pool.acquire(entry, f"s{index}", slow_build)
            for index in range(4)
        )
    )

    assert len(built) == 1
    assert pool.borrower_counts() == {"podman": 4}


# ======================================================================
# Failure
# ======================================================================


class _SlowClient(_FakeClient):
    """Handshakes slowly, so a second session would race it."""

    async def connect(self) -> object:
        await asyncio.sleep(0.05)

        self.connected = True

        return type("Info", (), {"version": "1.0"})()


class _BrokenClient(_FakeClient):
    async def connect(self) -> object:
        raise RuntimeError("server refused to start")

    def stderr_tail(self, limit: int = 500) -> str:
        return "Traceback: boom"


class _SlowBrokenClient(_BrokenClient):
    """Fails, but only after a waiter's turn has come round."""

    async def connect(self) -> object:
        await asyncio.sleep(0.05)

        raise RuntimeError("server refused to start")


async def test_a_failed_start_reports_the_stderr() -> None:
    pool = MCPProcessPool()

    with pytest.raises(RuntimeError, match="server refused"):
        await pool.acquire(_entry(), "s1", _BrokenClient)

    assert pool.process_count() == 0


async def test_a_waiter_learns_the_first_one_failed() -> None:
    """Otherwise the second session waits on a process that will
    never exist."""

    pool = MCPProcessPool()
    entry = _entry()

    def slow_build() -> _BrokenClient:
        return _SlowBrokenClient()

    results = await asyncio.gather(
        pool.acquire(entry, "s1", slow_build),
        pool.acquire(entry, "s2", slow_build),
        return_exceptions=True,
    )

    assert all(
        isinstance(result, RuntimeError) for result in results
    )

    assert pool.process_count() == 0


# ======================================================================
# Teardown
# ======================================================================


async def test_close_all_shuts_everything_down() -> None:
    pool = MCPProcessPool()
    built: list[_FakeClient] = []
    build = _factory(built)

    await pool.acquire(_entry(), "s1", build)
    await pool.acquire(
        _entry(command="docker"), "s1", build
    )

    await pool.close_all()

    assert pool.process_count() == 0
    assert all(client.closed for client in built)


async def test_releasing_an_unknown_entry_is_harmless() -> None:
    pool = MCPProcessPool()

    # A session releasing something it never acquired must not raise.
    await pool.release(_entry(), "ghost")

# ======================================================================
# Context window reconciliation
# ======================================================================


def test_a_missing_catalog_entry_falls_back_to_the_model() -> None:
    from agent_workflow.core.application.runtime import (
        _effective_context_size,
    )

    assert (
        _effective_context_size(None, 256_000, model_name="m")
        == 256_000
    )


def test_an_unreachable_server_leaves_the_catalog_in_charge() -> None:
    from agent_workflow.core.application.runtime import (
        _effective_context_size,
    )

    assert (
        _effective_context_size(128_000, None, model_name="m")
        == 128_000
    )


def test_a_catalog_below_the_model_is_still_the_ceiling() -> None:
    """num_ctx comes from the catalog, so no request can exceed it."""

    from agent_workflow.core.application.runtime import (
        _effective_context_size,
    )

    assert (
        _effective_context_size(128_000, 256_000, model_name="m")
        == 128_000
    )


def test_a_catalog_above_the_model_is_corrected_down() -> None:
    """A window the model does not have must not be promised."""

    from agent_workflow.core.application.runtime import (
        _effective_context_size,
    )

    assert (
        _effective_context_size(128_000, 64_000, model_name="m")
        == 64_000
    )


def test_agreement_is_left_alone() -> None:
    from agent_workflow.core.application.runtime import (
        _effective_context_size,
    )

    assert (
        _effective_context_size(128_000, 128_000, model_name="m")
        == 128_000
    )


def test_a_mismatch_is_reported(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Silently absorbing a wrong catalog is how capacity goes unused."""

    import logging

    from agent_workflow.core.application.runtime import (
        _effective_context_size,
    )

    with caplog.at_level(logging.WARNING):
        _effective_context_size(
            128_000, 256_000, model_name="gemma4:e4b-it-qat"
        )

    assert "128000" in caplog.text
    assert "256000" in caplog.text
    assert "gemma4:e4b-it-qat" in caplog.text

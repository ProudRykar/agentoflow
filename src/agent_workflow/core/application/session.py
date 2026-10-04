from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from agent_workflow.core.application.event_store import (
    EVENTS_DATABASE,
    EventStore,
    dialogue_from_events,
)
from agent_workflow.core.application.session_state_store import (
    STATE_DATABASE,
    SessionStateStore,
)
from agent_workflow.core.application.events import EventBus
from agent_workflow.core.application.session_store import (
    SessionRecord,
    SessionStore,
)
from agent_workflow.core.application.titles import (
    MAX_TITLE_LENGTH,
    derive_title,
    generate_title,
)
from agent_workflow.core.application.tool_stats import ToolStats
from agent_workflow.core.application.runtime import (
    STATS_DATABASE,
    AgentRuntime,
    create_runtime,
)
from agent_workflow.core.application.tool_stats_store import (
    ToolStatsStore,
)
from agent_workflow.core.infrastructure.paths import (
    AgentWorkflowPaths,
)
from agent_workflow.core.entities.models.approval import (
    ApprovalController,
    ApprovalRequested,
    ApprovalResolved,
)
from agent_workflow.core.entities.models.planner import Planner
from agent_workflow.core.entities.models.task_contract import TaskContract


# Streaming a reply emits an event per chunk. Batching keeps the
# SQLite write off the per-chunk path.
EVENT_FLUSH_THRESHOLD = 25


class SessionState(StrEnum):
    IDLE = "idle"
    RUNNING = "running"
    WAITING_APPROVAL = "waiting_approval"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    ERROR = "error"
    CLOSED = "closed"


@dataclass(slots=True)
class AgentSession:
    """One independent agent conversation plus its event fan-out.

    A session owns its own ``AgentRuntime`` (Agent, registry,
    memory store) and its own ``EventBus``. Two sessions never
    share an Agent, a ToolRegistry or an event stream.
    """

    session_id: str
    runtime: AgentRuntime
    approval: ApprovalController
    events: EventBus
    stats: ToolStats
    state: SessionState = SessionState.IDLE
    created_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    title: str = ""
    title_source: str = "auto"
    _title_task: asyncio.Task[Any] | None = None

    _run_task: asyncio.Task[Any] | None = None
    _last_error: str = ""
    _first_message_used: bool = False

    # Durable transcript. Injected by the SessionManager; a session
    # built without one behaves exactly as before.
    event_store: Any = None
    _pending_events: list[Any] = field(default_factory=list)

    # Called with this session when working state should be written.
    state_persist: Any = None
    _state_persist_error: str = ""

    def __post_init__(self) -> None:
        self.approval.set_on_event(self._on_approval_event)

        # Statistics are an observer on the same bus the UI reads,
        # so they need no hooks inside the agent loop.
        self.events.subscribe(self._observe_event)

        if self.event_store is not None:
            self.events.subscribe(self._persist_event)

        self.stats.session_id = self.session_id
        self.stats.sink = self.runtime.stats_store

    def _persist_event(
        self,
        seq: int,
        event: Any,
    ) -> None:
        """Buffer a published event for the durable log.

        Streaming emits an event per chunk, so writes are batched
        and flushed on a size threshold, at the end of a run and on
        close. A hard kill can therefore lose the last few events,
        which is the same guarantee an append-only log gives.
        """

        from agent_workflow.core.application.events import StoredEvent

        self._pending_events.append(
            StoredEvent(seq=seq, event=event)
        )

        if len(self._pending_events) >= EVENT_FLUSH_THRESHOLD:
            self.flush_events()

    def flush_events(self) -> None:
        """Write buffered events and working state to disk."""

        self.flush_working_state()

        if not self._pending_events or self.event_store is None:
            return

        batch = self._pending_events
        self._pending_events = []

        try:
            self.event_store.append(self.session_id, batch)
        except Exception:
            # Losing telemetry must never break the conversation.
            return

    def note_state_persist_failure(self, cause: Exception) -> None:
        """Record why working state could not be written.

        Swallowing this silently made a broken persistence path look
        like "nothing to save", which is exactly the bug this state
        store was added to fix.
        """

        self._state_persist_error = f"{type(cause).__name__}: {cause}"

    def flush_working_state(self) -> None:
        """Ask the owner to persist anchor/checkpoint/evidence."""

        if self.state_persist is None:
            return

        try:
            self.state_persist(self)
        except Exception:
            # Best-effort: a run must never fail on persistence.
            return

    def restore_transcript(
        self,
        events: list[Any],
    ) -> None:
        """Re-seed this session from its durable transcript.

        Called before anyone subscribes, so the restored events are
        replayed to the client instead of delivered twice.
        """

        self.events.restore(events)

        dialogue = dialogue_from_events(events)

        if not dialogue:
            return

        # runtime.context is the tool sandbox, not the
        # conversation: the window lives on the agent.
        self.runtime.agent.context_manager.restore(dialogue)

        # A restored transcript already contains a first turn, so the
        # next message must continue it. Without this the session
        # would answer with ``run``, and run clears the very window
        # that was just restored.
        self._first_message_used = True

    def _observe_event(
        self,
        seq: int,
        event: Any,
    ) -> None:
        del seq

        self.stats.observe(event)

    # ==================================================================
    # Accessors
    # ==================================================================

    @property
    def agent(self):
        return self.runtime.agent

    @property
    def context(self):
        return self.runtime.context

    @property
    def config(self):
        return self.runtime.config

    @property
    def working_directory(self) -> Path:
        return self.context.working_directory

    @property
    def last_error(self) -> str:
        return self._last_error

    def refresh_permissions(self) -> None:
        """Re-derive ToolContext permissions from current managers.

        Called after MCP servers are connected or disconnected so
        the agent never holds a permission the runtime no longer
        grants. Granted-but-stale permissions are removed; manual
        approvals are left alone because they belong to the user.
        """

        self.context.permissions = self.runtime.refresh_permissions()

    @property
    def running(self) -> bool:
        return self._run_task is not None and not (
            self._run_task.done()
        )

    @property
    def display_title(self) -> str:
        """Human label for the session.

        Falls back to the working directory so a session is always
        identifiable, even before its title has been generated.
        """

        if self.title:
            return self.title

        return (
            self.working_directory.name or self.session_id[:8]
        )

    @property
    def has_history(self) -> bool:
        """True once a first turn has been accepted.

        Callers use this to decide between ``run`` (new
        conversation) and ``continue_run`` (same conversation),
        which is the same branch the TUI drives with its
        ``first_message`` flag.
        """

        return self._first_message_used

    # ==================================================================
    # Events
    # ==================================================================

    async def _on_approval_event(
        self,
        event: ApprovalRequested | ApprovalResolved,
    ) -> None:
        if isinstance(event, ApprovalRequested):
            self.state = SessionState.WAITING_APPROVAL

        await self.events.publish(event)

        if isinstance(event, ApprovalResolved):
            self.state = (
                SessionState.RUNNING
                if self.running
                else self.state
            )

    async def emit(self, event: Any) -> None:
        """Adapter handed to ``Agent.run`` as ``on_event``.

        Keeping the reference on the session (rather than on
        ``ToolContext``) is what lets a WebSocket attach and
        detach without the Agent overwriting it on every run.
        """

        await self.events.publish(event)

    # ==================================================================
    # Task preparation
    # ==================================================================

    @staticmethod
    def build_task_contract(
        prompt: str,
    ) -> tuple[Any, TaskContract]:
        """Plan the request once, exactly like the TUI does.

        Planner is invoked per user request; the Agent must not
        re-plan a task the caller already prepared.
        """

        task_plan = Planner().plan(prompt)

        research = task_plan.research

        if research is None or not research.root_urls:
            research = None

        task_contract = TaskContract(
            requires_research=research is not None,
            research=research,
        )

        return task_plan, task_contract

    # ==================================================================
    # Execution
    # ==================================================================

    async def start_run(self, prompt: str) -> None:
        """Start a new conversation (``Agent.run``).

        The caller decides between run and continue; the session
        does not second-guess it. Use ``has_history`` to pick.
        """

        self._require_idle()

        task_plan, task_contract = self.build_task_contract(prompt)

        self.agent.orchestrator.prepare_task(
            task_plan,
            task_contract,
        )

        first_turn = not self._first_message_used

        self._first_message_used = True

        self._maybe_generate_title(prompt)

        await self._launch(
            lambda: self.agent.run(
                prompt=prompt,
                context=self.context,
                on_event=self.emit,
                task_contract=task_contract,
            ),
        )

    async def start_continue(self, prompt: str) -> None:
        """New task, preserved conversation (``continue_run``)."""

        self._require_idle()

        task_plan, task_contract = self.build_task_contract(prompt)

        self.agent.orchestrator.prepare_task(
            task_plan,
            task_contract,
        )

        self._first_message_used = True

        await self._launch(
            lambda: self.agent.continue_run(
                prompt=prompt,
                context=self.context,
                on_event=self.emit,
                task_contract=task_contract,
            ),
        )

    async def start_resume(self) -> None:
        """Resume the paused task (``resume_run``)."""

        self._require_idle()

        await self._launch(
            lambda: self.agent.resume_run(
                context=self.context,
                on_event=self.emit,
            ),
        )

    def _require_idle(self) -> None:
        if self.state is SessionState.CLOSED:
            raise RuntimeError("Session is closed")

        if self.running:
            raise RuntimeError(
                "A run is already in progress for this session",
            )

    async def _launch(
        self,
        factory: Callable[[], Any],
    ) -> None:
        self.state = SessionState.RUNNING
        self._last_error = ""

        task = asyncio.create_task(
            self._supervise(factory),
        )

        self._run_task = task

    async def _supervise(
        self,
        factory: Callable[[], Any],
    ) -> None:
        try:
            await factory()

            if self.state is SessionState.RUNNING:
                self.state = SessionState.COMPLETED

        except asyncio.CancelledError:
            if self.state is not SessionState.CLOSED:
                self.state = SessionState.CANCELLED

            raise

        except Exception as exc:
            self._last_error = str(exc)
            self.state = SessionState.ERROR

            await self.events.publish(
                RunFailed(
                    session_id=self.session_id,
                    error=str(exc),
                ),
            )

        finally:
            self._run_task = None

            if self.approval.active:
                self.approval.deny()

            self.flush_events()

            self.stats.flush()

    async def cancel(self) -> bool:
        """Cooperative cancellation of the current run."""

        if self.approval.active:
            await self.approval.cancel()

        task = self._run_task

        if task is None or task.done():
            if self.state is SessionState.RUNNING:
                self.state = SessionState.IDLE

            return False

        task.cancel()

        try:
            await task
        except (asyncio.CancelledError, Exception):
            pass

        self.state = SessionState.CANCELLED

        return True

    def rename(
        self,
        title: str,
    ) -> str:
        """Set the title manually; overrides any generated one."""

        cleaned = " ".join(str(title or "").split()).strip()

        if not cleaned:
            raise ValueError("Title cannot be empty")

        if len(cleaned) > MAX_TITLE_LENGTH:
            raise ValueError(
                f"Title must be at most {MAX_TITLE_LENGTH} characters"
            )

        self.title = cleaned
        self.title_source = "manual"

        return cleaned

    def _maybe_generate_title(self, prompt: str) -> None:
        """Kick off title generation for the first user turn.

        Runs in the background so the conversation is never delayed,
        and only while the title is still the derived placeholder.
        """

        if self.title or self.title_source == "manual":
            return

        self.title = derive_title(prompt)
        self.title_source = "auto"

        llm = getattr(self.runtime.agent, "_llm", None)

        if llm is None:
            return

        self._title_task = asyncio.create_task(
            self._generate_title(llm, prompt)
        )

    async def _generate_title(
        self,
        llm: Any,
        prompt: str,
    ) -> None:
        try:
            title = await generate_title(llm, prompt)
        except Exception:
            return

        # A manual rename during generation wins.
        if self.title_source == "manual" or self.title == title:
            return

        self.title = title
        self.title_source = "auto"

    def flush_stats(self) -> None:
        """Persist accumulated counters, if a store is configured."""

        self.flush_events()

        self.stats.flush()

    async def close(self) -> None:
        if self.state is SessionState.CLOSED:
            return

        self.state = SessionState.CLOSED

        await self.approval.cancel()

        task = self._run_task

        if task is not None and not task.done():
            task.cancel()

            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass

        self._run_task = None

        self.flush_events()

        self.stats.flush()

        await self.runtime.aclose()

    def snapshot(self) -> dict[str, Any]:
        state = self.agent.orchestrator.state

        return {
            "session_id": self.session_id,
            "state": self.state.value,
            "title": self.display_title,
            "title_source": self.title_source,
            "created_at": self.created_at,
            "metadata": dict(self.metadata),
            "model": self.config.llm.model,
            "working_directory": str(
                self.working_directory
            ),
            "phase": getattr(
                getattr(state, "phase", None),
                "value",
                None,
            ),
            "iteration": getattr(
                state,
                "iteration",
                None,
            ),
            "running": self.running,
            "last_error": self._last_error,
            "stats": self.stats.summary(),
            "approval": (
                {
                    "approval_id": self.approval.approval_id,
                    "tool_name": (
                        self.approval.request.tool_name
                        if self.approval.request
                        else ""
                    ),
                    "permission": (
                        self.approval.request.permission
                        if self.approval.request
                        else ""
                    ),
                    "reason": (
                        self.approval.request.reason
                        if self.approval.request
                        else ""
                    ),
                    "arguments": (
                        dict(self.approval.request.arguments)
                        if self.approval.request
                        else {}
                    ),
                }
                if self.approval.active
                else None
            ),
        }


@dataclass(slots=True, frozen=True)
class RunFailed:
    """Session-level failure, not an Agent event.

    Agent errors surface as exceptions from ``run``; the web
    layer needs them as a stream item so the browser can render
    them like any other message.
    """

    session_id: str
    error: str


class SessionManager:
    """Registry of independent sessions.

    Deliberately free of HTTP/FastAPI types so the same manager can
    back the web API, a CLI, or a test.

    Session *identity* is persisted through ``SessionStore`` so a
    restart does not orphan bookmarks. The live runtime is not
    persisted; a recorded session is rehydrated on first access.
    """

    def __init__(
        self,
        store: SessionStore | None = None,
        stats_store: ToolStatsStore | None = None,
        auto_restore: bool = True,
        event_store: EventStore | None = None,
        state_store: SessionStateStore | None = None,
    ) -> None:
        self._sessions: dict[str, AgentSession] = {}
        self._store = store
        self._stats_store = stats_store
        self._event_store = (
            event_store if event_store is not None else None
        )
        self._state_store = state_store

        # Recorded sessions whose runtime has not been rebuilt yet.
        self._pending: dict[str, SessionRecord] = {}

        if self._store is None and auto_restore:
            try:
                self._store = SessionStore.default()
            except Exception:
                self._store = None

        # ``auto_restore`` is authoritative: passing it as False
        # gives a manager that starts empty even when a store is
        # supplied, which is what tests and CLIs want.
        if self._store is not None and auto_restore:
            self._restore()

        if self._stats_store is None:
            self._stats_store = self._build_stats_store()

        if self._event_store is None and auto_restore:
            self._event_store = self._build_event_store()

        if self._state_store is None and auto_restore:
            self._state_store = self._build_state_store()

    def _build_stats_store(self) -> ToolStatsStore | None:
        """One shared statistics store for every session.

        Sessions must not each open their own connection to the
        same database, so the manager owns the store and hands it
        to every runtime it creates.
        """

        try:
            paths = AgentWorkflowPaths()

            paths.ensure()

            return ToolStatsStore(
                paths.resolve(STATS_DATABASE)
            )
        except Exception:
            # Statistics are optional; never block startup.
            return None

    def _build_event_store(self) -> EventStore | None:
        """Durable transcript log, shared like the statistics store."""

        try:
            paths = AgentWorkflowPaths()

            paths.ensure()

            return EventStore(paths.resolve(EVENTS_DATABASE))
        except Exception:
            # Persistence is an enhancement: a session that cannot
            # write its transcript must still run.
            return None

    def _build_state_store(self) -> SessionStateStore | None:
        """Durable working state: anchor, checkpoint, evidence."""

        try:
            paths = AgentWorkflowPaths()

            paths.ensure()

            return SessionStateStore(paths.resolve(STATE_DATABASE))
        except Exception:
            return None

    @property
    def state_store(self) -> SessionStateStore | None:
        return self._state_store

    @property
    def stats_store(self) -> ToolStatsStore | None:
        return self._stats_store

    @property
    def event_store(self) -> EventStore | None:
        return self._event_store

    def set_stats_store(
        self,
        store: ToolStatsStore | None,
    ) -> None:
        """Attach a statistics store after construction.

        Used by tests and embedders that need an isolated
        database instead of the one under the user's home.
        """

        self._stats_store = store

    def set_event_store(
        self,
        store: EventStore | None,
    ) -> None:
        """Attach a transcript store after construction."""

        self._event_store = store

    def set_state_store(
        self,
        store: SessionStateStore | None,
    ) -> None:
        """Attach a working-state store after construction."""

        self._state_store = store

    def _restore(self) -> None:
        if self._store is None:
            return

        try:
            records = self._store.load()
        except Exception:
            return

        self._pending = {
            record.session_id: record for record in records
        }

    async def create_session(
        self,
        metadata: dict[str, Any] | None = None,
        working_directory: Path | None = None,
    ) -> AgentSession:
        session_id = str(uuid.uuid4())

        approval = ApprovalController()

        runtime = await create_runtime(
            approval_handler=approval,
            working_directory=working_directory,
            stats_store=self._stats_store,
        )

        session = AgentSession(
            session_id=session_id,
            runtime=runtime,
            approval=approval,
            events=EventBus(),
            stats=ToolStats(),
            metadata=dict(metadata or {}),
            event_store=self._event_store,
            state_persist=self._remember_working_state,
        )

        self._sessions[session_id] = session
        self._pending.pop(session_id, None)

        self._remember(session)
        self._remember_working_state(session)

        return session

    def _remember(self, session: AgentSession) -> None:
        if self._store is None:
            return

        try:
            self._store.upsert(
                SessionRecord(
                    session_id=session.session_id,
                    working_directory=str(
                        session.working_directory
                    ),
                    created_at=session.created_at,
                    metadata=dict(session.metadata),
                    title=session.display_title,
                    title_source=session.title_source,
                )
            )
        except Exception:
            # Persistence is best-effort and must never break a run.
            pass

    def has_session(self, session_id: str) -> bool:
        return (
            session_id in self._sessions
            or session_id in self._pending
        )

    async def get_session(
        self,
        session_id: str,
    ) -> AgentSession | None:
        """Return a live session, rebuilding a recorded one if needed."""

        existing = self._sessions.get(session_id)

        if existing is not None:
            return existing

        record = self._pending.pop(session_id, None)

        if record is None:
            return None

        directory: Path | None = None

        if record.working_directory:
            candidate = Path(record.working_directory)

            if candidate.is_dir():
                directory = candidate

        try:
            approval = ApprovalController()

            runtime = await create_runtime(
                approval_handler=approval,
                working_directory=directory,
                stats_store=self._stats_store,
            )
        except Exception:
            # The record is deliberately kept: a transient failure
            # must not silently destroy a user's session entry, and
            # deleting on an unexpected error would also mask real
            # bugs as "session gone".
            return None

        session = AgentSession(
            session_id=record.session_id,
            runtime=runtime,
            approval=approval,
            events=EventBus(),
            stats=ToolStats(),
            state=SessionState.IDLE,
            created_at=record.created_at or time.time(),
            metadata=dict(record.metadata),
            title=record.title,
            title_source=record.title_source,
            event_store=self._event_store,
            state_persist=self._remember_working_state,
        )

        self._sessions[session_id] = session

        self._restore_transcript(session)

        return session

    def _restore_transcript(
        self,
        session: AgentSession,
    ) -> None:
        """Re-seed a rehydrated session with its stored transcript.

        Runs before the session is published to any caller, so the
        restored events are replayed to the client exactly like a
        live reconnect instead of arriving twice.
        """

        if self._event_store is not None:
            try:
                events = self._event_store.load(session.session_id)
            except Exception:
                events = []

            if events:
                session.restore_transcript(events)

        self._restore_working_state(session)

    def _restore_working_state(
        self,
        session: AgentSession,
    ) -> None:
        """Re-adopt anchor, checkpoint and evidence from disk.

        Without this a restored session came back with its transcript
        but an empty ``[TASK ANCHOR]``, ``[CHECKPOINT]`` and
        ``[EVIDENCE]``, and the next ``continue_run`` started a new
        task id that no longer matched the restored conversation.
        """

        if self._state_store is None:
            return

        session_id = session.session_id

        try:
            anchor = self._state_store.anchor(session_id)
            checkpoint = self._state_store.checkpoint(session_id)
            evidence = self._state_store.evidence(session_id)
        except Exception:
            return

        controller = session.runtime.agent.controller

        if anchor is not None:
            session.runtime.agent.orchestrator.restore_working_state(
                anchor=anchor,
                evidence=evidence,
            )

        if checkpoint is not None and controller is not None:
            try:
                controller.adopt_checkpoint(checkpoint)
            except Exception:
                pass

    def _remember_working_state(
        self,
        session: AgentSession,
    ) -> None:
        """Persist the working state alongside the transcript."""

        if self._state_store is None:
            return

        agent = session.runtime.agent

        try:
            orchestrator = agent.orchestrator
        except Exception:
            return

        # Each part is independent: reading task_state raises while
        # an agent has no plan yet, and that must not cost us the
        # anchor and evidence.
        parts: dict[str, Any] = {}

        for name, read in (
            ("anchor", lambda: orchestrator.task_anchor),
            ("evidence", lambda: orchestrator.evidence_store),
            ("task_state", lambda: orchestrator.task_state),
        ):
            try:
                parts[name] = read()
            except Exception as exc:
                session.note_state_persist_failure(exc)

        try:
            self._state_store.save(
                session.session_id,
                **parts,
            )
        except Exception as exc:
            session.note_state_persist_failure(exc)

    async def rename_session(
        self,
        session_id: str,
        title: str,
    ) -> str:
        """Rename a session and persist the new title."""

        session = await self.get_session(session_id)

        if session is None:
            raise KeyError(session_id)

        cleaned = session.rename(title)

        self._remember(session)

        return cleaned

    async def close_session(
        self,
        session_id: str,
    ) -> bool:
        session = self._sessions.pop(session_id, None)

        if session is None:
            self._pending.pop(session_id, None)

            self._forget(session_id)

            return False

        await session.close()

        self._forget(session_id)

        return True

    def _forget(self, session_id: str) -> None:
        """Erase a deleted session's record and transcript."""

        if self._store is not None:
            try:
                self._store.remove(session_id)
            except Exception:
                pass

        if self._event_store is not None:
            try:
                self._event_store.forget(session_id)
            except Exception:
                pass

        if self._state_store is not None:
            try:
                self._state_store.forget(session_id)
            except Exception:
                pass

    def list_sessions(self) -> list[AgentSession]:
        return list(self._sessions.values())

    def list_records(self) -> list[SessionRecord]:
        """Every known session, live or only recorded."""

        records: list[SessionRecord] = [
            SessionRecord(
                session_id=session.session_id,
                working_directory=str(session.working_directory),
                created_at=session.created_at,
                metadata=dict(session.metadata),
                title=session.display_title,
            )
            for session in self._sessions.values()
        ]

        known = {record.session_id for record in records}

        records.extend(
            record
            for record in self._pending.values()
            if record.session_id not in known
        )

        records.sort(key=lambda record: record.created_at)

        return records

    async def close_all(self) -> None:
        for session_id in list(self._sessions):
            session = self._sessions.get(session_id)

            if session is None:
                continue

            await session.close()

        self._sessions.clear()

        # Recorded sessions and their transcripts stay on disk:
        # shutting the server down must not delete anything, that is
        # exactly what has to survive a restart.

    def __len__(self) -> int:
        return len(self._sessions)

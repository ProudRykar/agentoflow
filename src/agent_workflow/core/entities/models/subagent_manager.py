from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import replace
from typing import Protocol

from agent_workflow.core.entities.models.agent import Agent, EventCallback
from agent_workflow.core.entities.models.agent_trace import AgentEvent, ToolStarted
from agent_workflow.core.entities.models.model_router import ModelRouter
from agent_workflow.core.entities.models.subagent import (
    AgentRun,
    SubagentPower,
    SubagentResult,
    SubagentStatus,
    SubagentTask,
)
from agent_workflow.core.entities.models.tool import ToolContext
from agent_workflow.core.infrastructure.config import SubagentConfig
from agent_workflow.core.infrastructure.llm.model_runtime import ModelRuntimeManager


class AgentFactory(Protocol):
    def create(
        self,
        *,
        run: AgentRun,
        task: SubagentTask,
        model: str,
    ) -> Agent:
        ...


class SubagentManager:
    def __init__(
        self,
        agent_factory: AgentFactory,
        model_router: ModelRouter,
        config: SubagentConfig,
        model_runtime: ModelRuntimeManager,
    ) -> None:
        self._agent_factory = agent_factory
        self._model_router = model_router
        self._config = config
        self._model_runtime = model_runtime

    async def run(
        self,
        task: SubagentTask,
        *,
        parent_run_id: str,
        parent_context: ToolContext,
        on_event: EventCallback | None = None,
    ) -> SubagentResult:
        effective_task = self._apply_defaults(task)

        model, result = await self._execute_once(
            effective_task,
            parent_run_id=parent_run_id,
            parent_context=parent_context,
            on_event=on_event,
        )

        if (
            self._config.escalation
            and result.status
            in (
                SubagentStatus.FAILED,
                SubagentStatus.TIMEOUT,
            )
        ):
            escalated = self._escalated_task(
                effective_task,
                model,
            )

            if escalated is not None:
                try:
                    preview = self._model_router.select(
                        escalated,
                    )
                except Exception:
                    preview = model

                # A higher tier resolving to the same model
                # means there is nothing stronger to try.
                if preview != model:
                    _, result = await self._execute_once(
                        escalated,
                        parent_run_id=parent_run_id,
                        parent_context=parent_context,
                        on_event=on_event,
                    )

        return result

    def _escalated_task(
        self,
        task: SubagentTask,
        model: str,
    ) -> SubagentTask | None:
        """One step up the power ladder, at most once."""

        base: SubagentPower | None = None

        power_for = getattr(
            self._model_router,
            "power_for",
            None,
        )

        if callable(power_for):
            try:
                base = power_for(model)
            except Exception:
                base = None

        if base is None:
            base = task.profile.power

        next_power = {
            SubagentPower.LOW: SubagentPower.MEDIUM,
            SubagentPower.MEDIUM: SubagentPower.HIGH,
        }.get(base)

        if next_power is None:
            return None

        return replace(
            task,
            profile=replace(
                task.profile,
                power=next_power,
            ),
        )

    async def _execute_once(
        self,
        task: SubagentTask,
        *,
        parent_run_id: str,
        parent_context: ToolContext,
        on_event: EventCallback | None = None,
    ) -> tuple[str, SubagentResult]:
        started_at = time.monotonic()
        run_id = self._new_run_id()

        effective_task = self._apply_defaults(task)

        model = self._model_router.select(
            effective_task,
        )

        run = AgentRun(
            run_id=run_id,
            parent_run_id=parent_run_id,
            agent_id=f"subagent:{effective_task.role}",
            role=effective_task.role,
            model=model,
        )

        context = self._build_context(
            parent_context=parent_context,
            task=effective_task,
            run_id=run_id,
            parent_run_id=parent_run_id,
        )

        agent = self._agent_factory.create(
            run=run,
            task=effective_task,
            model=model,
        )

        prompt = self._build_prompt(
            effective_task,
        )

        tool_calls = 0
        iterations = 0

        async def handle_event(
            event: AgentEvent,
        ) -> None:
            nonlocal tool_calls
            nonlocal iterations

            event_iteration = getattr(
                event,
                "iteration",
                None,
            )

            if event_iteration is not None:
                iterations = max(
                    iterations,
                    event_iteration,
                )

            if isinstance(event, ToolStarted):
                tool_calls += 1

            if on_event is not None:
                await on_event(event)

        timeout = effective_task.timeout_seconds

        try:
            async with self._model_runtime.use(model):
                if timeout is None:
                    output = await agent.run(
                        prompt,
                        context=context,
                        on_event=handle_event,
                    )
                else:
                    output = await asyncio.wait_for(
                        agent.run(
                            prompt,
                            context=context,
                            on_event=handle_event,
                        ),
                        timeout=timeout,
                    )

        except asyncio.TimeoutError:
            return (
                model,
                self._build_result(
                    run_id=run_id,
                    parent_run_id=parent_run_id,
                    status=SubagentStatus.TIMEOUT,
                    started_at=started_at,
                    iterations=iterations,
                    tool_calls=tool_calls,
                    error=(
                        "Subagent timed out after "
                        f"{timeout} seconds"
                    ),
                ),
            )

        except asyncio.CancelledError:
            raise

        except Exception as exc:
            return (
                model,
                self._build_result(
                    run_id=run_id,
                    parent_run_id=parent_run_id,
                    status=SubagentStatus.FAILED,
                    started_at=started_at,
                    iterations=iterations,
                    tool_calls=tool_calls,
                    error=str(exc),
                ),
            )

        return (
            model,
            self._build_result(
                run_id=run_id,
                parent_run_id=parent_run_id,
                status=SubagentStatus.COMPLETED,
                started_at=started_at,
                iterations=iterations,
                tool_calls=tool_calls,
                output=output,
            ),
        )

    def _apply_defaults(
        self,
        task: SubagentTask,
    ) -> SubagentTask:
        return replace(
            task,
            max_iterations=(
                task.max_iterations
                if task.max_iterations is not None
                else self._config.max_iterations
            ),
            max_tool_calls=(
                task.max_tool_calls
                if task.max_tool_calls is not None
                else self._config.max_tool_calls
            ),
            timeout_seconds=(
                task.timeout_seconds
                if task.timeout_seconds is not None
                else self._config.timeout
            ),
        )

    @staticmethod
    def _build_context(
        *,
        parent_context: ToolContext,
        task: SubagentTask,
        run_id: str,
        parent_run_id: str,
    ) -> ToolContext:
        parent_permissions = (
            parent_context.permissions
            | parent_context.approved_permissions
        )

        if task.permissions:
            permissions = (
                parent_permissions
                & task.permissions
            )
        else:
            permissions = parent_permissions

        return replace(
            parent_context,
            permissions=frozenset(permissions),
            approved_permissions=frozenset(),
            run_id=run_id,
            parent_run_id=parent_run_id,
        )

    @staticmethod
    def _build_prompt(
        task: SubagentTask,
    ) -> str:
        parts = [
            f"Your role: {task.role}",
            "",
            "Your objective:",
            task.objective,
        ]

        if task.instructions:
            parts.extend(
                [
                    "",
                    "Additional instructions:",
                    task.instructions,
                ],
            )

        if task.context:
            parts.extend(
                [
                    "",
                    "Provided context:",
                    task.context,
                ],
            )

        return "\n".join(parts)

    @staticmethod
    def _build_result(
        *,
        run_id: str,
        parent_run_id: str,
        status: SubagentStatus,
        started_at: float,
        iterations: int,
        tool_calls: int,
        output: str | None = None,
        error: str | None = None,
    ) -> SubagentResult:
        summary = (
            SubagentManager._build_summary(output)
            if output is not None
            else None
        )

        return SubagentResult(
            run_id=run_id,
            parent_run_id=parent_run_id,
            status=status,
            summary=summary,
            output=output,
            iterations=iterations,
            tool_calls=tool_calls,
            input_tokens=None,
            output_tokens=None,
            duration_seconds=(
                time.monotonic() - started_at
            ),
            error=error,
        )

    @staticmethod
    def _new_run_id() -> str:
        return uuid.uuid4().hex

    @staticmethod
    def _build_summary(
        output: str,
    ) -> str:
        output = output.strip()

        if len(output) <= 500:
            return output

        return output[:497] + "..."
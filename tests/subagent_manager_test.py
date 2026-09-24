from pathlib import Path
from typing import Any

import pytest

from core.application.subagents import (
    SubagentAgentFactory,
    options_by_model,
)
from core.context.context_assembler import ContextAssembler
from core.context.context_controller import ContextController
from core.entities.models.agent import Agent
from core.entities.models.llm import LLMResponse
from core.entities.models.llm_client import LLMClient
from core.entities.models.model_catalog import (
    ModelAttributes,
    ModelCapability,
    ModelCatalog,
    ModelProfile,
    ModelRequirements,
    ModelSpeed,
)
from core.entities.models.subagent import (
    AgentRun,
    SubagentPower,
    SubagentStatus,
    SubagentTask,
    TaskProfile,
)
from core.entities.models.subagent_manager import SubagentManager
from core.entities.models.tool import ToolContext
from core.entities.models.tool_executor import ToolExecutor
from core.entities.models.tool_registry import ToolRegistry
from core.infrastructure.config import SubagentConfig
from core.infrastructure.llm.model_runtime import (
    ModelRuntimeManager,
)


class DoneLLM(LLMClient):
    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: tuple[Any, ...] = (),
    ) -> LLMResponse:
        return LLMResponse(
            content="sub-done",
            tool_calls=(),
            raw={},
        )


class FailingLLM(LLMClient):
    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: tuple[Any, ...] = (),
    ) -> LLMResponse:
        raise RuntimeError("boom")


class ScriptedFactory:
    """Fails the first agent, succeeds afterwards."""

    def __init__(self) -> None:
        self.models: list[str] = []
        self.calls = 0

    def create(
        self,
        *,
        run: AgentRun,
        task: SubagentTask,
        model: str,
    ) -> Agent:
        self.models.append(model)
        self.calls += 1

        llm: LLMClient = (
            FailingLLM() if self.calls == 1 else DoneLLM()
        )

        registry = ToolRegistry()

        return Agent(
            llm=llm,
            registry=registry,
            executor=ToolExecutor(registry),
            run_id=run.run_id,
            parent_run_id=run.parent_run_id,
            agent_id=run.agent_id,
            role=run.role,
            model=model,
            max_iterations=2,
        )


class TierRouter:
    def __init__(self) -> None:
        self.seen: list[SubagentPower] = []

    def select(self, task: SubagentTask) -> str:
        self.seen.append(task.profile.power)

        return {
            SubagentPower.LOW: "low-model",
            SubagentPower.MEDIUM: "med-model",
            SubagentPower.HIGH: "high-model",
            SubagentPower.AUTO: "low-model",
        }[task.profile.power]

    def power_for(self, model: str) -> SubagentPower | None:
        return {
            "low-model": SubagentPower.LOW,
            "med-model": SubagentPower.MEDIUM,
            "high-model": SubagentPower.HIGH,
        }.get(model)


class NoopRuntimeClient:
    def __init__(self, model: str) -> None:
        self._model = model

    @property
    def model(self) -> str:
        return self._model

    async def load(self) -> None:
        return None

    async def unload(self) -> None:
        return None


def _manager(
    factory: ScriptedFactory,
    router: TierRouter,
    escalation: bool,
) -> SubagentManager:
    return SubagentManager(
        agent_factory=factory,
        model_router=router,  # type: ignore[arg-type]
        config=SubagentConfig(
            max_iterations=2,
            max_tool_calls=2,
            timeout=30.0,
            escalation=escalation,
        ),
        model_runtime=ModelRuntimeManager(
            lambda name: NoopRuntimeClient(name),
        ),
    )


def _context(tmp_path: Path) -> ToolContext:
    return ToolContext(
        working_directory=tmp_path,
        environment={},
        allowed_path=(tmp_path,),
        permissions=frozenset(),
    )


def _task() -> SubagentTask:
    return SubagentTask(
        role="researcher",
        objective="find things",
    )


@pytest.mark.asyncio
async def test_no_escalation_single_attempt(
    tmp_path: Path,
) -> None:
    factory = ScriptedFactory()
    manager = _manager(factory, TierRouter(), escalation=False)

    result = await manager.run(
        _task(),
        parent_run_id="parent",
        parent_context=_context(tmp_path),
    )

    assert result.status is SubagentStatus.FAILED
    assert factory.calls == 1
    assert factory.models == ["low-model"]


@pytest.mark.asyncio
async def test_escalation_retries_one_tier_up(
    tmp_path: Path,
) -> None:
    factory = ScriptedFactory()
    manager = _manager(factory, TierRouter(), escalation=True)

    result = await manager.run(
        _task(),
        parent_run_id="parent",
        parent_context=_context(tmp_path),
    )

    assert result.status is SubagentStatus.COMPLETED
    assert result.output == "sub-done"
    assert factory.models == ["low-model", "med-model"]


@pytest.mark.asyncio
async def test_escalation_stops_at_high(
    tmp_path: Path,
) -> None:
    factory = ScriptedFactory()
    router = TierRouter()
    manager = _manager(factory, router, escalation=True)

    task = SubagentTask(
        role="researcher",
        objective="hard work",
        profile=TaskProfile(
            capability=ModelCapability.GENERAL,
            power=SubagentPower.HIGH,
        ),
    )

    result = await manager.run(
        task,
        parent_run_id="parent",
        parent_context=_context(tmp_path),
    )

    assert result.status is SubagentStatus.FAILED
    assert factory.models == ["high-model"]


class SameModelRouter(TierRouter):
    """Always resolves to the same model."""

    def select(self, task: SubagentTask) -> str:
        self.seen.append(task.profile.power)

        return "only-model"

    def power_for(self, model: str) -> SubagentPower | None:
        return SubagentPower.MEDIUM


@pytest.mark.asyncio
async def test_escalation_skips_same_model_retry(
    tmp_path: Path,
) -> None:
    factory = ScriptedFactory()
    manager = _manager(
        factory,
        SameModelRouter(),  # type: ignore[arg-type]
        escalation=True,
    )

    result = await manager.run(
        _task(),
        parent_run_id="parent",
        parent_context=_context(tmp_path),
    )

    assert result.status is SubagentStatus.FAILED
    assert factory.calls == 1


def test_options_by_model_uses_catalog_ctx() -> None:
    catalog = ModelCatalog(
        models=(
            ModelProfile(
                name="e2b",
                description="",
                capabilities={},
                attributes=ModelAttributes(
                    speed=ModelSpeed.FAST,
                    quality=2,
                    resource_usage=1,
                ),
                requirements=ModelRequirements(
                    context_size=4096,
                ),
            ),
            ModelProfile(
                name="big",
                description="",
                capabilities={},
                attributes=ModelAttributes(
                    speed=ModelSpeed.MEDIUM,
                    quality=5,
                    resource_usage=4,
                ),
                requirements=ModelRequirements(),
            ),
        ),
    )

    options = options_by_model(catalog)

    assert options["e2b"].num_ctx == 4096
    assert options["e2b"].think is False
    assert "big" not in options


def test_thinking_flag_flows_into_options() -> None:
    catalog = ModelCatalog(
        models=(
            ModelProfile(
                name="thinker",
                description="",
                capabilities={},
                attributes=ModelAttributes(
                    speed=ModelSpeed.MEDIUM,
                    quality=5,
                    resource_usage=4,
                ),
                requirements=ModelRequirements(
                    context_size=8192,
                    thinking=True,
                ),
            ),
        ),
    )

    options = options_by_model(catalog)

    assert options["thinker"].think is True


def test_factory_builds_isolated_agent() -> None:
    parent = ToolRegistry()

    factory = SubagentAgentFactory(
        base_url="http://127.0.0.1:11434",
        timeout=10.0,
        parent_registry=parent,
        memory=None,
        assembler=ContextAssembler(),
        controller=ContextController(),
        options_by_model={},
    )

    agent = factory.create(
        run=AgentRun(
            run_id="run-1",
            parent_run_id="parent",
            agent_id="subagent:researcher",
            role="researcher",
            model="e2b",
        ),
        task=_task(),
        model="e2b",
    )

    assert agent.model == "e2b"
    assert agent.role == "researcher"
    assert agent.run_id == "run-1"

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from core.context.context_assembler import ContextAssembler
from core.context.context_controller import ContextController
from core.entities.models.agent import Agent
from core.entities.models.builtin.subagent import (
    create_subagent_tool,
)
from core.entities.models.memory_manager import MemoryManager
from core.entities.models.model_catalog import ModelCatalog
from core.entities.models.model_router import DefaultModelRouter
from core.entities.models.subagent import (
    AgentRun,
    SubagentPower,
    SubagentTask,
)
from core.entities.models.subagent_manager import SubagentManager
from core.entities.models.tool import Tool
from core.entities.models.tool_executor import ToolExecutor
from core.entities.models.tool_registry import (
    ToolRegistry,
    ToolRegistryError,
)
from core.infrastructure.config import SubagentConfig
from core.infrastructure.llm.model_runtime import (
    ModelRuntimeManager,
)
from core.infrastructure.model_catalog import ModelCatalogLoader
from core.infrastructure.ollama_client import (
    OllamaClient,
    OllamaOptions,
)


@dataclass(slots=True)
class SubagentStack:
    manager: SubagentManager
    tool: Tool
    router: DefaultModelRouter
    model_runtime: ModelRuntimeManager
    catalog: ModelCatalog


class SubagentAgentFactory:
    """Builds isolated agents for subagent runs.

    Each subagent gets its own LLM client (selected model with
    per-tier options) and a filtered tool registry, but shares
    the parent's memory, assembler, controller, and history.
    """

    def __init__(
        self,
        *,
        base_url: str,
        timeout: float,
        parent_registry: ToolRegistry,
        memory: MemoryManager | None,
        assembler: ContextAssembler,
        controller: ContextController,
        options_by_model: dict[str, OllamaOptions],
    ) -> None:
        self._base_url = base_url
        self._timeout = timeout
        self._parent_registry = parent_registry
        self._memory = memory
        self._assembler = assembler
        self._controller = controller
        self._options_by_model = options_by_model

    def create(
        self,
        *,
        run: AgentRun,
        task: SubagentTask,
        model: str,
    ) -> Agent:
        registry = ToolRegistry()

        for name in task.tools:
            try:
                registry.register(
                    self._parent_registry.get(name),
                )
            except ToolRegistryError:
                continue

        base_options = self._options_by_model.get(model)

        catalog_think = (
            base_options.think
            if base_options is not None
            else True
        )

        # Small tiers never think: weaker tool-calling plus a
        # 400-risk on models without thinking support.
        think = catalog_think and task.profile.power not in (
            SubagentPower.LOW,
            SubagentPower.MEDIUM,
        )

        if base_options is not None:
            options = OllamaOptions(
                num_ctx=base_options.num_ctx,
                num_predict=base_options.num_predict,
                num_gpu=base_options.num_gpu,
                keep_alive=base_options.keep_alive,
                think=think,
            )
        else:
            options = OllamaOptions(think=think)

        llm = OllamaClient(
            model=model,
            base_url=self._base_url,
            timeout=self._timeout,
            options=options,
        )

        return Agent(
            llm=llm,
            registry=registry,
            executor=ToolExecutor(registry),
            assembler=self._assembler,
            controller=self._controller,
            memory=self._memory,
            run_id=run.run_id,
            parent_run_id=run.parent_run_id,
            agent_id=run.agent_id,
            role=run.role,
            model=model,
        )


def main_model_context_size(
    catalog_path: Path,
    model_name: str,
) -> int | None:
    """Real model num_ctx for UI budget display, if known."""

    try:
        catalog = ModelCatalogLoader().load(catalog_path)
    except Exception:
        return None

    for model in catalog.models:
        if model.name == model_name:
            return model.requirements.context_size

    return None


def options_by_model(
    catalog: ModelCatalog,
) -> dict[str, OllamaOptions]:
    """num_ctx per model from the catalog requirements."""

    options: dict[str, OllamaOptions] = {}

    for model in catalog.models:
        requirements = model.requirements

        if (
            requirements.context_size is None
            and not requirements.thinking
        ):
            continue

        options[model.name] = OllamaOptions(
            num_ctx=requirements.context_size,
            think=requirements.thinking,
        )

    return options


def build_subagent_stack(
    *,
    catalog_path: Path,
    main_model: str,
    vram_budget_gb: float | None,
    single_model_mode: bool,
    subagent_config: SubagentConfig,
    base_url: str,
    timeout: float,
    parent_registry: ToolRegistry,
    memory: MemoryManager | None,
    assembler: ContextAssembler,
    controller: ContextController,
) -> SubagentStack:
    catalog = ModelCatalogLoader().load(catalog_path)

    model_runtime = ModelRuntimeManager(
        lambda name: OllamaClient(
            model=name,
            base_url=base_url,
            timeout=timeout,
        ),
    )

    router = DefaultModelRouter(
        catalog,
        vram_budget_gb=vram_budget_gb,
        main_model=main_model,
        single_model_mode=single_model_mode,
        resident_provider=lambda: model_runtime.loaded_model,
    )

    manager = SubagentManager(
        agent_factory=SubagentAgentFactory(
            base_url=base_url,
            timeout=timeout,
            parent_registry=parent_registry,
            memory=memory,
            assembler=assembler,
            controller=controller,
            options_by_model=options_by_model(catalog),
        ),
        model_router=router,
        config=subagent_config,
        model_runtime=model_runtime,
    )

    return SubagentStack(
        manager=manager,
        tool=create_subagent_tool(manager),
        router=router,
        model_runtime=model_runtime,
        catalog=catalog,
    )

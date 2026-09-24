from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import tomllib


@dataclass(slots=True, frozen=True)
class AgentConfig:
    max_iterations: int = 10


@dataclass(slots=True, frozen=True)
class LLMConfig:
    provider: str = "ollama"
    model: str = "gemma4:12b"
    timeout: float = 600.0


@dataclass(slots=True, frozen=True)
class SubagentConfig:
    max_iterations: int = 5
    max_tool_calls: int = 10
    timeout: float = 600.0
    escalation: bool = False


@dataclass(slots=True, frozen=True)
class ModelsConfig:
    catalog: str = "models.toml"
    vram_budget_gb: float | None = None
    single_model_mode: bool = False


@dataclass(slots=True, frozen=True)
class ContextConfig:
    max_messages: int | None = 4
    token_estimation_divisor: int = 4


@dataclass(slots=True, frozen=True)
class MemoryConfig:
    database: str = "memory.db"


@dataclass(slots=True, frozen=True)
class Config:
    agent: AgentConfig
    llm: LLMConfig
    subagent: SubagentConfig
    context: ContextConfig
    memory: MemoryConfig
    models: ModelsConfig


class ConfigLoader:
    def load(
        self,
        path: Path,
    ) -> Config:
        with path.open("rb") as file:
            data = tomllib.load(file)

        agent_data = data.get(
            "agent",
            {},
        )

        llm_data = data.get(
            "llm",
            {},
        )

        subagent_data = data.get(
            "subagent",
            {},
        )

        context_data = data.get(
            "context",
            {},
        )

        memory_data = data.get(
            "memory",
            {},
        )

        models_data = data.get(
            "models",
            {},
        )

        return Config(
            agent=AgentConfig(
                max_iterations=agent_data.get(
                    "max_iterations",
                    10,
                ),
            ),
            llm=LLMConfig(
                provider=llm_data.get(
                    "provider",
                    "ollama",
                ),
                model=llm_data.get(
                    "model",
                    "gemma4:12b",
                ),
                timeout=llm_data.get(
                    "timeout",
                    600.0,
                ),
            ),
            subagent=SubagentConfig(
                max_iterations=subagent_data.get(
                    "max_iterations",
                    25,
                ),
                max_tool_calls=subagent_data.get(
                    "max_tool_calls",
                    10,
                ),
                timeout=subagent_data.get(
                    "timeout",
                    600.0,
                ),
                escalation=subagent_data.get(
                    "escalation",
                    False,
                ),
            ),
            context=ContextConfig(
                max_messages=context_data.get(
                    "max_messages",
                    4,
                ),
                token_estimation_divisor=context_data.get(
                    "token_estimation_divisor",
                    4,
                ),
            ),
            memory=MemoryConfig(
                database=memory_data.get(
                    "database",
                    "memory.db",
                ),
            ),
            models=ModelsConfig(
                catalog=models_data.get(
                    "catalog",
                    "models.toml",
                ),
                vram_budget_gb=models_data.get(
                    "vram_budget_gb",
                ),
                single_model_mode=models_data.get(
                    "single_model_mode",
                    False,
                ),
            ),
        )
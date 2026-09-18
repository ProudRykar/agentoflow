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


@dataclass(slots=True, frozen=True)
class ContextConfig:
    max_messages: int | None = 4


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
            ),
            context=ContextConfig(
                max_messages=context_data.get(
                    "max_messages",
                    4,
                ),
            ),
            memory=MemoryConfig(
                database=memory_data.get(
                    "database",
                    "memory.db",
                ),
            ),
        )
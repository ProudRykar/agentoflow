from abc import ABC, abstractmethod
from typing import Any

from core.entities.models.llm import LLMResponse
from core.entities.models.tool_definition import ToolDefinition


class LLMClient(ABC):
    @abstractmethod
    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: tuple[ToolDefinition, ...] = (),
    ) -> LLMResponse:
        ...
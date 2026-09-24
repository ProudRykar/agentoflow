from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class ModelCapability(StrEnum):
    GENERAL = "general"
    CODING = "coding"
    REASONING = "reasoning"
    RESEARCH = "research"
    LONG_CONTEXT = "long_context"


class ModelSpeed(StrEnum):
    FAST = "fast"
    MEDIUM = "medium"
    SLOW = "slow"


@dataclass(slots=True, frozen=True)
class ModelAttributes:
    speed: ModelSpeed = ModelSpeed.MEDIUM
    quality: int = 3
    resource_usage: int = 3


@dataclass(slots=True, frozen=True)
class ModelRequirements:
    context_size: int | None = None
    vram_gb: float | None = None
    ram_gb: float | None = None
    thinking: bool = False


@dataclass(slots=True, frozen=True)
class ModelProfile:
    name: str
    description: str

    capabilities: dict[ModelCapability, int]

    attributes: ModelAttributes
    requirements: ModelRequirements


@dataclass(slots=True, frozen=True)
class ModelCatalog:
    models: tuple[ModelProfile, ...]

    def get(
        self,
        name: str,
    ) -> ModelProfile:
        for model in self.models:
            if model.name == name:
                return model

        raise KeyError(
            f"Model not found in catalog: {name!r}"
        )

    def has(
        self,
        name: str,
    ) -> bool:
        return any(
            model.name == name
            for model in self.models
        )

    def describe(self) -> str:
        """
        Human/LLM-readable description of available models.

        This deliberately contains only information from the
        user-maintained catalog. The LLM cannot invent model
        capabilities here.
        """
        if not self.models:
            return "No subagent models are configured."

        parts: list[str] = [
            "Available subagent models:",
            "",
        ]

        for model in self.models:
            parts.append(
                f"Model: {model.name}"
            )

            if model.description:
                parts.extend([
                    "Description:",
                    model.description.strip(),
                ])

            capabilities = ", ".join(
                f"{capability.value}={score}/5"
                for capability, score
                in model.capabilities.items()
            )

            if capabilities:
                parts.append(
                    f"Capabilities: {capabilities}"
                )

            parts.append(
                "Speed: "
                f"{model.attributes.speed.value}"
            )

            parts.append(
                "Quality: "
                f"{model.attributes.quality}/5"
            )

            parts.append(
                "Resource usage: "
                f"{model.attributes.resource_usage}/5 "
                "(1-2=low tier, 3=medium, 4-5=heavy; "
                "high power picks the best of 3+, heavy "
                "models may force a VRAM model swap)"
            )

            if model.requirements.vram_gb is not None:
                parts.append(
                    "VRAM: "
                    f"{model.requirements.vram_gb:g} GB"
                )

            if model.requirements.context_size is not None:
                parts.append(
                    "Context: "
                    f"{model.requirements.context_size:,} tokens"
                )

            parts.append("")

        return "\n".join(parts).strip()
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from agent_workflow.core.entities.models.model_catalog import (
    ModelCatalog,
    ModelProfile,
    ModelSpeed,
)
from agent_workflow.core.entities.models.subagent import (
    SubagentPower,
    SubagentTask,
    TaskComplexity,
)

# Compute tier chosen by the caller maps onto resource_usage.
# LOW/MEDIUM are disjoint bands. HIGH means "best available
# regardless of cost": usage 3 and up, so the strong main model
# (usually already resident, no swap) can win it.
POWER_USAGE_BANDS: dict[SubagentPower, tuple[int, int]] = {
    SubagentPower.LOW: (1, 2),
    SubagentPower.MEDIUM: (3, 3),
    SubagentPower.HIGH: (3, 5),
}

# Usage at or above this always risks a VRAM swap: never picked
# implicitly for non-complex work.
HIGH_SWAP_THRESHOLD = 4

# Bonus for the already resident model: a VRAM swap costs
# more than a small quality gap.
RESIDENT_AFFINITY_BONUS = 8.0


class ModelRoutingError(RuntimeError):
    pass


class ModelRouter(Protocol):
    def select(
        self,
        task: SubagentTask,
    ) -> str:
        ...


@dataclass(slots=True, frozen=True)
class ModelCandidate:
    model: ModelProfile
    score: float


class DefaultModelRouter:
    """
    Selects a model using the user-maintained model catalog.

    The router never assumes that a particular model is good
    at coding, reasoning, research, etc. All such information
    comes from ModelCatalog.
    """

    def __init__(
        self,
        catalog: ModelCatalog,
        *,
        vram_budget_gb: float | None = None,
        main_model: str | None = None,
        single_model_mode: bool = False,
        resident_provider: Callable[[], str | None] | None = None,
    ) -> None:
        self._catalog = catalog
        self._vram_budget_gb = vram_budget_gb
        self._main_model = main_model
        self._single_model_mode = single_model_mode
        self._resident_provider = resident_provider

    @property
    def vram_budget_gb(self) -> float | None:
        return self._vram_budget_gb

    def power_for(
        self,
        model_name: str,
    ) -> SubagentPower | None:
        """Compute tier of a catalog model, if known."""

        for power, (low, high) in POWER_USAGE_BANDS.items():
            for model in self._catalog.models:
                if model.name != model_name:
                    continue

                usage = model.attributes.resource_usage

                if low <= usage <= high:
                    return power

        return None

    def select(
        self,
        task: SubagentTask,
    ) -> str:
        if task.model != "auto":
            return task.model

        if self._single_model_mode and self._main_model:
            return self._main_model

        candidate = self._select_best(
            task,
        )

        if candidate is None:
            raise ModelRoutingError(
                "No suitable model found for task"
            )

        return candidate.model.name

    def rank(
        self,
        task: SubagentTask,
    ) -> tuple[ModelCandidate, ...]:
        candidates = [
            ModelCandidate(
                model=model,
                score=self._score(
                    model,
                    task,
                ),
            )
            for model in self._eligible_models(task)
        ]

        candidates.sort(
            key=lambda candidate: (
                candidate.score,
                candidate.model.attributes.quality,
            ),
            reverse=True,
        )

        return tuple(candidates)

    def _select_best(
        self,
        task: SubagentTask,
    ) -> ModelCandidate | None:
        candidates = self.rank(task)

        if not candidates:
            return None

        best = candidates[0]

        if best.score <= 0:
            return None

        return best

    def _score(
        self,
        model: ModelProfile,
        task: SubagentTask,
    ) -> float:
        score = 0.0

        profile = task.profile

        capability_score = model.capabilities.get(
            profile.capability,
            0,
        )

        if capability_score <= 0:
            return 0.0

        score += capability_score * 10

        score += self._complexity_score(
            model,
            profile.complexity,
        )

        score += (
            model.attributes.quality * 1.5
        )

        # Nearly-tied candidates resolve towards cheaper models.
        score += (
            6 - model.attributes.resource_usage
        ) * 0.1

        score += self._speed_score(
            model,
            profile.complexity,
        )

        score += self._context_score(
            model,
            profile.min_context_tokens,
        )

        resident = self._resident_model()

        if (
            resident is not None
            and model.name == resident
        ):
            score += RESIDENT_AFFINITY_BONUS

        return score

    def _eligible_models(
        self,
        task: SubagentTask,
    ) -> tuple[ModelProfile, ...]:
        profile = task.profile

        band = POWER_USAGE_BANDS.get(profile.power)

        eligible: list[ModelProfile] = []

        for model in self._catalog.models:
            usage = model.attributes.resource_usage

            if band is not None and not (
                band[0] <= usage <= band[1]
            ):
                continue

            if not self._fits_vram(model):
                continue

            # A heavy-model swap is the most expensive operation:
            # never pick it implicitly for non-complex work.
            if (
                profile.power is SubagentPower.AUTO
                and profile.complexity
                is not TaskComplexity.COMPLEX
                and usage >= HIGH_SWAP_THRESHOLD
            ):
                continue

            eligible.append(model)

        return tuple(eligible)

    def _fits_vram(
        self,
        model: ModelProfile,
    ) -> bool:
        if self._vram_budget_gb is None:
            return True

        required = model.requirements.vram_gb

        if required is None:
            return True

        return required <= self._vram_budget_gb

    def _resident_model(self) -> str | None:
        if self._resident_provider is None:
            return None

        try:
            return self._resident_provider()
        except Exception:
            return None

    @staticmethod
    def _complexity_score(
        model: ModelProfile,
        complexity: TaskComplexity,
    ) -> float:
        quality = model.attributes.quality

        match complexity:
            case TaskComplexity.SIMPLE:
                return (6 - quality) * 2.0

            case TaskComplexity.MEDIUM:
                return quality * 0.5

            case TaskComplexity.COMPLEX:
                return quality * 2.0

    @staticmethod
    def _speed_score(
        model: ModelProfile,
        complexity: TaskComplexity,
    ) -> float:
        speed_scores = {
            ModelSpeed.FAST: 3.0,
            ModelSpeed.MEDIUM: 2.0,
            ModelSpeed.SLOW: 1.0,
        }

        score = speed_scores[
            model.attributes.speed
        ]

        if complexity == TaskComplexity.SIMPLE:
            return score * 2.0

        if complexity == TaskComplexity.COMPLEX:
            return score * 0.5

        return score

    @staticmethod
    def _context_score(
        model: ModelProfile,
        min_context_tokens: int | None,
    ) -> float:
        if min_context_tokens is None:
            return 0.0

        context_size = (
            model.requirements.context_size
        )

        if context_size is None:
            return 0.0

        if context_size < min_context_tokens:
            return 0.0

        return min(
            context_size / min_context_tokens,
            3.0,
        )
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from core.entities.models.model_catalog import (
    ModelCatalog,
    ModelProfile,
    ModelSpeed,
)
from core.entities.models.subagent import (
    SubagentTask,
    TaskComplexity,
)


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
    ) -> None:
        self._catalog = catalog

    def select(
        self,
        task: SubagentTask,
    ) -> str:
        if task.model != "auto":
            return task.model

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
            for model in self._catalog.models
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

        score += self._speed_score(
            model,
            profile.complexity,
        )

        score += self._context_score(
            model,
            profile.min_context_tokens,
        )

        return score

    @staticmethod
    def _complexity_score(
        model: ModelProfile,
        complexity: TaskComplexity,
    ) -> float:
        quality = model.attributes.quality

        match complexity:
            case TaskComplexity.SIMPLE:
                return (6 - quality) * 1.0

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
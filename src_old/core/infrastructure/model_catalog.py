from __future__ import annotations

import tomllib
from pathlib import Path

from core.entities.models.model_catalog import (
    ModelAttributes,
    ModelCapability,
    ModelCatalog,
    ModelProfile,
    ModelRequirements,
    ModelSpeed,
)


class ModelCatalogError(RuntimeError):
    pass


class ModelCatalogLoader:
    def load(
        self,
        path: Path,
    ) -> ModelCatalog:
        if not path.exists():
            raise ModelCatalogError(
                f"Model catalog does not exist: {path}"
            )

        try:
            with path.open("rb") as file:
                data = tomllib.load(file)
        except (OSError, tomllib.TOMLDecodeError) as exc:
            raise ModelCatalogError(
                f"Failed to load model catalog: {path}"
            ) from exc

        models_data = data.get("models", [])

        if not isinstance(models_data, list):
            raise ModelCatalogError(
                "'models' must be an array of tables"
            )

        models: list[ModelProfile] = []

        for index, raw_model in enumerate(
            models_data,
            start=1,
        ):
            try:
                models.append(
                    self._parse_model(raw_model)
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ModelCatalogError(
                    f"Invalid model definition at "
                    f"models[{index}]"
                ) from exc

        self._validate_duplicates(models)

        return ModelCatalog(
            models=tuple(models),
        )

    @staticmethod
    def _parse_model(
        data: object,
    ) -> ModelProfile:
        if not isinstance(data, dict):
            raise TypeError(
                "Model definition must be a table"
            )

        name = data["name"]

        if not isinstance(name, str) or not name.strip():
            raise ValueError(
                "Model name must be a non-empty string"
            )

        description = data.get(
            "description",
            "",
        )

        if not isinstance(description, str):
            raise TypeError(
                "Model description must be a string"
            )

        capabilities_data = data.get(
            "capabilities",
            {},
        )

        if not isinstance(
            capabilities_data,
            dict,
        ):
            raise TypeError(
                "Model capabilities must be a table"
            )

        capabilities: dict[
            ModelCapability,
            int,
        ] = {}

        for raw_capability, raw_score in (
            capabilities_data.items()
        ):
            capability = ModelCapability(
                raw_capability,
            )

            score = int(raw_score)

            if not 0 <= score <= 5:
                raise ValueError(
                    "Capability score must be "
                    "between 0 and 5"
                )

            capabilities[capability] = score

        attributes_data = data.get(
            "attributes",
            {},
        )

        if not isinstance(
            attributes_data,
            dict,
        ):
            raise TypeError(
                "Model attributes must be a table"
            )

        speed = ModelSpeed(
            attributes_data.get(
                "speed",
                "medium",
            )
        )

        quality = int(
            attributes_data.get(
                "quality",
                3,
            )
        )

        resource_usage = int(
            attributes_data.get(
                "resource_usage",
                3,
            )
        )

        if not 1 <= quality <= 5:
            raise ValueError(
                "Quality must be between 1 and 5"
            )

        if not 1 <= resource_usage <= 5:
            raise ValueError(
                "Resource usage must be between 1 and 5"
            )

        requirements_data = data.get(
            "requirements",
            {},
        )

        if not isinstance(
            requirements_data,
            dict,
        ):
            raise TypeError(
                "Model requirements must be a table"
            )

        context_size = requirements_data.get(
            "context_size",
        )

        if context_size is not None:
            context_size = int(context_size)

        vram_gb = requirements_data.get(
            "vram_gb",
        )

        if vram_gb is not None:
            vram_gb = float(vram_gb)

        ram_gb = requirements_data.get(
            "ram_gb",
        )

        if ram_gb is not None:
            ram_gb = float(ram_gb)

        thinking = requirements_data.get(
            "thinking",
            False,
        )

        if not isinstance(thinking, bool):
            raise ValueError(
                "Thinking must be a boolean"
            )

        return ModelProfile(
            name=name,
            description=description,
            capabilities=capabilities,
            attributes=ModelAttributes(
                speed=speed,
                quality=quality,
                resource_usage=resource_usage,
            ),
            requirements=ModelRequirements(
                context_size=context_size,
                vram_gb=vram_gb,
                ram_gb=ram_gb,
                thinking=thinking,
            ),
        )

    @staticmethod
    def _validate_duplicates(
        models: list[ModelProfile],
    ) -> None:
        names = [model.name for model in models]

        duplicates = {
            name
            for name in names
            if names.count(name) > 1
        }

        if duplicates:
            names_text = ", ".join(
                sorted(duplicates)
            )

            raise ModelCatalogError(
                "Duplicate model names: "
                f"{names_text}"
            )
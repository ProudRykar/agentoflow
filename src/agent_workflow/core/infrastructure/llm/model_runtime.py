from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Protocol


class ModelRuntimeClient(Protocol):
    @property
    def model(self) -> str:
        ...

    async def load(self) -> None:
        ...

    async def unload(self) -> None:
        ...


class ModelRuntimeError(RuntimeError):
    pass


class ModelRuntimeManager:
    def __init__(
        self,
        client_factory: Callable[[str], ModelRuntimeClient],
        initial_model: str | None = None,
    ) -> None:
        self._client_factory = client_factory
        self._clients: dict[str, ModelRuntimeClient] = {}
        self._loaded_model = initial_model
        self._lock = asyncio.Lock()

    @property
    def loaded_model(self) -> str | None:
        return self._loaded_model

    def _get_client(
        self,
        model: str,
    ) -> ModelRuntimeClient:
        client = self._clients.get(model)

        if client is None:
            client = self._client_factory(model)
            self._clients[model] = client

        return client

    async def load(
        self,
        model: str,
    ) -> None:
        async with self._lock:
            await self._load_locked(model)

    async def unload(
        self,
        model: str,
    ) -> None:
        async with self._lock:
            await self._unload_locked(model)

    async def switch(
        self,
        model: str,
    ) -> None:
        async with self._lock:
            await self._switch_locked(model)

    async def release_current(self) -> None:
        async with self._lock:
            if self._loaded_model is None:
                return

            await self._unload_locked(
                self._loaded_model,
            )

    @asynccontextmanager
    async def use(
        self,
        model: str,
    ) -> AsyncIterator[None]:
        """
        Temporarily make a model the active resident model.

        The previously active model is restored when the context exits,
        including when the wrapped operation raises an exception.
        """

        async with self._lock:
            previous_model = self._loaded_model

            await self._switch_locked(model)

            try:
                yield
            finally:
                await self._restore_locked(
                    previous_model,
                )

    async def _load_locked(
        self,
        model: str,
    ) -> None:
        if self._loaded_model == model:
            return

        if self._loaded_model is not None:
            await self._unload_locked(
                self._loaded_model,
            )

        client = self._get_client(model)

        try:
            await client.load()
        except Exception as exc:
            raise ModelRuntimeError(
                f"Failed to load model {model!r}: {exc}"
            ) from exc

        self._loaded_model = model

    async def _switch_locked(
        self,
        model: str,
    ) -> None:
        if self._loaded_model == model:
            return

        if self._loaded_model is not None:
            await self._unload_locked(
                self._loaded_model,
            )

        client = self._get_client(model)

        try:
            await client.load()
        except Exception as exc:
            raise ModelRuntimeError(
                f"Failed to unload model {model!r}: {exc}"
            ) from exc

        self._loaded_model = model

    async def _unload_locked(
        self,
        model: str,
    ) -> None:
        client = self._get_client(model)

        try:
            await client.unload()
        except Exception as exc:
            raise ModelRuntimeError(
                f"Failed to unload model {model!r}"
            ) from exc

        if self._loaded_model == model:
            self._loaded_model = None

    async def _restore_locked(
        self,
        model: str | None,
    ) -> None:
        if self._loaded_model is not None:
            await self._unload_locked(
                self._loaded_model,
            )

        if model is None:
            return

        client = self._get_client(model)

        try:
            await client.load()
        except Exception as exc:
            raise ModelRuntimeError(
                f"Failed to restore model {model!r}: {exc}"
            ) from exc

        self._loaded_model = model
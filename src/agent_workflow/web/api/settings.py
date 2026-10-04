from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from agent_workflow.core.application.settings_service import (
    EDITABLE_FILES,
    SettingsError,
    SettingsService,
    SettingsValidationError,
)
from agent_workflow.core.infrastructure.paths import (
    AgentWorkflowPaths,
)
from agent_workflow.web.schemas import (
    ModelProfileInfo,
    ModelWriteRequest,
    SettingsFileInfo,
    SettingsWriteRequest,
)


router = APIRouter()

# The UI edits redacted documents, so it never sees a stored
# secret. Returning them on request would defeat the redaction.
ALLOW_REVEAL = False


def service_of(request: Request) -> SettingsService:
    cached = getattr(
        request.app.state,
        "settings_service",
        None,
    )

    if isinstance(cached, SettingsService):
        return cached

    service = SettingsService(
        AgentWorkflowPaths()
    )

    request.app.state.settings_service = service

    return service


def _fail(exc: Exception) -> HTTPException:
    if isinstance(exc, SettingsValidationError):
        return HTTPException(
            status_code=422,
            detail=str(exc),
        )

    if isinstance(exc, SettingsError):
        return HTTPException(
            status_code=400,
            detail=str(exc),
        )

    return HTTPException(
        status_code=500,
        detail="settings operation failed",
    )


@router.get("", response_model=list[str])
async def list_files(
    request: Request,
) -> list[str]:
    return list(EDITABLE_FILES)


@router.get("/{name}", response_model=SettingsFileInfo)
async def read_settings(
    name: str,
    request: Request,
) -> SettingsFileInfo:
    service = service_of(request)

    try:
        document = service.read(name)

        text = service.raw_text(
            name,
            reveal_secrets=ALLOW_REVEAL,
        )
    except SettingsError as exc:
        raise _fail(exc) from exc

    return SettingsFileInfo(
        name=document.name,
        path=str(document.path),
        exists=document.exists,
        data=document.data,
        text=text,
    )


@router.put("/{name}", response_model=SettingsFileInfo)
async def write_settings(
    name: str,
    body: SettingsWriteRequest,
    request: Request,
) -> SettingsFileInfo:
    service = service_of(request)

    try:
        if body.text is not None:
            document = service.write_text(
                name,
                body.text,
                keep_secrets=body.keep_secrets,
            )
        elif body.data is not None:
            document = service.write(
                name,
                body.data,
                keep_secrets=body.keep_secrets,
            )
        else:
            raise SettingsValidationError(
                "Provide either 'data' or 'text'"
            )
    except SettingsError as exc:
        raise _fail(exc) from exc

    return SettingsFileInfo(
        name=document.name,
        path=str(document.path),
        exists=True,
        data=document.data,
        text=document.raw,
    )


# ======================================================================
# Model catalog
# ======================================================================


@router.get("/models/list", response_model=list[ModelProfileInfo])
async def list_models(
    request: Request,
) -> list[ModelProfileInfo]:
    service = service_of(request)

    try:
        models = service.models()
    except SettingsError as exc:
        raise _fail(exc) from exc

    result: list[ModelProfileInfo] = []

    for model in models:
        result.append(
            ModelProfileInfo(
                name=str(model.get("name", "")),
                description=str(model.get("description", "")),
                capabilities=dict(
                    model.get("capabilities", {})
                    if isinstance(model.get("capabilities"), dict)
                    else {}
                ),
                attributes=dict(
                    model.get("attributes", {})
                    if isinstance(model.get("attributes"), dict)
                    else {}
                ),
                requirements=dict(
                    model.get("requirements", {})
                    if isinstance(model.get("requirements"), dict)
                    else {}
                ),
            )
        )

    return result


@router.put("/models/entry", response_model=SettingsFileInfo)
async def upsert_model(
    body: ModelWriteRequest,
    request: Request,
) -> SettingsFileInfo:
    service = service_of(request)

    profile: dict[str, object] = {
        "name": body.name.strip(),
        "description": body.description,
    }

    if body.capabilities:
        profile["capabilities"] = {
            key: int(value)
            for key, value in body.capabilities.items()
        }

    if body.attributes:
        profile["attributes"] = dict(body.attributes)

    if body.requirements:
        profile["requirements"] = dict(body.requirements)

    try:
        document = service.add_model(profile)
    except SettingsError as exc:
        raise _fail(exc) from exc

    return SettingsFileInfo(
        name=document.name,
        path=str(document.path),
        exists=True,
        data=document.data,
        text=document.raw,
    )


@router.delete("/models/entry/{name}")
async def delete_model(
    name: str,
    request: Request,
) -> dict:
    service = service_of(request)

    try:
        service.remove_model(name)
    except SettingsError as exc:
        raise _fail(exc) from exc

    return {"name": name, "deleted": True}

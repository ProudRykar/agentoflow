from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from agent_workflow.core.application.skills_service import (
    SkillError,
    SkillsService,
    SkillValidationError,
)
from agent_workflow.web.schemas import (
    SkillDetail,
    SkillWriteRequest,
)


router = APIRouter(prefix="/library")


def service_of(request: Request) -> SkillsService:
    cached = getattr(
        request.app.state,
        "skills_library",
        None,
    )

    if isinstance(cached, SkillsService):
        return cached

    service = SkillsService()

    request.app.state.skills_library = service

    return service


def _fail(exc: Exception) -> HTTPException:
    if isinstance(exc, SkillValidationError):
        return HTTPException(status_code=422, detail=str(exc))

    if isinstance(exc, SkillError):
        if "not found" in str(exc).lower():
            return HTTPException(status_code=404, detail=str(exc))

        return HTTPException(status_code=400, detail=str(exc))

    return HTTPException(
        status_code=500,
        detail="skill operation failed",
    )


@router.get("", response_model=list[SkillDetail])
async def list_library(request: Request) -> list[SkillDetail]:
    service = service_of(request)

    return [
        SkillDetail(
            name=summary.name,
            description=summary.description,
            version=summary.version,
            instructions="",
            metadata=summary.metadata,
            path=summary.path,
        )
        for summary in service.list()
    ]


@router.get("/{name}", response_model=SkillDetail)
async def read_skill(
    name: str,
    request: Request,
) -> SkillDetail:
    service = service_of(request)

    try:
        document = service.get(name)
    except SkillError as exc:
        raise _fail(exc) from exc

    return SkillDetail(
        name=document.name,
        description=document.description,
        version=document.version,
        instructions=document.instructions,
        metadata=document.metadata,
        path=document.path,
    )


@router.post("", response_model=SkillDetail)
async def create_skill(
    body: SkillWriteRequest,
    request: Request,
) -> SkillDetail:
    service = service_of(request)

    try:
        document = service.create(
            body.name,
            description=body.description,
            instructions=body.instructions,
            version=body.version,
            metadata=body.metadata,
        )
    except SkillError as exc:
        raise _fail(exc) from exc

    return SkillDetail(
        name=document.name,
        description=document.description,
        version=document.version,
        instructions=document.instructions,
        metadata=document.metadata,
        path=document.path,
    )


@router.put("/{name}", response_model=SkillDetail)
async def update_skill(
    name: str,
    body: SkillWriteRequest,
    request: Request,
) -> SkillDetail:
    service = service_of(request)

    try:
        document = service.upsert(
            body.name or name,
            description=body.description,
            instructions=body.instructions,
            version=body.version,
            metadata=body.metadata,
        )
    except SkillError as exc:
        raise _fail(exc) from exc

    return SkillDetail(
        name=document.name,
        description=document.description,
        version=document.version,
        instructions=document.instructions,
        metadata=document.metadata,
        path=document.path,
    )


@router.delete("/{name}")
async def delete_skill(
    name: str,
    request: Request,
) -> dict:
    service = service_of(request)

    try:
        deleted = service.delete(name)
    except SkillError as exc:
        raise _fail(exc) from exc

    if not deleted:
        raise HTTPException(
            status_code=404,
            detail=f"Skill '{name}' does not exist",
        )

    return {"name": name, "deleted": True}

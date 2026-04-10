from fastapi import APIRouter, Depends, HTTPException, Query, Request

from vision_service.container import ServiceContainer
from vision_service.contracts import (
    CatalogResponse,
    ModelListResponse,
    ModelSelectionRequest,
    ModelSelectionResponse,
    SyncRequest,
    SyncResponse,
)
from vision_service.vision.backend import ModelNotFoundError, ModelRegistryError

router = APIRouter()


def get_container(request: Request) -> ServiceContainer:
    return request.app.state.container


def _raise_model_http_error(exc: ModelRegistryError) -> None:
    status_code = 404 if isinstance(exc, ModelNotFoundError) else 500
    raise HTTPException(status_code=status_code, detail=str(exc)) from exc


@router.get("/entities", response_model=CatalogResponse)
async def get_entities(
    model_name: str | None = Query(default=None),
    container: ServiceContainer = Depends(get_container),
) -> CatalogResponse:
    try:
        return await container.backend.get_catalog(
            service_version=container.settings.service_version,
            model_name=model_name,
        )
    except ModelRegistryError as exc:
        _raise_model_http_error(exc)


@router.get("/models", response_model=ModelListResponse)
async def get_models(
    container: ServiceContainer = Depends(get_container),
) -> ModelListResponse:
    try:
        return await container.backend.list_models(
            service_version=container.settings.service_version,
        )
    except ModelRegistryError as exc:
        _raise_model_http_error(exc)


@router.put("/model", response_model=ModelSelectionResponse)
async def select_model(
    payload: ModelSelectionRequest,
    container: ServiceContainer = Depends(get_container),
) -> ModelSelectionResponse:
    try:
        return await container.backend.select_model(payload.model_name)
    except ModelRegistryError as exc:
        _raise_model_http_error(exc)


@router.put("", response_model=SyncResponse)
async def sync_configuration(
    payload: SyncRequest,
    container: ServiceContainer = Depends(get_container),
) -> SyncResponse:
    await container.manager.apply_config(payload)
    return SyncResponse(ok=True)

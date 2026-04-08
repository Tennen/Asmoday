from fastapi import APIRouter, Depends, Request

from vision_service.container import ServiceContainer
from vision_service.contracts import CatalogResponse, SyncRequest, SyncResponse

router = APIRouter()


def get_container(request: Request) -> ServiceContainer:
    return request.app.state.container


@router.get("/entities", response_model=CatalogResponse)
async def get_entities(
    container: ServiceContainer = Depends(get_container),
) -> CatalogResponse:
    return await container.backend.get_catalog(
        service_version=container.settings.service_version,
    )


@router.put("", response_model=SyncResponse)
async def sync_configuration(
    payload: SyncRequest,
    container: ServiceContainer = Depends(get_container),
) -> SyncResponse:
    await container.manager.apply_config(payload)
    return SyncResponse(ok=True)

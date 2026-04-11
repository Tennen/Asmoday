from fastapi import APIRouter, Depends, WebSocket

from vision_service.container import ServiceContainer

router = APIRouter()


def get_container(websocket: WebSocket) -> ServiceContainer:
    return websocket.app.state.container


@router.websocket("")
async def gateway_session(
    websocket: WebSocket,
    container: ServiceContainer = Depends(get_container),
) -> None:
    await container.gateway_session.handle_connection(websocket)

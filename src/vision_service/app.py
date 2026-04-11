from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from vision_service.api import router
from vision_service.container import ServiceContainer
from vision_service.gateway import GatewaySessionController
from vision_service.runtime.manager import RuntimeManager
from vision_service.settings import get_settings
from vision_service.vision.backend import VisionBackend


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    backend = VisionBackend(settings)
    manager = RuntimeManager(
        settings=settings,
        backend=backend,
    )
    container = ServiceContainer(
        settings=settings,
        backend=backend,
        manager=manager,
        gateway_session=GatewaySessionController(
            settings=settings,
            backend=backend,
            manager=manager,
        ),
    )
    app.state.container = container

    await container.manager.start()
    try:
        yield
    finally:
        await container.manager.stop()


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.service_name,
        version=settings.service_version,
        lifespan=lifespan,
    )
    app.include_router(router, prefix=settings.gateway_ws_path)
    return app


def main() -> None:
    settings = get_settings()
    uvicorn.run(
        "vision_service.app:create_app",
        factory=True,
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level,
        ws_max_size=settings.websocket_max_message_bytes,
    )

from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from vision_service.api import router
from vision_service.container import ServiceContainer
from vision_service.runtime import RuntimeManager
from vision_service.settings import get_settings
from vision_service.vision import VisionBackend


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    container = ServiceContainer(
        settings=settings,
        backend=VisionBackend(settings),
        manager=RuntimeManager(settings),
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
    app.include_router(router, prefix=settings.api_prefix)
    return app


def main() -> None:
    settings = get_settings()
    uvicorn.run(
        "vision_service.app:create_app",
        factory=True,
        host=settings.host,
        port=settings.port,
    )

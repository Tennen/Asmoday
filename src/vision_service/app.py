from contextlib import asynccontextmanager
import logging

import uvicorn
from fastapi import FastAPI

from vision_service.api import router
from vision_service.container import ServiceContainer
from vision_service.gateway.session import GatewaySessionController
from vision_service.runtime.manager import RuntimeManager
from vision_service.settings import get_settings
from vision_service.vision.backend import VisionBackend


def configure_logging(level_name: str) -> None:
    level = getattr(logging, level_name.upper(), logging.INFO)
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(
            level=level,
            format="%(asctime)s %(levelname)s %(name)s %(message)s",
        )
    else:
        root_logger.setLevel(level)

    logging.getLogger("vision_service").setLevel(level)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level)
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
    app.include_router(router, prefix=settings.control_ws_path)
    return app


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    uvicorn.run(
        "vision_service.app:create_app",
        factory=True,
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level,
        ws_max_size=settings.websocket_max_message_bytes,
    )

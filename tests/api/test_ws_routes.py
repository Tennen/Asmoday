from fastapi import FastAPI
from fastapi.testclient import TestClient

from vision_service.api import router
from vision_service.container import ServiceContainer
from vision_service.contracts import (
    CatalogResponse,
    EntityDescriptor,
    ModelDescriptor,
    ModelListResponse,
    ModelSelectionResponse,
)
from vision_service.gateway.session import GatewaySessionController
from vision_service.runtime.manager import RuntimeManager
from vision_service.settings import Settings


class BackendStub:
    def __init__(self) -> None:
        self.catalog_requests: list[str | None] = []
        self.selected_model_name: str | None = None
        self.reset_calls = 0

    async def get_catalog(
        self,
        *,
        service_version: str,
        model_name: str | None = None,
    ) -> CatalogResponse:
        self.catalog_requests.append(model_name)
        return CatalogResponse.build(
            entities=[
                EntityDescriptor(
                    value="cat",
                    display_name="Cat",
                )
            ],
            service_version=service_version,
            model_name=model_name or "default.pt",
        )

    async def list_models(self, *, service_version: str) -> ModelListResponse:
        return ModelListResponse.build(
            models=[
                ModelDescriptor(
                    name="default.pt",
                    created_at="2026-04-10T08:00:00Z",
                    is_selected=self.selected_model_name in {None, "default.pt"},
                    is_default=True,
                ),
                ModelDescriptor(
                    name="alt.pt",
                    created_at="2026-04-10T09:00:00Z",
                    is_selected=self.selected_model_name == "alt.pt",
                    is_default=False,
                ),
            ],
            service_version=service_version,
            current_model_name=self.selected_model_name or "default.pt",
            default_model_name="default.pt",
        )

    async def select_model(self, model_name: str | None) -> ModelSelectionResponse:
        self.selected_model_name = model_name
        return ModelSelectionResponse.build(
            model_name=model_name or "default.pt",
        )

    async def reset_model_selection(self) -> None:
        self.reset_calls += 1
        self.selected_model_name = None


def _build_client() -> tuple[TestClient, BackendStub]:
    settings = Settings(
        control_ws_path="/ws/control",
        status_interval_seconds=3600,
    )
    backend = BackendStub()
    manager = RuntimeManager(settings=settings, backend=backend)
    session = GatewaySessionController(
        settings=settings,
        backend=backend,
        manager=manager,
    )

    app = FastAPI()
    app.include_router(router, prefix=settings.control_ws_path)
    app.state.container = ServiceContainer(
        settings=settings,
        backend=backend,
        manager=manager,
        gateway_session=session,
    )
    return TestClient(app), backend


def test_websocket_session_supports_query_commands_and_sync() -> None:
    client, backend = _build_client()

    with client.websocket_connect("/ws/control") as ws:
        hello = ws.receive_json()
        initial_status = ws.receive_json()

        assert hello["type"] == "hello"
        assert hello["payload"]["schema_version"] == "celestia.vision.ws.v1"
        assert initial_status["type"] == "runtime_status"
        assert initial_status["payload"]["status"] == "stopped"

        ws.send_json({"type": "get_models", "request_id": "models-1"})
        models = ws.receive_json()
        assert models["type"] == "models"
        assert models["request_id"] == "models-1"
        assert models["payload"]["current_model_name"] == "default.pt"

        ws.send_json(
            {
                "type": "select_model",
                "request_id": "select-1",
                "payload": {"model_name": "alt.pt"},
            }
        )
        model_selected = ws.receive_json()
        assert model_selected["type"] == "model_selected"
        assert model_selected["payload"]["model_name"] == "alt.pt"

        ws.send_json(
            {
                "type": "get_entities",
                "request_id": "entities-1",
                "payload": {"model_name": "alt.pt"},
            }
        )
        catalog = ws.receive_json()
        assert catalog["type"] == "entity_catalog"
        assert catalog["payload"]["model_name"] == "alt.pt"
        assert backend.catalog_requests == ["alt.pt"]

        ws.send_json(
            {
                "type": "sync_config",
                "request_id": "sync-1",
                "payload": {
                    "schema_version": "celestia.vision.control.ws.v1",
                    "sent_at": "2026-04-11T00:00:00Z",
                    "recognition_enabled": True,
                    "rules": [],
                },
            }
        )
        synced_status = ws.receive_json()
        sync_applied = ws.receive_json()

        assert synced_status["type"] == "runtime_status"
        assert sync_applied["type"] == "sync_applied"
        assert sync_applied["request_id"] == "sync-1"
        assert sync_applied["payload"]["ok"] is True

    assert backend.reset_calls == 1


def test_websocket_session_reports_protocol_errors() -> None:
    client, _ = _build_client()

    with client.websocket_connect("/ws/control") as ws:
        ws.receive_json()
        ws.receive_json()

        ws.send_json({"type": "unsupported", "request_id": "bad-1"})
        error = ws.receive_json()

        assert error["type"] == "error"
        assert error["request_id"] == "bad-1"
        assert error["payload"]["code"] == "unsupported_message_type"

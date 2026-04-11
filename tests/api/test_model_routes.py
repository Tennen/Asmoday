from types import SimpleNamespace
from typing import Any

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
from vision_service.vision.backend import ModelNotFoundError


class BackendStub:
    def __init__(self) -> None:
        self.catalog_requests: list[dict[str, Any]] = []
        self.selected_model_name: str | None = None

    async def get_catalog(
        self,
        *,
        service_version: str,
        model_name: str | None = None,
    ) -> CatalogResponse:
        self.catalog_requests.append(
            {
                "service_version": service_version,
                "model_name": model_name,
            }
        )
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


class MissingModelBackend(BackendStub):
    async def get_catalog(
        self,
        *,
        service_version: str,
        model_name: str | None = None,
    ) -> CatalogResponse:
        raise ModelNotFoundError(
            f"model '{model_name}' was not found in directory /tmp/models"
        )


class ManagerStub:
    def __init__(self) -> None:
        self.payloads: list[Any] = []

    async def apply_config(self, payload: Any) -> None:
        self.payloads.append(payload)


def _build_sync_payload() -> dict[str, Any]:
    return {
        "sent_at": "2026-04-11T00:00:00Z",
        "recognition_enabled": True,
        "callbacks": {
            "status_path": "/status",
            "event_path": "/events",
            "evidence_path": "/evidence",
        },
        "rules": [],
    }


def _build_client(backend: Any, manager: ManagerStub | None = None) -> TestClient:
    app = FastAPI()
    app.include_router(router, prefix="/api/v1/capabilities/vision_entity_stay_zone")
    manager = manager or ManagerStub()
    app.state.container = ServiceContainer(
        settings=SimpleNamespace(service_version="test-version"),
        backend=backend,
        gateway_client=object(),
        manager=manager,
    )
    return TestClient(app)


def test_entities_route_accepts_optional_model_name_query() -> None:
    backend = BackendStub()
    client = _build_client(backend)

    response = client.get(
        "/api/v1/capabilities/vision_entity_stay_zone/entities",
        params={"model_name": "alt.pt"},
    )

    assert response.status_code == 200
    assert response.json()["model_name"] == "alt.pt"
    assert backend.catalog_requests == [
        {
            "service_version": "test-version",
            "model_name": "alt.pt",
        }
    ]


def test_model_routes_list_and_select_active_model() -> None:
    backend = BackendStub()
    client = _build_client(backend)

    list_response = client.get("/api/v1/capabilities/vision_entity_stay_zone/models")
    select_response = client.put(
        "/api/v1/capabilities/vision_entity_stay_zone/model",
        json={"model_name": "alt.pt"},
    )

    assert list_response.status_code == 200
    assert list_response.json()["current_model_name"] == "default.pt"
    assert select_response.status_code == 200
    assert select_response.json()["model_name"] == "alt.pt"
    assert backend.selected_model_name == "alt.pt"


def test_model_not_found_errors_become_http_404() -> None:
    client = _build_client(MissingModelBackend())

    response = client.get(
        "/api/v1/capabilities/vision_entity_stay_zone/entities",
        params={"model_name": "missing.pt"},
    )

    assert response.status_code == 404
    assert "missing.pt" in response.json()["detail"]


def test_sync_route_accepts_trailing_slash() -> None:
    backend = BackendStub()
    manager = ManagerStub()
    client = _build_client(backend, manager)

    response = client.put(
        "/api/v1/capabilities/vision_entity_stay_zone/",
        json=_build_sync_payload(),
    )

    assert response.status_code == 200
    assert response.json() == {"ok": True}
    assert len(manager.payloads) == 1

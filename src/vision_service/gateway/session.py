import asyncio
import logging
from uuid import uuid4

from fastapi import WebSocket, WebSocketDisconnect, status
from pydantic import ValidationError

from vision_service.contracts import (
    EntityCatalogRequest,
    ErrorPayload,
    ModelSelectionRequest,
    SessionHelloPayload,
    SyncAppliedPayload,
    SyncRequest,
    WebSocketEnvelope,
)
from vision_service.gateway.transport import GatewayWebSocketTransport
from vision_service.runtime.manager import RuntimeManager
from vision_service.settings import Settings
from vision_service.vision.backend import ModelNotFoundError, ModelRegistryError, VisionBackend

logger = logging.getLogger(__name__)


class GatewaySessionController:
    def __init__(
        self,
        *,
        settings: Settings,
        backend: VisionBackend,
        manager: RuntimeManager,
    ) -> None:
        self._settings = settings
        self._backend = backend
        self._manager = manager
        self._session_lock = asyncio.Lock()
        self._active_session_id: str | None = None

    async def handle_connection(self, websocket: WebSocket) -> None:
        await websocket.accept()

        session_id = uuid4().hex
        transport = GatewayWebSocketTransport(websocket)
        if not await self._acquire_session(session_id):
            await transport.send_error(
                code="session_exists",
                message="only one Gateway websocket session is allowed at a time",
            )
            await websocket.close(
                code=status.WS_1008_POLICY_VIOLATION,
                reason="session already active",
            )
            return

        logger.info(
            "gateway websocket connected session_id=%s client=%s",
            session_id,
            websocket.client,
        )
        await self._manager.attach_transport(transport)

        try:
            await transport.send_hello(
                SessionHelloPayload.build(
                    service_version=self._settings.service_version,
                ).model_dump(mode="json")
            )
            await self._manager.report_status()

            while True:
                message = await websocket.receive_json()
                await self._handle_message(
                    transport=transport,
                    raw_message=message,
                )
        except WebSocketDisconnect as exc:
            logger.info(
                "gateway websocket disconnected session_id=%s code=%s",
                session_id,
                exc.code,
            )
        finally:
            await self._manager.clear_session(
                reason="gateway websocket disconnected",
            )
            await self._backend.reset_model_selection()
            await self._release_session(session_id)

    async def _handle_message(
        self,
        *,
        transport: GatewayWebSocketTransport,
        raw_message: object,
    ) -> None:
        request_id: str | None = None
        try:
            envelope = WebSocketEnvelope.model_validate(raw_message)
            request_id = envelope.request_id
            logger.info(
                "received websocket message type=%s request_id=%s",
                envelope.type,
                request_id,
            )

            if envelope.type == "get_models":
                response = await self._backend.list_models(
                    service_version=self._settings.service_version,
                )
                await transport.send_message(
                    type="models",
                    request_id=request_id,
                    payload=response.model_dump(mode="json", exclude_none=True),
                )
                return

            if envelope.type == "select_model":
                payload = ModelSelectionRequest.model_validate(envelope.payload or {})
                response = await self._backend.select_model(payload.model_name)
                await transport.send_message(
                    type="model_selected",
                    request_id=request_id,
                    payload=response.model_dump(mode="json", exclude_none=True),
                )
                return

            if envelope.type == "get_entities":
                payload = EntityCatalogRequest.model_validate(envelope.payload or {})
                response = await self._backend.get_catalog(
                    service_version=self._settings.service_version,
                    model_name=payload.model_name,
                )
                await transport.send_message(
                    type="entity_catalog",
                    request_id=request_id,
                    payload=response.model_dump(mode="json", exclude_none=True),
                )
                return

            if envelope.type == "sync_config":
                payload = SyncRequest.model_validate(envelope.payload or {})
                enabled_rule_count = sum(1 for rule in payload.rules if rule.enabled)
                logger.info(
                    "received sync_config request_id=%s sent_at=%s "
                    "recognition_enabled=%s configured_rules=%s enabled_rules=%s",
                    request_id,
                    payload.sent_at.isoformat(),
                    payload.recognition_enabled,
                    len(payload.rules),
                    enabled_rule_count,
                )
                await self._manager.apply_config(payload)
                await transport.send_message(
                    type="sync_applied",
                    request_id=request_id,
                    payload=SyncAppliedPayload.build().model_dump(mode="json"),
                )
                logger.info(
                    "sent sync_applied request_id=%s sent_at=%s",
                    request_id,
                    payload.sent_at.isoformat(),
                )
                return

            await self._send_error(
                transport=transport,
                request_id=request_id,
                code="unsupported_message_type",
                message=f"unsupported message type: {envelope.type}",
            )
        except ValidationError as exc:
            await self._send_error(
                transport=transport,
                request_id=request_id,
                code="invalid_message",
                message=str(exc),
            )
        except ModelNotFoundError as exc:
            await self._send_error(
                transport=transport,
                request_id=request_id,
                code="model_not_found",
                message=str(exc),
            )
        except ModelRegistryError as exc:
            await self._send_error(
                transport=transport,
                request_id=request_id,
                code="model_registry_error",
                message=str(exc),
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "gateway websocket message handling failed request_id=%s",
                request_id,
            )
            await self._send_error(
                transport=transport,
                request_id=request_id,
                code="internal_error",
                message=str(exc),
            )

    async def _send_error(
        self,
        *,
        transport: GatewayWebSocketTransport,
        request_id: str | None,
        code: str,
        message: str,
    ) -> None:
        payload = ErrorPayload(code=code, message=message)
        await transport.send_message(
            type="error",
            request_id=request_id,
            payload=payload.model_dump(mode="json"),
        )

    async def _acquire_session(self, session_id: str) -> bool:
        async with self._session_lock:
            if self._active_session_id is not None:
                return False
            self._active_session_id = session_id
            return True

    async def _release_session(self, session_id: str) -> None:
        async with self._session_lock:
            if self._active_session_id == session_id:
                self._active_session_id = None

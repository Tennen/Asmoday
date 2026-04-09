import asyncio
from base64 import b64encode
from datetime import UTC, datetime
from uuid import uuid4

from vision_service.contracts import (
    EvidenceCallbackPayload,
    EvidenceCapture,
    EventCallbackPayload,
    EventRecord,
    RuntimeStatusPayload,
    SyncRequest,
)
from vision_service.gateway import CallbackDeliveryError, GatewayCallbackClient
from vision_service.runtime.events import RuleEvent
from vision_service.settings import Settings


class RuntimeManager:
    def __init__(
        self,
        settings: Settings,
        gateway_client: GatewayCallbackClient,
    ) -> None:
        self._settings = settings
        self._gateway_client = gateway_client
        self._lock = asyncio.Lock()
        self._desired_state: SyncRequest | None = None
        self._last_runtime_error: str | None = None
        self._last_delivery_error: str | None = None
        self._status_task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        await self._gateway_client.start()
        if self._status_task is None:
            self._status_task = asyncio.create_task(self._status_loop())

    async def stop(self) -> None:
        if self._status_task is not None:
            self._status_task.cancel()
            try:
                await self._status_task
            except asyncio.CancelledError:
                pass
            self._status_task = None
        await self._gateway_client.stop()

    async def apply_config(self, payload: SyncRequest) -> None:
        async with self._lock:
            self._desired_state = payload.model_copy(deep=True)
            self._last_runtime_error = None
            self._last_delivery_error = None
        try:
            await self.report_status()
        except CallbackDeliveryError:
            return

    async def current_config(self) -> SyncRequest | None:
        async with self._lock:
            if self._desired_state is None:
                return None
            return self._desired_state.model_copy(deep=True)

    async def snapshot_status(self) -> RuntimeStatusPayload:
        async with self._lock:
            desired_state = self._desired_state
            last_runtime_error = self._last_runtime_error
            last_delivery_error = self._last_delivery_error

        if desired_state is None:
            return RuntimeStatusPayload(
                status="stopped",
                message="awaiting configuration from Gateway",
                service_version=self._settings.service_version,
                reported_at=datetime.now(tz=UTC),
                runtime={
                    "configured_rules": 0,
                    "enabled_rules": 0,
                    "active_streams": 0,
                },
            )

        enabled_rules = [rule for rule in desired_state.rules if rule.enabled]
        recognition_enabled = desired_state.recognition_enabled

        if not recognition_enabled:
            status = "stopped"
            message = "recognition disabled by Gateway"
        elif last_runtime_error:
            status = "degraded"
            message = last_runtime_error
        elif last_delivery_error:
            status = "degraded"
            message = last_delivery_error
        else:
            status = "healthy"
            message = f"configured {len(enabled_rules)} active rule(s)"

        return RuntimeStatusPayload(
            status=status,
            message=message,
            service_version=self._settings.service_version,
            reported_at=datetime.now(tz=UTC),
            runtime={
                "configured_rules": len(desired_state.rules),
                "enabled_rules": len(enabled_rules),
                "active_streams": len(enabled_rules) if recognition_enabled else 0,
                "last_delivery_error": last_delivery_error,
            },
        )

    async def set_runtime_error(self, message: str) -> None:
        async with self._lock:
            self._last_runtime_error = message

    async def report_status(self) -> None:
        status_payload = await self.snapshot_status()
        callback_path = await self._status_callback_path()
        if callback_path is None:
            return

        try:
            await self._gateway_client.post_status(
                callback_path=callback_path,
                payload=status_payload,
            )
        except CallbackDeliveryError as exc:
            async with self._lock:
                self._last_delivery_error = str(exc)
            raise

        async with self._lock:
            self._last_delivery_error = None

    async def report_rule_event(self, event: RuleEvent) -> str:
        async with self._lock:
            desired_state = self._desired_state

        if desired_state is None:
            raise RuntimeError("cannot emit events before configuration is applied")

        event_id = event.event_id or self._new_event_id()
        try:
            await self._gateway_client.post_events(
                callback_path=desired_state.callbacks.event_path,
                payload=EventCallbackPayload(
                    events=[
                        EventRecord(
                            event_id=event_id,
                            rule_id=event.rule_id,
                            camera_device_id=event.camera_device_id,
                            status=event.status,
                            observed_at=event.observed_at,
                            dwell_seconds=event.dwell_seconds,
                            entity_value=event.entity_value,
                            metadata=event.metadata or None,
                        )
                    ]
                ),
            )

            if event.evidence:
                await self._gateway_client.post_evidence(
                    callback_path=desired_state.callbacks.evidence_path,
                    payload=EvidenceCallbackPayload(
                        captures=[
                            EvidenceCapture(
                                capture_id=f"{event_id}:{capture.phase}",
                                event_id=event_id,
                                rule_id=event.rule_id,
                                camera_device_id=event.camera_device_id,
                                phase=capture.phase,
                                captured_at=capture.captured_at,
                                content_type="image/jpeg",
                                image_base64=b64encode(capture.image_bytes).decode("ascii"),
                            )
                            for capture in event.evidence
                        ]
                    ),
                )
        except CallbackDeliveryError as exc:
            async with self._lock:
                self._last_delivery_error = str(exc)
            raise

        async with self._lock:
            self._last_delivery_error = None
        return event_id

    async def _status_callback_path(self) -> str | None:
        async with self._lock:
            if self._desired_state is None:
                return None
            return self._desired_state.callbacks.status_path

    async def _status_loop(self) -> None:
        while True:
            await asyncio.sleep(self._settings.status_interval_seconds)
            try:
                await self.report_status()
            except CallbackDeliveryError:
                continue

    def _new_event_id(self) -> str:
        return f"{self._settings.event_id_prefix}-{uuid4()}"

import asyncio
from base64 import b64encode
from dataclasses import asdict
from datetime import UTC, datetime
from uuid import uuid4

from vision_service.contracts import (
    EvidenceCallbackPayload,
    EvidenceCapture,
    EventCallbackPayload,
    EventRecord,
    RuntimeStatusPayload,
    SyncRequest,
    VisionRule,
)
from vision_service.gateway import CallbackDeliveryError, GatewayCallbackClient
from vision_service.runtime.events import RuleEvent
from vision_service.settings import Settings
from vision_service.vision.backend import VisionBackend
from vision_service.vision.pipeline import RuleVisionWorker
from vision_service.vision.stream import FrameStream, SharedRTSPStream


class RuntimeManager:
    def __init__(
        self,
        settings: Settings,
        gateway_client: GatewayCallbackClient,
        backend: VisionBackend,
    ) -> None:
        self._settings = settings
        self._gateway_client = gateway_client
        self._backend = backend
        self._lock = asyncio.Lock()
        self._reconcile_lock = asyncio.Lock()
        self._desired_state: SyncRequest | None = None
        self._last_runtime_error: str | None = None
        self._last_delivery_error: str | None = None
        self._status_task: asyncio.Task[None] | None = None
        self._workers: dict[str, RuleVisionWorker] = {}
        self._streams: dict[str, SharedRTSPStream] = {}

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
        workers = await self._current_workers()
        for worker in workers:
            await worker.stop()
        streams = await self._current_streams()
        for stream in streams:
            await stream.stop()
        await self._gateway_client.stop()

    async def apply_config(self, payload: SyncRequest) -> None:
        async with self._reconcile_lock:
            workers_to_stop, workers_to_keep, workers_to_start = await self._plan_reconcile(
                payload=payload,
            )
            desired_rules = self._enabled_rules(payload)

            for worker in workers_to_stop:
                await worker.stop()

            reconciled_streams, streams_to_start, streams_to_stop = (
                await self._plan_stream_reconcile(rules=desired_rules)
            )

            for stream in streams_to_stop:
                await stream.stop()
            for stream in streams_to_start:
                await stream.start()

            async with self._lock:
                self._desired_state = payload.model_copy(deep=True)
                self._workers = {worker.rule_id: worker for worker in workers_to_keep}
                self._streams = reconciled_streams
                self._last_runtime_error = None
                self._last_delivery_error = None

                for rule in workers_to_start:
                    self._workers[rule.id] = self._create_worker(
                        rule=rule,
                        frame_stream=self._streams[self._stream_key_for_rule(rule)],
                    )

                new_workers = [
                    self._workers[rule.id]
                    for rule in workers_to_start
                ]

            for worker in new_workers:
                await worker.start()

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
            workers = list(self._workers.values())
            streams = list(self._streams.values())

        worker_snapshots = [worker.snapshot() for worker in workers]
        stream_snapshots = [stream.snapshot() for stream in streams]

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
        elif any(
            snapshot.state == "degraded" or snapshot.last_error
            for snapshot in worker_snapshots
        ):
            status = "degraded"
            message = next(
                (
                    snapshot.last_error
                    for snapshot in worker_snapshots
                    if snapshot.last_error
                ),
                "one or more workers are degraded",
            )
        elif any(
            snapshot.state == "degraded" or snapshot.last_error
            for snapshot in stream_snapshots
        ):
            status = "degraded"
            message = next(
                (
                    snapshot.last_error
                    for snapshot in stream_snapshots
                    if snapshot.last_error
                ),
                "one or more streams are degraded",
            )
        else:
            status = "healthy"
            message = (
                f"tracking {len(stream_snapshots)} stream(s) across "
                f"{len(worker_snapshots)} rule(s)"
            )

        return RuntimeStatusPayload(
            status=status,
            message=message,
            service_version=self._settings.service_version,
            reported_at=datetime.now(tz=UTC),
            runtime={
                "configured_rules": len(desired_state.rules),
                "enabled_rules": len(enabled_rules),
                "active_streams": sum(
                    1
                    for snapshot in stream_snapshots
                    if snapshot.state in {"starting", "running"}
                ),
                "last_delivery_error": last_delivery_error,
                "workers": [asdict(snapshot) for snapshot in worker_snapshots],
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

    async def _current_workers(self) -> list[RuleVisionWorker]:
        async with self._lock:
            return list(self._workers.values())

    async def _current_streams(self) -> list[SharedRTSPStream]:
        async with self._lock:
            return list(self._streams.values())

    async def _plan_reconcile(
        self,
        *,
        payload: SyncRequest,
    ) -> tuple[list[RuleVisionWorker], list[RuleVisionWorker], list[VisionRule]]:
        async with self._lock:
            existing_workers = list(self._workers.values())
            current_state = self._desired_state

        callbacks_changed = (
            current_state is not None
            and current_state.callbacks.model_dump(mode="json")
            != payload.callbacks.model_dump(mode="json")
        )
        desired_rules = {
            rule.id: rule
            for rule in self._enabled_rules(payload)
        }

        workers_to_stop: list[RuleVisionWorker] = []
        workers_to_keep: list[RuleVisionWorker] = []

        for worker in existing_workers:
            desired_rule = desired_rules.pop(worker.rule_id, None)
            if desired_rule is None or callbacks_changed or not worker.matches(desired_rule):
                workers_to_stop.append(worker)
                continue
            workers_to_keep.append(worker)

        return workers_to_stop, workers_to_keep, list(desired_rules.values())

    async def _plan_stream_reconcile(
        self,
        *,
        rules: list[VisionRule],
    ) -> tuple[dict[str, SharedRTSPStream], list[SharedRTSPStream], list[SharedRTSPStream]]:
        async with self._lock:
            current_streams = dict(self._streams)

        required_stream_keys = {
            self._stream_key_for_rule(rule)
            for rule in rules
        }
        reconciled_streams = {
            stream_key: stream
            for stream_key, stream in current_streams.items()
            if stream_key in required_stream_keys
        }
        streams_to_stop = [
            stream
            for stream_key, stream in current_streams.items()
            if stream_key not in required_stream_keys
        ]

        streams_to_start: list[SharedRTSPStream] = []
        for stream_key in sorted(required_stream_keys):
            if stream_key in reconciled_streams:
                continue
            stream = self._create_stream(url=stream_key)
            reconciled_streams[stream_key] = stream
            streams_to_start.append(stream)

        return reconciled_streams, streams_to_start, streams_to_stop

    def _enabled_rules(self, payload: SyncRequest) -> list[VisionRule]:
        return [
            rule.model_copy(deep=True)
            for rule in payload.rules
            if payload.recognition_enabled and rule.enabled
        ]

    def _stream_key_for_rule(self, rule: VisionRule) -> str:
        return rule.rtsp_source.url

    def _create_stream(self, *, url: str) -> SharedRTSPStream:
        return SharedRTSPStream(url=url, settings=self._settings)

    def _create_worker(
        self,
        *,
        rule: VisionRule,
        frame_stream: FrameStream,
    ) -> RuleVisionWorker:
        return RuleVisionWorker(
            rule=rule,
            backend=self._backend,
            settings=self._settings,
            emit_rule_event=self.report_rule_event,
            frame_stream=frame_stream,
        )

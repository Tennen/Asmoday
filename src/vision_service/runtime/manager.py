import asyncio
import logging
from dataclasses import asdict
from datetime import UTC, datetime
from uuid import uuid4

from vision_service.contracts import (
    RuntimeStatusPayload,
    SyncRequest,
    VisionRule,
)
from vision_service.gateway.transport import GatewayTransport, GatewayTransportError
from vision_service.runtime.events import RuleEvent
from vision_service.runtime.telemetry import (
    build_evidence_callback_payload,
    build_event_callback_payload,
)
from vision_service.settings import Settings
from vision_service.vision.analysis import AnalyzedFrameStream, SharedInferenceStream
from vision_service.vision.backend import VisionBackend
from vision_service.vision.pipeline import RuleVisionWorker
from vision_service.vision.stream import FrameStream, SharedRTSPStream

logger = logging.getLogger(__name__)


class RuntimeManager:
    def __init__(
        self,
        settings: Settings,
        backend: VisionBackend,
    ) -> None:
        self._settings = settings
        self._backend = backend
        self._lock = asyncio.Lock()
        self._reconcile_lock = asyncio.Lock()
        self._desired_state: SyncRequest | None = None
        self._last_runtime_error: str | None = None
        self._last_delivery_error: str | None = None
        self._status_task: asyncio.Task[None] | None = None
        self._transport: GatewayTransport | None = None
        self._accepting_telemetry = False
        self._workers: dict[str, RuleVisionWorker] = {}
        self._streams: dict[str, SharedRTSPStream] = {}
        self._analysis_streams: dict[str, SharedInferenceStream] = {}

    async def start(self) -> None:
        if self._status_task is None:
            self._status_task = asyncio.create_task(self._status_loop())
            logger.info(
                "runtime manager started status_interval_seconds=%s",
                self._settings.status_interval_seconds,
            )

    async def stop(self) -> None:
        if self._status_task is not None:
            self._status_task.cancel()
            try:
                await self._status_task
            except asyncio.CancelledError:
                pass
            self._status_task = None
        await self.clear_session(reason="service stopping")
        logger.info("runtime manager stopped")

    async def attach_transport(self, transport: GatewayTransport) -> None:
        async with self._lock:
            self._transport = transport
            self._accepting_telemetry = True
            self._last_delivery_error = None
        logger.info("runtime manager attached websocket transport")

    async def clear_session(self, *, reason: str) -> None:
        async with self._reconcile_lock:
            workers = await self._current_workers()
            streams = await self._current_streams()

            async with self._lock:
                self._transport = None
                self._accepting_telemetry = False

            for worker in workers:
                await worker.stop()
            for stream in streams:
                await stream.stop()

            async with self._lock:
                self._desired_state = None
                self._last_runtime_error = None
                self._last_delivery_error = None
                self._workers = {}
                self._streams = {}
                self._analysis_streams = {}

        logger.info(
            "cleared runtime session reason=%s stopped_workers=%s stopped_streams=%s",
            reason,
            len(workers),
            len(streams),
        )

    async def apply_config(self, payload: SyncRequest) -> None:
        async with self._reconcile_lock:
            workers_to_stop, workers_to_keep, workers_to_start = await self._plan_reconcile(
                payload=payload,
            )
            desired_rules = self._enabled_rules(payload)
            enabled_rule_count = sum(1 for rule in payload.rules if rule.enabled)

            logger.info(
                "reconciling config sync sent_at=%s recognition_enabled=%s "
                "configured_rules=%s enabled_rules=%s runnable_rules=%s "
                "workers_to_keep=%s workers_to_stop=%s workers_to_start=%s",
                payload.sent_at.isoformat(),
                payload.recognition_enabled,
                len(payload.rules),
                enabled_rule_count,
                len(desired_rules),
                len(workers_to_keep),
                len(workers_to_stop),
                len(workers_to_start),
            )
            if not desired_rules:
                logger.info(
                    "config sync left no runnable rules sent_at=%s "
                    "recognition_enabled=%s configured_rules=%s enabled_rules=%s",
                    payload.sent_at.isoformat(),
                    payload.recognition_enabled,
                    len(payload.rules),
                    enabled_rule_count,
                )

            for worker in workers_to_stop:
                snapshot = worker.snapshot()
                logger.info(
                    "stopping rule worker due to config change rule_id=%s "
                    "camera_device_id=%s",
                    snapshot.rule_id,
                    snapshot.camera_device_id,
                )
                await worker.stop()

            reconciled_streams, streams_to_start, streams_to_stop = (
                await self._plan_stream_reconcile(rules=desired_rules)
            )

            for stream in streams_to_stop:
                logger.info("stopping unused RTSP stream url=%s", stream.url)
                await stream.stop()
            for stream in streams_to_start:
                await stream.start()

            async with self._lock:
                self._desired_state = payload.model_copy(deep=True)
                self._workers = {worker.rule_id: worker for worker in workers_to_keep}
                self._streams = reconciled_streams
                self._analysis_streams = self._reconcile_analysis_streams(
                    streams=reconciled_streams,
                )
                self._last_runtime_error = None
                self._last_delivery_error = None

                for rule in workers_to_start:
                    logger.info(
                        "creating rule worker rule_id=%s camera_device_id=%s "
                        "entity_value=%s stay_threshold_seconds=%s stream_url=%s",
                        rule.id,
                        rule.camera.device_id,
                        rule.entity_selector.value,
                        rule.stay_threshold_seconds,
                        rule.rtsp_source.url,
                    )
                    self._workers[rule.id] = self._create_worker(
                        rule=rule,
                        frame_stream=self._analysis_streams[
                            self._stream_key_for_rule(rule)
                        ],
                    )

                new_workers = [self._workers[rule.id] for rule in workers_to_start]

            for worker in new_workers:
                await worker.start()

            logger.info(
                "config sync applied sent_at=%s active_workers=%s active_streams=%s "
                "started_workers=%s stopped_workers=%s started_streams=%s "
                "stopped_streams=%s",
                payload.sent_at.isoformat(),
                len(workers_to_keep) + len(new_workers),
                len(reconciled_streams),
                len(new_workers),
                len(workers_to_stop),
                len(streams_to_start),
                len(streams_to_stop),
            )

        try:
            await self.report_status()
        except GatewayTransportError:
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
                message="awaiting websocket configuration from Gateway",
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
        transport = await self._transport_for_send()
        if transport is None:
            return

        try:
            await transport.send_status(status_payload)
        except GatewayTransportError as exc:
            logger.warning("runtime status delivery failed error=%s", exc)
            async with self._lock:
                self._last_delivery_error = str(exc)
            raise

        async with self._lock:
            self._last_delivery_error = None

    async def report_rule_event(self, event: RuleEvent) -> str:
        async with self._lock:
            desired_state = self._desired_state
            transport = self._transport if self._accepting_telemetry else None

        event_id = event.event_id or self._new_event_id()
        if transport is None:
            return event_id
        if desired_state is None:
            raise RuntimeError("cannot emit events before configuration is applied")

        try:
            await transport.send_events(
                build_event_callback_payload(
                    event=event,
                    event_id=event_id,
                )
            )

            evidence_payload = build_evidence_callback_payload(
                event=event,
                event_id=event_id,
            )
            if evidence_payload is not None:
                await transport.send_evidence(
                    evidence_payload,
                )
        except GatewayTransportError as exc:
            logger.warning(
                "rule event delivery failed rule_id=%s camera_device_id=%s "
                "event_id=%s status=%s error=%s",
                event.rule_id,
                event.camera_device_id,
                event_id,
                event.status,
                exc,
            )
            async with self._lock:
                self._last_delivery_error = str(exc)
            raise

        async with self._lock:
            self._last_delivery_error = None
        logger.info(
            "delivered rule event rule_id=%s camera_device_id=%s event_id=%s "
            "status=%s dwell_seconds=%s evidence_captures=%s",
            event.rule_id,
            event.camera_device_id,
            event_id,
            event.status,
            event.dwell_seconds,
            len(event.evidence),
        )
        return event_id

    async def _status_loop(self) -> None:
        while True:
            await asyncio.sleep(self._settings.status_interval_seconds)
            try:
                await self.report_status()
            except GatewayTransportError:
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

        desired_rules = {rule.id: rule for rule in self._enabled_rules(payload)}
        workers_to_stop: list[RuleVisionWorker] = []
        workers_to_keep: list[RuleVisionWorker] = []

        for worker in existing_workers:
            desired_rule = desired_rules.pop(worker.rule_id, None)
            if desired_rule is None or not worker.matches(desired_rule):
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

        required_stream_keys = {self._stream_key_for_rule(rule) for rule in rules}
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

    def _reconcile_analysis_streams(
        self,
        *,
        streams: dict[str, SharedRTSPStream],
    ) -> dict[str, SharedInferenceStream]:
        return {
            stream_key: self._analysis_streams.get(stream_key)
            or self._create_analysis_stream(frame_stream=stream)
            for stream_key, stream in streams.items()
        }

    def _create_analysis_stream(
        self,
        *,
        frame_stream: FrameStream,
    ) -> SharedInferenceStream:
        return SharedInferenceStream(
            frame_stream=frame_stream,
            backend=self._backend,
        )

    def _create_worker(
        self,
        *,
        rule: VisionRule,
        frame_stream: AnalyzedFrameStream,
    ) -> RuleVisionWorker:
        return RuleVisionWorker(
            rule=rule,
            settings=self._settings,
            emit_rule_event=self.report_rule_event,
            frame_stream=frame_stream,
        )

    async def _transport_for_send(self) -> GatewayTransport | None:
        async with self._lock:
            if not self._accepting_telemetry:
                return None
            return self._transport

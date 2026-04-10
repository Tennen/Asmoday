import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Awaitable, Callable, Literal

import numpy as np

from vision_service.contracts import VisionRule
from vision_service.runtime.dwell import DwellTransition, RuleDwellTracker
from vision_service.runtime.events import EventEvidence, RuleEvent
from vision_service.settings import Settings
from vision_service.vision.backend import DetectionBatch, VisionBackend
from vision_service.vision.capture import open_rtsp_capture

WorkerState = Literal["starting", "running", "stopped", "degraded"]
EmitRuleEvent = Callable[[RuleEvent], Awaitable[str]]


@dataclass(slots=True, frozen=True)
class WorkerSnapshot:
    rule_id: str
    camera_device_id: str
    state: WorkerState
    active: bool
    last_frame_at: datetime | None
    last_error: str | None
    emitted_threshold_events: int


class RuleVisionWorker:
    def __init__(
        self,
        *,
        rule: VisionRule,
        backend: VisionBackend,
        settings: Settings,
        emit_rule_event: EmitRuleEvent,
    ) -> None:
        self._rule = rule.model_copy(deep=True)
        self._backend = backend
        self._settings = settings
        self._emit_rule_event = emit_rule_event

        self._state: WorkerState = "starting"
        self._active = False
        self._last_frame_at: datetime | None = None
        self._last_error: str | None = None
        self._emitted_threshold_events = 0

        self._task: asyncio.Task[None] | None = None
        self._stop_event = asyncio.Event()

    @property
    def rule_id(self) -> str:
        return self._rule.id

    def matches(self, rule: VisionRule) -> bool:
        return self._rule.model_dump(mode="json") == rule.model_dump(mode="json")

    def snapshot(self) -> WorkerSnapshot:
        return WorkerSnapshot(
            rule_id=self._rule.id,
            camera_device_id=self._rule.camera.device_id,
            state=self._state,
            active=self._active,
            last_frame_at=self._last_frame_at,
            last_error=self._last_error,
            emitted_threshold_events=self._emitted_threshold_events,
        )

    async def start(self) -> None:
        if self._task is not None:
            return
        self._stop_event.clear()
        self._task = asyncio.create_task(self._run_wrapper())

    async def stop(self) -> None:
        if self._task is None:
            self._state = "stopped"
            return
        self._stop_event.set()
        await self._task
        self._task = None

    async def _run_wrapper(self) -> None:
        try:
            await self._run()
        except Exception as exc:  # noqa: BLE001
            self._state = "degraded"
            self._last_error = str(exc)
        else:
            self._state = "stopped"

    async def _run(self) -> None:
        import supervision as sv

        capture = await asyncio.to_thread(
            open_rtsp_capture,
            url=self._rule.rtsp_source.url,
            settings=self._settings,
        )
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(
                "unable to open RTSP stream for "
                f"rule {self._rule.id} with rtsp_transport={self._settings.rtsp_transport}"
            )

        tracker = sv.ByteTrack(
            lost_track_buffer=self._settings.tracker_lost_track_buffer,
        )
        dwell_tracker = RuleDwellTracker(
            threshold_seconds=self._rule.stay_threshold_seconds,
            sample_interval_seconds=self._settings.frame_sample_interval_seconds,
            max_samples=self._settings.evidence_buffer_max_samples,
        )
        self._state = "running"

        try:
            while not self._stop_event.is_set():
                success, frame = await asyncio.to_thread(capture.read)
                observed_at = datetime.now(tz=UTC)

                if not success:
                    self._last_error = (
                        "failed to read frame from RTSP stream with "
                        f"rtsp_transport={self._settings.rtsp_transport}"
                    )
                    transition = dwell_tracker.observe(
                        observed_at=observed_at,
                        visible_tracks={},
                    )
                    if transition is not None:
                        await self._emit_transition(transition)
                    await asyncio.sleep(
                        self._settings.frame_failure_backoff_seconds,
                    )
                    continue

                transition = await self._process_frame(
                    tracker=tracker,
                    frame=frame,
                    observed_at=observed_at,
                    dwell_tracker=dwell_tracker,
                )
                self._active = dwell_tracker.active
                self._last_frame_at = observed_at
                if transition is not None:
                    await self._emit_transition(transition)
                await asyncio.sleep(self._settings.idle_sleep_seconds)
        finally:
            transition = dwell_tracker.force_clear(observed_at=datetime.now(tz=UTC))
            if transition is not None:
                await self._emit_transition(transition)
            await asyncio.to_thread(capture.release)
            self._active = False

    async def _process_frame(
        self,
        *,
        tracker: Any,
        frame: np.ndarray[Any, Any],
        observed_at: datetime,
        dwell_tracker: RuleDwellTracker,
    ) -> DwellTransition | None:
        batch = await self._backend.detect(frame)
        target_detections = self._select_target_detections(batch)
        tracked_detections = tracker.update_with_detections(target_detections)
        visible_tracks = self._visible_tracks_in_zone(
            detections=tracked_detections,
            frame=frame,
        )
        return dwell_tracker.observe(
            observed_at=observed_at,
            visible_tracks=visible_tracks,
        )

    def _select_target_detections(self, batch: DetectionBatch) -> Any:
        detections = batch.detections
        if len(detections) == 0 or detections.class_id is None:
            return detections

        mask = np.array(
            [
                batch.labels.get(int(class_id)) == self._rule.entity_selector.value
                for class_id in detections.class_id
            ],
            dtype=bool,
        )
        return detections[mask]

    def _visible_tracks_in_zone(
        self,
        *,
        detections: Any,
        frame: np.ndarray[Any, Any],
    ) -> dict[int, bytes]:
        if len(detections) == 0 or detections.tracker_id is None:
            return {}

        frame_height, frame_width = frame.shape[:2]
        zone_left = self._rule.zone.x * frame_width
        zone_top = self._rule.zone.y * frame_height
        zone_right = zone_left + (self._rule.zone.width * frame_width)
        zone_bottom = zone_top + (self._rule.zone.height * frame_height)

        track_ids: list[int] = []
        for bounding_box, tracker_id in zip(detections.xyxy, detections.tracker_id):
            if tracker_id is None:
                continue
            center_x = float((bounding_box[0] + bounding_box[2]) / 2.0)
            center_y = float((bounding_box[1] + bounding_box[3]) / 2.0)
            if zone_left <= center_x <= zone_right and zone_top <= center_y <= zone_bottom:
                track_ids.append(int(tracker_id))

        if not track_ids:
            return {}

        encoded_frame = self._encode_frame(frame)
        return {track_id: encoded_frame for track_id in track_ids}

    def _encode_frame(self, frame: np.ndarray[Any, Any]) -> bytes:
        import cv2

        success, buffer = cv2.imencode(
            ".jpg",
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, self._settings.jpeg_quality],
        )
        if not success:
            raise RuntimeError("failed to encode evidence frame")
        return buffer.tobytes()

    async def _emit_transition(self, transition: DwellTransition) -> None:
        evidence: tuple[EventEvidence, ...] = ()
        if transition.status == "threshold_met" and transition.evidence_samples:
            phases = ("start", "middle", "end")
            evidence = tuple(
                EventEvidence(
                    phase=phase,
                    captured_at=sample.captured_at,
                    image_bytes=sample.image_bytes,
                )
                for phase, sample in zip(phases, transition.evidence_samples)
            )

        try:
            await self._emit_rule_event(
                RuleEvent(
                    rule_id=self._rule.id,
                    camera_device_id=self._rule.camera.device_id,
                    status=transition.status,
                    observed_at=transition.observed_at,
                    dwell_seconds=transition.dwell_seconds,
                    entity_value=self._rule.entity_selector.value,
                    metadata=(
                        {"track_id": str(transition.track_id)}
                        if transition.track_id is not None
                        else {}
                    ),
                    evidence=evidence,
                )
            )
        except Exception as exc:  # noqa: BLE001
            self._last_error = str(exc)
            return

        self._last_error = None
        if transition.status == "threshold_met":
            self._emitted_threshold_events += 1
            self._active = True
        elif transition.status == "cleared":
            self._active = False

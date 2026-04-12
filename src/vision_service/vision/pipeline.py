import asyncio
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Awaitable, Callable, Literal

import numpy as np

from vision_service.contracts import EntityDescriptor, VisionRule
from vision_service.runtime.dwell import DwellTransition, RuleDwellTracker
from vision_service.runtime.events import EventEvidence, RuleEvent
from vision_service.settings import Settings
from vision_service.vision.backend import DetectionBatch, VisionBackend
from vision_service.vision.entities import (
    TransitionContext,
    ZoneObservation,
    build_transition_context,
    dedupe_entities,
    default_entity_for_rule,
    entity_descriptor_for_detection,
    evidence_metadata,
)
from vision_service.vision.stream import FrameStream, StreamReadResult

logger = logging.getLogger(__name__)

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
        frame_stream: FrameStream,
    ) -> None:
        self._rule = rule.model_copy(deep=True)
        self._backend = backend
        self._settings = settings
        self._emit_rule_event = emit_rule_event
        self._frame_stream = frame_stream

        self._state: WorkerState = "starting"
        self._active = False
        self._last_frame_at: datetime | None = None
        self._last_error: str | None = None
        self._emitted_threshold_events = 0
        self._default_entity = default_entity_for_rule(self._rule)
        self._current_track_entities: dict[int, EntityDescriptor] = {}
        self._current_zone_entities: tuple[EntityDescriptor, ...] = ()

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
        logger.info(
            "starting rule worker rule_id=%s camera_device_id=%s "
            "entity_value=%s stay_threshold_seconds=%s stream_url=%s",
            self._rule.id,
            self._rule.camera.device_id,
            self._rule.entity_selector.value,
            self._rule.stay_threshold_seconds,
            self._stream_url(),
        )
        self._task = asyncio.create_task(self._run_wrapper())

    async def stop(self) -> None:
        if self._task is None:
            self._state = "stopped"
            logger.info(
                "stopped rule worker rule_id=%s camera_device_id=%s",
                self._rule.id,
                self._rule.camera.device_id,
            )
            return
        self._stop_event.set()
        await self._task
        self._task = None
        logger.info(
            "stopped rule worker rule_id=%s camera_device_id=%s",
            self._rule.id,
            self._rule.camera.device_id,
        )

    async def _run_wrapper(self) -> None:
        try:
            await self._run()
        except Exception as exc:  # noqa: BLE001
            self._state = "degraded"
            self._last_error = str(exc)
            logger.exception(
                "rule worker degraded rule_id=%s camera_device_id=%s stream_url=%s",
                self._rule.id,
                self._rule.camera.device_id,
                self._stream_url(),
            )
        else:
            self._state = "stopped"

    async def _run(self) -> None:
        import supervision as sv

        tracker = sv.ByteTrack(
            lost_track_buffer=self._settings.tracker_lost_track_buffer,
        )
        dwell_tracker = RuleDwellTracker(
            threshold_seconds=self._rule.stay_threshold_seconds,
            sample_interval_seconds=self._settings.frame_sample_interval_seconds,
            max_samples=self._settings.evidence_buffer_max_samples,
        )
        self._state = "running"
        logger.info(
            "rule worker running rule_id=%s camera_device_id=%s stream_url=%s",
            self._rule.id,
            self._rule.camera.device_id,
            self._stream_url(),
        )
        last_token: int | None = None

        try:
            while not self._stop_event.is_set():
                result = await self._wait_for_stream_result(after_token=last_token)
                if result is None:
                    continue
                last_token = result.token

                if result.frame is None:
                    self._last_error = result.error
                    transition = dwell_tracker.observe(
                        observed_at=result.observed_at,
                        visible_tracks={},
                    )
                    context = build_transition_context(
                        transition=transition,
                        current_track_entities={},
                        removed_track_entities=self._current_track_entities,
                        current_entities=(),
                        default_entity=self._default_entity,
                    )
                    self._current_track_entities = {}
                    self._current_zone_entities = ()
                    if transition is not None:
                        await self._emit_transition(transition, context=context)
                    continue

                observed_at = result.observed_at
                self._last_error = None
                transition, context = await self._process_frame(
                    tracker=tracker,
                    frame=result.frame,
                    observed_at=observed_at,
                    dwell_tracker=dwell_tracker,
                )
                self._active = dwell_tracker.active
                self._last_frame_at = observed_at
                if transition is not None:
                    await self._emit_transition(transition, context=context)
                await asyncio.sleep(self._settings.idle_sleep_seconds)
        finally:
            transition = dwell_tracker.force_clear(observed_at=datetime.now(tz=UTC))
            context = build_transition_context(
                transition=transition,
                current_track_entities=self._current_track_entities,
                removed_track_entities={},
                current_entities=self._current_zone_entities,
                default_entity=self._default_entity,
            )
            if transition is not None:
                await self._emit_transition(transition, context=context)
            self._current_track_entities = {}
            self._current_zone_entities = ()
            self._active = False

    async def _wait_for_stream_result(
        self,
        *,
        after_token: int | None,
    ) -> StreamReadResult | None:
        result_task = asyncio.create_task(
            self._frame_stream.wait_for_result(after_token=after_token),
        )
        stop_task = asyncio.create_task(self._stop_event.wait())
        done, pending = await asyncio.wait(
            {result_task, stop_task},
            return_when=asyncio.FIRST_COMPLETED,
        )

        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)

        if stop_task in done:
            return None

        stop_task.cancel()
        await asyncio.gather(stop_task, return_exceptions=True)
        return await result_task

    async def _process_frame(
        self,
        *,
        tracker: Any,
        frame: np.ndarray[Any, Any],
        observed_at: datetime,
        dwell_tracker: RuleDwellTracker,
    ) -> tuple[DwellTransition | None, TransitionContext]:
        batch = await self._backend.detect(frame)
        target_detections, class_mask = self._select_target_detections(batch)
        encoded_frame = self._encode_annotated_frame(
            batch=batch,
            frame=frame,
            class_mask=class_mask,
        )
        tracked_detections = tracker.update_with_detections(target_detections)
        zone_observation = self._visible_tracks_in_zone(
            detections=tracked_detections,
            frame=frame,
            labels=batch.labels,
            encoded_frame=encoded_frame,
        )
        previous_track_entities = dict(self._current_track_entities)
        transition = dwell_tracker.observe(
            observed_at=observed_at,
            visible_tracks=zone_observation.visible_tracks,
        )
        removed_track_entities = {
            track_id: entity
            for track_id, entity in previous_track_entities.items()
            if track_id not in zone_observation.track_entities
        }
        self._current_track_entities = zone_observation.track_entities
        self._current_zone_entities = zone_observation.entities
        return transition, build_transition_context(
            transition=transition,
            current_track_entities=zone_observation.track_entities,
            removed_track_entities=removed_track_entities,
            current_entities=zone_observation.entities,
            default_entity=self._default_entity,
        )

    def _select_target_detections(
        self,
        batch: DetectionBatch,
    ) -> tuple[Any, np.ndarray[Any, Any] | None]:
        detections = batch.detections
        if (
            len(detections) == 0
            or detections.class_id is None
            or self._rule.entity_selector.value == ""
        ):
            return detections, None

        mask = np.array(
            [
                batch.labels.get(int(class_id)) == self._rule.entity_selector.value
                for class_id in detections.class_id
            ],
            dtype=bool,
        )
        return detections[mask], mask

    def _visible_tracks_in_zone(
        self,
        *,
        detections: Any,
        frame: np.ndarray[Any, Any],
        labels: dict[int, str],
        encoded_frame: bytes,
    ) -> ZoneObservation:
        if len(detections) == 0 or detections.tracker_id is None:
            return ZoneObservation(
                visible_tracks={},
                track_entities={},
                entities=(),
            )

        frame_height, frame_width = frame.shape[:2]
        zone_left = self._rule.zone.x * frame_width
        zone_top = self._rule.zone.y * frame_height
        zone_right = zone_left + (self._rule.zone.width * frame_width)
        zone_bottom = zone_top + (self._rule.zone.height * frame_height)

        visible_tracks: dict[int, bytes] = {}
        track_entities: dict[int, EntityDescriptor] = {}
        class_ids = detections.class_id

        for index, (bounding_box, tracker_id) in enumerate(
            zip(detections.xyxy, detections.tracker_id)
        ):
            if tracker_id is None:
                continue
            center_x = float((bounding_box[0] + bounding_box[2]) / 2.0)
            center_y = float((bounding_box[1] + bounding_box[3]) / 2.0)
            if zone_left <= center_x <= zone_right and zone_top <= center_y <= zone_bottom:
                track_id = int(tracker_id)
                visible_tracks[track_id] = encoded_frame
                track_entities[track_id] = entity_descriptor_for_detection(
                    class_id=(
                        int(class_ids[index])
                        if class_ids is not None
                        else None
                    ),
                    labels=labels,
                    default_entity=self._default_entity,
                )

        return ZoneObservation(
            visible_tracks=visible_tracks,
            track_entities=track_entities,
            entities=dedupe_entities(track_entities.values()),
        )

    def _encode_annotated_frame(
        self,
        *,
        batch: DetectionBatch,
        frame: np.ndarray[Any, Any],
        class_mask: np.ndarray[Any, Any] | None,
    ) -> bytes:
        plot_result = batch.result
        if class_mask is not None and getattr(batch.result, "boxes", None) is not None:
            plot_result = batch.result.new()
            plot_result.boxes = batch.result.boxes[class_mask]

        annotated_frame = plot_result.plot(
            img=frame,
            boxes=True,
            labels=True,
            masks=False,
            probs=False,
        )
        return self._encode_frame(annotated_frame)

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

    async def _emit_transition(
        self,
        transition: DwellTransition,
        *,
        context: TransitionContext,
    ) -> None:
        evidence: tuple[EventEvidence, ...] = ()
        if transition.status == "threshold_met" and transition.evidence_samples:
            phases = ("start", "middle", "end")
            evidence = tuple(
                EventEvidence(
                    phase=phase,
                    captured_at=sample.captured_at,
                    image_bytes=sample.image_bytes,
                    metadata=evidence_metadata(),
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
                    entity_value=(
                        context.primary_entity.value
                        if context.primary_entity is not None
                        else None
                    ),
                    entities=context.entities,
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
            logger.warning(
                "failed to emit transition rule_id=%s camera_device_id=%s "
                "status=%s track_id=%s error=%s",
                self._rule.id,
                self._rule.camera.device_id,
                transition.status,
                transition.track_id,
                exc,
            )
            return

        self._last_error = None
        logger.info(
            "emitted transition rule_id=%s camera_device_id=%s "
            "status=%s track_id=%s dwell_seconds=%s",
            self._rule.id,
            self._rule.camera.device_id,
            transition.status,
            transition.track_id,
            transition.dwell_seconds,
        )
        if transition.status == "threshold_met":
            self._emitted_threshold_events += 1

    def _stream_url(self) -> str:
        stream_url = getattr(self._frame_stream, "url", None)
        if isinstance(stream_url, str):
            return stream_url
        return "<unknown>"

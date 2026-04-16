import asyncio
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Awaitable, Callable, Literal

import numpy as np

from vision_service.contracts import EntityDescriptor, VisionRule
from vision_service.runtime.dwell import DwellTransition, RuleDwellTracker
from vision_service.runtime.events import RuleEvent
from vision_service.settings import Settings
from vision_service.vision.analysis import AnalyzedFrameStream
from vision_service.vision.backend import DetectionBatch
from vision_service.vision.entities import (
    ProcessedFrame,
    TransitionContext,
    ZoneObservation,
    build_transition_context,
    default_entity_for_rule,
)
from vision_service.vision.roi import ROIOccupancyDetector, ROIOccupancyObservation
from vision_service.vision.key_entity_matcher import (
    KeyEntityMatcher,
    OpenAICompatibleKeyEntityMatcher,
)
from vision_service.vision.key_entity_runtime import identify_transition_key_entity
from vision_service.vision.semantic import (
    OpenAICompatibleSemanticChecker,
    SemanticChecker,
)
from vision_service.vision.semantic_fallback import (
    SemanticFallbackTracker,
    SemanticFallbackTransition,
)
from vision_service.vision.semantic_runtime import (
    build_semantic_fallback_tracker,
    observe_semantic_fallback_safely,
)
from vision_service.vision.worker_events import (
    build_semantic_rule_event,
    build_yolo_rule_event,
)
from vision_service.vision.worker_runtime import wait_for_worker_result
from vision_service.vision.zone import (
    select_target_detections,
    visible_tracks_in_zone,
)

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
        settings: Settings,
        emit_rule_event: EmitRuleEvent,
        frame_stream: AnalyzedFrameStream,
        semantic_checker: SemanticChecker | None = None,
        key_entity_matcher: KeyEntityMatcher | None = None,
    ) -> None:
        self._rule = rule.model_copy(deep=True)
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
        self._current_track_confidences: dict[int, float] = {}
        self._current_zone_entities: tuple[EntityDescriptor, ...] = ()
        self._last_roi_observation: ROIOccupancyObservation | None = None
        self._roi_detector = (
            ROIOccupancyDetector(rule=self._rule, settings=self._settings)
            if self._settings.roi_enabled
            else None
        )
        self._semantic_checker = (
            semantic_checker
            if semantic_checker is not None
            else OpenAICompatibleSemanticChecker.from_settings(self._settings)
        )
        self._key_entity_matcher = (
            key_entity_matcher
            if key_entity_matcher is not None
            else OpenAICompatibleKeyEntityMatcher.from_settings(self._settings)
        )

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
        semantic_fallback = build_semantic_fallback_tracker(
            rule=self._rule,
            settings=self._settings,
            checker=self._semantic_checker,
            roi_enabled=self._roi_detector is not None,
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
                result = await wait_for_worker_result(
                    frame_stream=self._frame_stream,
                    stop_event=self._stop_event,
                    after_token=last_token,
                    require_detection=self._should_request_detection(),
                )
                if result is None:
                    continue
                last_token = result.token

                if result.frame is None:
                    self._last_error = result.error
                    yolo_was_active = dwell_tracker.active
                    transition = dwell_tracker.observe(
                        observed_at=result.observed_at,
                        visible_tracks={},
                    )
                    semantic_transition = (
                        semantic_fallback.force_clear(
                            observed_at=result.observed_at,
                            yolo_threshold_observed=(
                                yolo_was_active or transition is not None
                            ),
                        )
                        if semantic_fallback is not None
                        else None
                    )
                    context = build_transition_context(
                        transition=transition,
                        current_track_entities={},
                        removed_track_entities=self._current_track_entities,
                        current_track_confidences={},
                        removed_track_confidences=self._current_track_confidences,
                        current_entities=(),
                        default_entity=self._default_entity,
                    )
                    self._current_track_entities = {}
                    self._current_track_confidences = {}
                    self._current_zone_entities = ()
                    self._last_roi_observation = None
                    if transition is not None:
                        await self._emit_transition(transition, context=context)
                    if semantic_transition is not None:
                        await self._emit_semantic_transition(semantic_transition)
                    continue

                observed_at = result.observed_at
                self._last_error = None
                processed = await self._process_frame(
                    tracker=tracker,
                    frame=result.frame,
                    batch=result.batch,
                    observed_at=observed_at,
                    dwell_tracker=dwell_tracker,
                )
                semantic_transition, semantic_error = await observe_semantic_fallback_safely(
                    semantic_fallback=semantic_fallback,
                    rule=self._rule,
                    settings=self._settings,
                    frame=result.frame,
                    observed_at=observed_at,
                    processed=processed,
                    yolo_threshold_observed=(
                        dwell_tracker.active or processed.transition is not None
                    ),
                )
                if semantic_error is not None:
                    self._last_error = semantic_error
                    logger.warning(
                        "semantic checker failed rule_id=%s camera_device_id=%s error=%s",
                        self._rule.id,
                        self._rule.camera.device_id,
                        semantic_error,
                    )
                self._active = dwell_tracker.active or (
                    semantic_fallback.active if semantic_fallback is not None else False
                )
                self._last_frame_at = observed_at
                if processed.transition is not None:
                    await self._emit_transition(
                        processed.transition,
                        context=processed.context,
                    )
                if semantic_transition is not None:
                    await self._emit_semantic_transition(semantic_transition)
                await asyncio.sleep(self._settings.idle_sleep_seconds)
        finally:
            yolo_was_active = dwell_tracker.active
            transition = dwell_tracker.force_clear(observed_at=datetime.now(tz=UTC))
            semantic_transition = (
                semantic_fallback.force_clear(
                    observed_at=datetime.now(tz=UTC),
                    yolo_threshold_observed=(yolo_was_active or transition is not None),
                )
                if semantic_fallback is not None
                else None
            )
            context = build_transition_context(
                transition=transition,
                current_track_entities=self._current_track_entities,
                removed_track_entities={},
                current_track_confidences=self._current_track_confidences,
                removed_track_confidences={},
                current_entities=self._current_zone_entities,
                default_entity=self._default_entity,
            )
            if transition is not None:
                await self._emit_transition(transition, context=context)
            if semantic_transition is not None:
                await self._emit_semantic_transition(semantic_transition)
            self._current_track_entities = {}
            self._current_track_confidences = {}
            self._current_zone_entities = ()
            self._last_roi_observation = None
            self._active = False

    async def _process_frame(
        self,
        *,
        tracker: Any,
        frame: np.ndarray[Any, Any],
        batch: DetectionBatch | None,
        observed_at: datetime,
        dwell_tracker: RuleDwellTracker,
    ) -> ProcessedFrame:
        previous_track_entities = dict(self._current_track_entities)
        previous_track_confidences = dict(self._current_track_confidences)
        roi_observation = (
            self._roi_detector.observe(frame=frame, observed_at=observed_at)
            if self._roi_detector is not None
            else None
        )
        zone_observation = ZoneObservation(
            visible_tracks={},
            track_entities={},
            track_confidences={},
            entities=(),
        )
        if batch is not None:
            target_detections, _class_mask = select_target_detections(
                batch=batch,
                entity_value=self._rule.entity_selector.value,
            )
            tracked_detections = tracker.update_with_detections(target_detections)
            zone_observation = visible_tracks_in_zone(
                rule=self._rule,
                detections=tracked_detections,
                frame=frame,
                labels=batch.labels,
                default_entity=self._default_entity,
                jpeg_quality=self._settings.jpeg_quality,
            )
        transition = dwell_tracker.observe(
            observed_at=observed_at,
            visible_tracks=zone_observation.visible_tracks,
        )
        removed_track_entities = {
            track_id: entity
            for track_id, entity in previous_track_entities.items()
            if track_id not in zone_observation.track_entities
        }
        removed_track_confidences = {
            track_id: confidence
            for track_id, confidence in previous_track_confidences.items()
            if track_id not in zone_observation.track_confidences
        }
        self._current_track_entities = zone_observation.track_entities
        self._current_track_confidences = zone_observation.track_confidences
        self._current_zone_entities = zone_observation.entities
        self._last_roi_observation = roi_observation
        return ProcessedFrame(
            transition=transition,
            context=build_transition_context(
                transition=transition,
                current_track_entities=zone_observation.track_entities,
                current_track_confidences=zone_observation.track_confidences,
                removed_track_entities=removed_track_entities,
                removed_track_confidences=removed_track_confidences,
                current_entities=zone_observation.entities,
                default_entity=self._default_entity,
            ),
            zone_observation=zone_observation,
            roi_observation=roi_observation,
        )

    def _should_request_detection(self) -> bool:
        if self._settings.yolo_run_mode == "always" or self._roi_detector is None:
            return True
        return self._roi_detector.current_state in {"candidate_occupied", "occupied"}

    async def _emit_transition(
        self,
        transition: DwellTransition,
        *,
        context: TransitionContext,
    ) -> None:
        key_entity_identification = await identify_transition_key_entity(
            transition=transition,
            rule=self._rule,
            matcher=self._key_entity_matcher,
        )
        try:
            event = build_yolo_rule_event(
                rule=self._rule,
                transition=transition,
                context=context,
                roi_observation=self._last_roi_observation,
                key_entity_identification=key_entity_identification,
            )
            await self._emit_rule_event(event)
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

        if (
            key_entity_identification is not None
            and key_entity_identification.error_message is not None
        ):
            self._last_error = key_entity_identification.error_message
            logger.warning(
                "key entity match degraded rule_id=%s camera_device_id=%s error=%s",
                self._rule.id,
                self._rule.camera.device_id,
                key_entity_identification.error_message,
            )
        else:
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

    async def _emit_semantic_transition(
        self,
        transition: SemanticFallbackTransition,
    ) -> None:
        try:
            await self._emit_rule_event(
                build_semantic_rule_event(
                    rule=self._rule,
                    default_entity=self._default_entity,
                    transition=transition,
                )
            )
        except Exception as exc:  # noqa: BLE001
            self._last_error = str(exc)
            logger.warning(
                "failed to emit semantic fallback rule_id=%s camera_device_id=%s "
                "error=%s",
                self._rule.id,
                self._rule.camera.device_id,
                exc,
            )
            return

        self._last_error = None
        self._emitted_threshold_events += 1
        logger.info(
            "emitted semantic fallback rule_id=%s camera_device_id=%s "
            "dwell_seconds=%s verdict=%s confidence_score=%s",
            self._rule.id,
            self._rule.camera.device_id,
            transition.dwell_seconds,
            transition.semantic_result.verdict,
            transition.confidence.score,
        )

    def _stream_url(self) -> str:
        stream_url = self._rule.rtsp_source.url
        if isinstance(stream_url, str) and stream_url:
            return stream_url
        stream_url = getattr(self._frame_stream, "url", None)
        if isinstance(stream_url, str):
            return stream_url
        return "<unknown>"

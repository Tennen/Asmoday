from datetime import UTC, datetime
import numpy as np
import pytest
import supervision as sv

from vision_service.contracts import (
    CameraIdentity,
    EntityDescriptor,
    EvidenceDetection,
    EntitySelector,
    KeyEntityImage,
    KeyEntityReference,
    NormalizedBoundingBox,
    RTSPSource,
    VisionRule,
    ZoneRect,
)
from vision_service.runtime.dwell import DwellTransition, EvidenceSample, TrackEvidence
from vision_service.settings import Settings
from vision_service.vision.backend import DetectionBatch
from vision_service.vision.entities import TransitionContext
from vision_service.vision.key_entity_matcher import KeyEntityFrameMatch
from vision_service.vision.pipeline import RuleVisionWorker
from vision_service.vision.roi.models import ROIOccupancyObservation
from vision_service.vision.zone import (
    encode_annotated_frame,
    select_target_detections,
    visible_tracks_in_zone,
)


class DummyStream:
    url = "rtsp://camera/test"


class FakeResult:
    def __init__(self, boxes: np.ndarray) -> None:
        self.boxes = boxes
        self.clone: FakeResult | None = None
        self.last_plot_kwargs: dict[str, object] | None = None

    def new(self) -> "FakeResult":
        clone = FakeResult(self.boxes.copy())
        self.clone = clone
        return clone

    def plot(self, **kwargs):  # noqa: ANN003, ANN201
        self.last_plot_kwargs = kwargs
        return np.zeros((4, 4, 3), dtype=np.uint8)


def build_rule(
    *,
    entity_value: str,
    key_entities: list[KeyEntityReference] | None = None,
) -> VisionRule:
    return VisionRule(
        id="rule-1",
        name="Rule 1",
        enabled=True,
        camera=CameraIdentity(device_id="camera-1"),
        rtsp_source=RTSPSource(url="rtsp://camera/test"),
        entity_selector=EntitySelector(value=entity_value),
        key_entities=key_entities or [],
        zone=ZoneRect(x=0.1, y=0.1, width=0.8, height=0.8),
        stay_threshold_seconds=5,
    )


def build_worker(
    *,
    entity_value: str,
    emit_rule_event,
    frame_stream: object | None = None,
    settings: Settings | None = None,
    key_entities: list[KeyEntityReference] | None = None,
    key_entity_matcher: object | None = None,
) -> RuleVisionWorker:
    return RuleVisionWorker(
        rule=build_rule(entity_value=entity_value, key_entities=key_entities),
        settings=settings or Settings(),
        emit_rule_event=emit_rule_event,
        frame_stream=frame_stream or DummyStream(),
        key_entity_matcher=key_entity_matcher,
    )


def test_select_target_detections_skips_class_filter_for_wildcard_rule() -> None:
    async def emit_rule_event(event):  # noqa: ANN001, ANN202
        return "unused"

    worker = build_worker(entity_value="", emit_rule_event=emit_rule_event)
    detections = sv.Detections(
        xyxy=np.array([[1, 1, 2, 2], [2, 2, 3, 3]], dtype=np.float32),
        confidence=np.array([0.9, 0.8], dtype=np.float32),
        class_id=np.array([0, 1], dtype=np.int32),
    )
    batch = DetectionBatch(
        result=FakeResult(np.array(["cat-box", "dog-box"], dtype=object)),
        detections=detections,
        labels={0: "cat", 1: "dog"},
    )

    selected, mask = select_target_detections(
        batch=batch,
        entity_value=worker._rule.entity_selector.value,
    )

    assert selected is detections
    assert mask is None


def test_encode_annotated_frame_filters_result_boxes_before_plot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def emit_rule_event(event):  # noqa: ANN001, ANN202
        return "unused"

    worker = build_worker(entity_value="cat", emit_rule_event=emit_rule_event)
    fake_result = FakeResult(np.array(["cat-box", "dog-box"], dtype=object))
    batch = DetectionBatch(
        result=fake_result,
        detections=sv.Detections(
            xyxy=np.array([[1, 1, 2, 2], [2, 2, 3, 3]], dtype=np.float32),
            confidence=np.array([0.9, 0.8], dtype=np.float32),
            class_id=np.array([0, 1], dtype=np.int32),
        ),
        labels={0: "cat", 1: "dog"},
    )
    monkeypatch.setattr(
        "vision_service.vision.zone.encode_frame",
        lambda *, frame, jpeg_quality: b"annotated",
    )

    image_bytes = encode_annotated_frame(
        batch=batch,
        frame=np.zeros((4, 4, 3), dtype=np.uint8),
        class_mask=np.array([True, False], dtype=bool),
        jpeg_quality=worker._settings.jpeg_quality,
    )

    assert image_bytes == b"annotated"
    assert fake_result.clone is not None
    assert fake_result.clone.boxes.tolist() == ["cat-box"]
    assert fake_result.clone.last_plot_kwargs is not None
    assert isinstance(fake_result.clone.last_plot_kwargs["img"], np.ndarray)
    assert fake_result.clone.last_plot_kwargs["boxes"] is True
    assert fake_result.clone.last_plot_kwargs["labels"] is True
    assert fake_result.clone.last_plot_kwargs["masks"] is False
    assert fake_result.clone.last_plot_kwargs["probs"] is False


def test_visible_tracks_in_zone_skips_encoding_when_no_detection_is_in_zone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def emit_rule_event(event):  # noqa: ANN001, ANN202
        return "unused"

    worker = build_worker(entity_value="cat", emit_rule_event=emit_rule_event)
    monkeypatch.setattr(
        "vision_service.vision.zone.encode_frame",
        lambda **kwargs: pytest.fail("encoding should be skipped"),
    )
    detections = sv.Detections(
        xyxy=np.array([[0, 0, 1, 1]], dtype=np.float32),
        confidence=np.array([0.9], dtype=np.float32),
        class_id=np.array([0], dtype=np.int32),
        tracker_id=np.array([7], dtype=np.int32),
    )
    batch = DetectionBatch(
        result=FakeResult(np.array(["cat-box"], dtype=object)),
        detections=detections,
        labels={0: "cat"},
    )

    observation = visible_tracks_in_zone(
        rule=worker._rule,
        detections=detections,
        frame=np.zeros((10, 10, 3), dtype=np.uint8),
        labels=batch.labels,
        default_entity=worker._default_entity,
        jpeg_quality=worker._settings.jpeg_quality,
    )

    assert observation.visible_tracks == {}
    assert observation.track_entities == {}
    assert observation.entities == ()


def test_stream_url_falls_back_to_rule_rtsp_source_when_frame_stream_has_no_url() -> None:
    async def emit_rule_event(event):  # noqa: ANN001, ANN202
        return "unused"

    class StreamWithoutUrl:
        async def wait_for_result(self, *, after_token: int | None):  # noqa: ANN202
            return None

    worker = build_worker(
        entity_value="cat",
        emit_rule_event=emit_rule_event,
        frame_stream=StreamWithoutUrl(),
    )

    assert worker._stream_url() == "rtsp://camera/test"


def test_roi_triggered_mode_only_requests_detection_when_roi_is_active() -> None:
    async def emit_rule_event(event):  # noqa: ANN001, ANN202
        return "unused"

    worker = build_worker(
        entity_value="cat",
        emit_rule_event=emit_rule_event,
        settings=Settings(roi_enabled=True, yolo_run_mode="roi_triggered"),
    )

    assert worker._should_request_detection() is False

    class FakeDetector:
        current_state = "candidate_occupied"

    worker._roi_detector = FakeDetector()

    assert worker._should_request_detection() is True


@pytest.mark.asyncio
async def test_process_frame_does_not_reuse_last_confirmed_tracks_when_detection_is_missing() -> None:
    async def emit_rule_event(event):  # noqa: ANN001, ANN202
        return "unused"

    worker = build_worker(
        entity_value="cat",
        emit_rule_event=emit_rule_event,
        settings=Settings(roi_enabled=True),
    )
    entity = EntityDescriptor(kind="label", value="cat", display_name="Cat")
    worker._current_track_entities = {7: entity}
    worker._current_track_confidences = {7: 0.91}
    worker._current_zone_entities = (entity,)

    class FakeDetector:
        def observe(self, *, frame, observed_at):  # noqa: ANN001, ANN201
            return ROIOccupancyObservation(
                observed_at=observed_at,
                state="candidate_occupied",
                frame_present=True,
                occupancy_ratio=0.12,
                largest_blob_area=512,
                roi_area_pixels=1000,
                foreground_pixels=120,
            )

    worker._roi_detector = FakeDetector()

    class FakeDwellTracker:
        def __init__(self) -> None:
            self.visible_tracks: dict[int, TrackEvidence] | None = None

        def observe(self, *, observed_at, visible_tracks):  # noqa: ANN001, ANN201
            self.visible_tracks = visible_tracks
            return None

    dwell_tracker = FakeDwellTracker()

    processed = await worker._process_frame(
        tracker=object(),
        frame=np.zeros((4, 4, 3), dtype=np.uint8),
        batch=None,
        observed_at=datetime(2026, 4, 12, 8, 0, tzinfo=UTC),
        dwell_tracker=dwell_tracker,  # type: ignore[arg-type]
    )

    assert processed.transition is None
    assert processed.context == TransitionContext()
    assert dwell_tracker.visible_tracks == {}
    assert worker._current_track_entities == {}
    assert worker._current_track_confidences == {}
    assert worker._current_zone_entities == ()


@pytest.mark.asyncio
async def test_emit_transition_includes_entities_and_annotation_metadata() -> None:
    emitted_events = []

    async def emit_rule_event(event):  # noqa: ANN001, ANN202
        emitted_events.append(event)
        return "vision-evt-1"

    worker = build_worker(entity_value="", emit_rule_event=emit_rule_event)
    transition = DwellTransition(
        status="threshold_met",
        observed_at=datetime(2026, 4, 12, 8, 0, tzinfo=UTC),
        dwell_seconds=5,
        track_id=7,
        evidence_samples=(
            EvidenceSample(
                captured_at=datetime(2026, 4, 12, 7, 59, 58, tzinfo=UTC),
                image_bytes=b"start",
                detections=(
                    EvidenceDetection(
                        kind="label",
                        value="dog",
                        display_name="Dog",
                        confidence=0.94,
                        track_id="7",
                        box=NormalizedBoundingBox(
                            x=0.1,
                            y=0.2,
                            width=0.3,
                            height=0.4,
                        ),
                    ),
                ),
                crop_bytes=b"crop-start",
            ),
            EvidenceSample(
                captured_at=datetime(2026, 4, 12, 8, 0, tzinfo=UTC),
                image_bytes=b"middle",
                detections=(
                    EvidenceDetection(
                        kind="label",
                        value="dog",
                        display_name="Dog",
                        confidence=0.95,
                        track_id="7",
                        box=NormalizedBoundingBox(
                            x=0.11,
                            y=0.21,
                            width=0.31,
                            height=0.41,
                        ),
                    ),
                ),
                crop_bytes=b"crop-middle",
            ),
            EvidenceSample(
                captured_at=datetime(2026, 4, 12, 8, 0, 2, tzinfo=UTC),
                image_bytes=b"end",
                detections=(
                    EvidenceDetection(
                        kind="label",
                        value="dog",
                        display_name="Dog",
                        confidence=0.91,
                        track_id="7",
                        box=NormalizedBoundingBox(
                            x=0.12,
                            y=0.22,
                            width=0.32,
                            height=0.42,
                        ),
                    ),
                ),
                crop_bytes=b"crop-end",
            ),
        ),
    )

    await worker._emit_transition(
        transition,
        context=TransitionContext(
            primary_entity=EntityDescriptor(
                kind="label",
                value="dog",
                display_name="Dog",
            ),
            entities=(
                EntityDescriptor(kind="label", value="dog", display_name="Dog"),
                EntityDescriptor(kind="label", value="cat", display_name="Cat"),
            ),
        ),
    )

    assert len(emitted_events) == 1
    event = emitted_events[0]
    assert event.entity_value == "dog"
    assert [entity.value for entity in event.entities] == ["dog", "cat"]
    assert all(
        capture.metadata["annotations"]["image_kind"] == "raw"
        for capture in event.evidence
    )
    assert all(
        capture.metadata["annotations"]["source"] == "ultralytics.boxes"
        for capture in event.evidence
    )
    assert event.evidence[0].metadata["annotations"]["detections"][0]["track_id"] == "7"


@pytest.mark.asyncio
async def test_emit_transition_includes_key_entity_vote_result() -> None:
    emitted_events = []

    async def emit_rule_event(event):  # noqa: ANN001, ANN202
        emitted_events.append(event)
        return "vision-evt-2"

    class FakeKeyEntityMatcher:
        def __init__(self) -> None:
            self.calls = 0

        async def match(self, *, image_bytes: bytes, key_entities):  # noqa: ANN001, ANN201
            self.calls += 1
            assert image_bytes.startswith(b"crop-")
            assert len(key_entities) == 2
            return KeyEntityFrameMatch(
                key_entity_id=101,
                confidence=0.88,
                reason="花纹一致",
                raw_output='{"key_entity_id":101}',
                model_name="mini-vlm",
                checked_at=datetime(2026, 4, 12, 8, 0, tzinfo=UTC),
            )

    matcher = FakeKeyEntityMatcher()
    worker = build_worker(
        entity_value="dog",
        emit_rule_event=emit_rule_event,
        key_entities=[
            KeyEntityReference(
                id=101,
                image=KeyEntityImage(base64="aW1hZ2U="),
            ),
            KeyEntityReference(
                id=102,
                description="黑色项圈",
            ),
        ],
        key_entity_matcher=matcher,
    )
    transition = DwellTransition(
        status="threshold_met",
        observed_at=datetime(2026, 4, 12, 8, 0, tzinfo=UTC),
        dwell_seconds=5,
        track_id=7,
        evidence_samples=(
            EvidenceSample(
                captured_at=datetime(2026, 4, 12, 7, 59, 58, tzinfo=UTC),
                image_bytes=b"start",
                crop_bytes=b"crop-start",
            ),
            EvidenceSample(
                captured_at=datetime(2026, 4, 12, 8, 0, tzinfo=UTC),
                image_bytes=b"middle",
                crop_bytes=b"crop-middle",
            ),
            EvidenceSample(
                captured_at=datetime(2026, 4, 12, 8, 0, 2, tzinfo=UTC),
                image_bytes=b"end",
                crop_bytes=b"crop-end",
            ),
        ),
    )

    await worker._emit_transition(
        transition,
        context=TransitionContext(
            primary_entity=EntityDescriptor(
                kind="label",
                value="dog",
                display_name="Dog",
            ),
        ),
    )

    assert matcher.calls == 3
    assert emitted_events[0].key_entity_id == 101
    assert emitted_events[0].metadata["key_entity_match"]["winner_id"] == 101
    assert emitted_events[0].metadata["key_entity_match"]["status"] == "matched"


@pytest.mark.asyncio
async def test_emit_transition_skips_key_entity_without_yolo_crop() -> None:
    emitted_events = []

    async def emit_rule_event(event):  # noqa: ANN001, ANN202
        emitted_events.append(event)
        return "vision-evt-3"

    class FailingKeyEntityMatcher:
        async def match(self, *, image_bytes: bytes, key_entities):  # noqa: ANN001, ANN201
            raise AssertionError("key entity matcher should not be called")

    worker = build_worker(
        entity_value="dog",
        emit_rule_event=emit_rule_event,
        key_entities=[KeyEntityReference(id=101, description="橘猫")],
        key_entity_matcher=FailingKeyEntityMatcher(),
    )
    transition = DwellTransition(
        status="threshold_met",
        observed_at=datetime(2026, 4, 12, 8, 0, tzinfo=UTC),
        dwell_seconds=5,
        track_id=7,
        evidence_samples=(
            EvidenceSample(
                captured_at=datetime(2026, 4, 12, 7, 59, 58, tzinfo=UTC),
                image_bytes=b"start",
            ),
        ),
    )

    await worker._emit_transition(
        transition,
        context=TransitionContext(
            primary_entity=EntityDescriptor(
                kind="label",
                value="dog",
                display_name="Dog",
            ),
        ),
    )

    assert len(emitted_events) == 1
    assert emitted_events[0].key_entity_id is None
    assert "key_entity_match" not in emitted_events[0].metadata

from datetime import UTC, datetime
import numpy as np
import pytest
import supervision as sv

from vision_service.contracts import (
    CameraIdentity,
    EntityDescriptor,
    EntitySelector,
    RTSPSource,
    VisionRule,
    ZoneRect,
)
from vision_service.runtime.dwell import DwellTransition, EvidenceSample
from vision_service.settings import Settings
from vision_service.vision.backend import DetectionBatch
from vision_service.vision.entities import TransitionContext
from vision_service.vision.pipeline import RuleVisionWorker


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


def build_rule(*, entity_value: str) -> VisionRule:
    return VisionRule(
        id="rule-1",
        name="Rule 1",
        enabled=True,
        camera=CameraIdentity(device_id="camera-1"),
        rtsp_source=RTSPSource(url="rtsp://camera/test"),
        entity_selector=EntitySelector(value=entity_value),
        zone=ZoneRect(x=0.1, y=0.1, width=0.8, height=0.8),
        stay_threshold_seconds=5,
    )


def build_worker(
    *,
    entity_value: str,
    emit_rule_event,
    frame_stream: object | None = None,
) -> RuleVisionWorker:
    return RuleVisionWorker(
        rule=build_rule(entity_value=entity_value),
        settings=Settings(),
        emit_rule_event=emit_rule_event,
        frame_stream=frame_stream or DummyStream(),
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

    selected, mask = worker._select_target_detections(batch)

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
    monkeypatch.setattr(worker, "_encode_frame", lambda frame: b"annotated")

    image_bytes = worker._encode_annotated_frame(
        batch=batch,
        frame=np.zeros((4, 4, 3), dtype=np.uint8),
        class_mask=np.array([True, False], dtype=bool),
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
        worker,
        "_encode_annotated_frame",
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

    observation = worker._visible_tracks_in_zone(
        detections=detections,
        frame=np.zeros((10, 10, 3), dtype=np.uint8),
        labels=batch.labels,
        batch=batch,
        class_mask=np.array([True], dtype=bool),
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
            ),
            EvidenceSample(
                captured_at=datetime(2026, 4, 12, 8, 0, tzinfo=UTC),
                image_bytes=b"middle",
            ),
            EvidenceSample(
                captured_at=datetime(2026, 4, 12, 8, 0, 2, tzinfo=UTC),
                image_bytes=b"end",
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
        capture.metadata == {
            "annotations": {
                "image_kind": "annotated",
                "source": "ultralytics.plot",
            }
        }
        for capture in event.evidence
    )

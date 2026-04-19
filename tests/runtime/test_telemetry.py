from datetime import UTC, datetime

from vision_service.contracts import EntityDescriptor
from vision_service.runtime.events import EventEvidence, RuleEvent
from vision_service.runtime.telemetry import (
    build_evidence_callback_payload,
    build_event_callback_payload,
)


def test_build_event_callback_payload_includes_entities() -> None:
    observed_at = datetime(2026, 4, 12, 8, 0, tzinfo=UTC)
    event = RuleEvent(
        rule_id="feeder-zone",
        camera_device_id="camera-1",
        status="threshold_met",
        observed_at=observed_at,
        dwell_seconds=5,
        entity_value="cat",
        key_entity_id=101,
        entities=(
            EntityDescriptor(kind="label", value="cat", display_name="Cat"),
            EntityDescriptor(kind="label", value="dog", display_name="Dog"),
        ),
        metadata={"track_id": "7"},
    )

    payload = build_event_callback_payload(
        event=event,
        event_id="vision-evt-1",
    )

    record = payload.events[0]
    assert record.event_id == "vision-evt-1"
    assert record.entity_value == "cat"
    assert record.key_entity_id == 101
    assert [entity.value for entity in record.entities or []] == ["cat", "dog"]
    assert record.metadata == {"track_id": "7"}


def test_build_evidence_callback_payload_includes_annotation_metadata() -> None:
    captured_at = datetime(2026, 4, 12, 8, 0, tzinfo=UTC)
    event = RuleEvent(
        rule_id="feeder-zone",
        camera_device_id="camera-1",
        status="threshold_met",
        observed_at=captured_at,
        dwell_seconds=5,
        evidence=(
            EventEvidence(
                phase="sample_002",
                captured_at=captured_at,
                image_bytes=b"image",
                metadata={
                    "annotations": {
                        "image_kind": "raw",
                        "coordinate_space": "normalized_xywh",
                        "source": "ultralytics.boxes",
                        "detections": [],
                    }
                },
            ),
        ),
    )

    payload = build_evidence_callback_payload(
        event=event,
        event_id="vision-evt-1",
    )

    assert payload is not None
    capture = payload.captures[0]
    assert capture.capture_id == "vision-evt-1:sample_002"
    assert capture.phase == "sample_002"
    assert capture.image_base64 == "aW1hZ2U="
    assert capture.metadata == {
        "annotations": {
            "image_kind": "raw",
            "coordinate_space": "normalized_xywh",
            "source": "ultralytics.boxes",
            "detections": [],
        }
    }

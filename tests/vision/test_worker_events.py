from datetime import UTC, datetime

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
from vision_service.runtime.dwell import DwellTransition, EvidenceSample
from vision_service.vision.confidence import ConfidenceAssessment
from vision_service.vision.entities import TransitionContext
from vision_service.vision.roi.models import ROIOccupancyObservation
from vision_service.vision.semantic import SemanticCheckResult
from vision_service.vision.key_entity_matcher import KeyEntityIdentification
from vision_service.vision.semantic_fallback import (
    SemanticFallbackTransition,
    SemanticVoteSummary,
)
from vision_service.vision.worker_events import (
    build_semantic_rule_event,
    build_yolo_rule_event,
)


def build_rule() -> VisionRule:
    return VisionRule(
        id="rule-1",
        name="Rule 1",
        enabled=True,
        camera=CameraIdentity(device_id="camera-1"),
        rtsp_source=RTSPSource(url="rtsp://camera/stream"),
        entity_selector=EntitySelector(value="cat"),
        behavior="进食",
        key_entities=[
            KeyEntityReference(
                id=101,
                image=KeyEntityImage(base64="aW1hZ2U="),
            )
        ],
        zone=ZoneRect(x=0.1, y=0.1, width=0.2, height=0.2),
        stay_threshold_seconds=5,
    )


def build_samples() -> tuple[EvidenceSample, ...]:
    return (
        EvidenceSample(
            captured_at=datetime(2026, 4, 13, 8, 0, tzinfo=UTC),
            image_bytes=b"start",
            detections=(
                EvidenceDetection(
                    kind="label",
                    value="cat",
                    display_name="Cat",
                    confidence=0.93,
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
            captured_at=datetime(2026, 4, 13, 8, 0, 2, tzinfo=UTC),
            image_bytes=b"middle",
            detections=(
                EvidenceDetection(
                    kind="label",
                    value="cat",
                    display_name="Cat",
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
            captured_at=datetime(2026, 4, 13, 8, 0, 4, tzinfo=UTC),
            image_bytes=b"end",
            detections=(
                EvidenceDetection(
                    kind="label",
                    value="cat",
                    display_name="Cat",
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
    )


def test_build_yolo_rule_event_includes_confidence_metadata() -> None:
    rule = build_rule()
    event = build_yolo_rule_event(
        rule=rule,
        transition=DwellTransition(
            status="threshold_met",
            observed_at=datetime(2026, 4, 13, 8, 0, 5, tzinfo=UTC),
            dwell_seconds=7,
            track_id=7,
            evidence_samples=build_samples(),
        ),
        context=TransitionContext(
            primary_entity=EntityDescriptor(
                kind="label",
                value="cat",
                display_name="Cat",
            ),
            primary_confidence=0.93,
            entities=(
                EntityDescriptor(kind="label", value="cat", display_name="Cat"),
            ),
        ),
        roi_observation=ROIOccupancyObservation(
            observed_at=datetime(2026, 4, 13, 8, 0, 5, tzinfo=UTC),
            state="occupied",
            frame_present=True,
            occupancy_ratio=0.18,
            largest_blob_area=640,
            roi_area_pixels=1000,
            foreground_pixels=180,
        ),
        key_entity_identification=KeyEntityIdentification(
            key_entity_id=101,
            metadata={
                "status": "matched",
                "winner_id": 101,
            },
        ),
    )

    assert event.key_entity_id == 101
    assert event.metadata["track_id"] == "7"
    assert event.metadata["decision"]["source"] == "yolo_track"
    assert event.metadata["key_entity_match"]["winner_id"] == 101
    assert event.metadata["decision"]["confidence_score"] >= 0.8
    assert len(event.evidence) == 3
    assert event.evidence[0].metadata["annotations"]["image_kind"] == "raw"
    assert event.evidence[0].metadata["annotations"]["detections"][0]["track_id"] == "7"


def test_build_semantic_rule_event_includes_semantic_metadata() -> None:
    rule = build_rule()
    default_entity = EntityDescriptor(kind="label", value="cat", display_name="Cat")
    event = build_semantic_rule_event(
        rule=rule,
        default_entity=default_entity,
        transition=SemanticFallbackTransition(
            observed_at=datetime(2026, 4, 13, 8, 1, tzinfo=UTC),
            dwell_seconds=8,
            semantic_result=SemanticCheckResult(
                verdict="疑似有",
                raw_output="疑似有",
                model_name="mini-vlm",
                checked_at=datetime(2026, 4, 13, 8, 0, 30, tzinfo=UTC),
            ),
            vote_summary=SemanticVoteSummary(
                attempts=3,
                positive_votes=2,
                verdicts=("无法确定", "疑似有", "有"),
            ),
            confidence=ConfidenceAssessment(
                source="roi_vlm_fallback",
                score=0.62,
                breakdown={"vlm": 0.64, "roi": 0.73, "yolo": 0.18},
            ),
            consecutive_yolo_failures=6,
            yolo_support_confidence=None,
            evidence_samples=build_samples(),
        ),
    )

    assert event.entity_value == "cat"
    assert event.metadata["decision"]["source"] == "roi_vlm_fallback"
    assert event.metadata["decision"]["semantic_check"]["verdict"] == "疑似有"
    assert event.metadata["decision"]["semantic_check"]["positive_votes"] == 2
    assert event.evidence[0].metadata["annotations"]["source"] == "roi.zone_crop"

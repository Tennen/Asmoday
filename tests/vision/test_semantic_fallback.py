from datetime import UTC, datetime, timedelta

import pytest

from vision_service.contracts import (
    CameraIdentity,
    EntitySelector,
    RTSPSource,
    VisionRule,
    ZoneRect,
)
from vision_service.vision.semantic import SemanticCheckResult
from vision_service.vision.semantic_fallback import SemanticFallbackTracker
from vision_service.vision.roi.models import ROIOccupancyObservation


class FakeChecker:
    def __init__(self, verdict: str) -> None:
        self._verdict = verdict
        self.calls = 0

    async def check(self, *, image_bytes: bytes, rule: VisionRule) -> SemanticCheckResult:
        self.calls += 1
        return SemanticCheckResult(
            verdict=self._verdict,  # type: ignore[arg-type]
            raw_output=self._verdict,
            model_name="mini-vlm",
            checked_at=datetime(2026, 4, 13, 8, 0, tzinfo=UTC),
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
        zone=ZoneRect(x=0.1, y=0.1, width=0.2, height=0.2),
        stay_threshold_seconds=5,
    )


def occupied_roi(observed_at: datetime) -> ROIOccupancyObservation:
    return ROIOccupancyObservation(
        observed_at=observed_at,
        state="occupied",
        frame_present=True,
        occupancy_ratio=0.24,
        largest_blob_area=820,
        roi_area_pixels=1000,
        foreground_pixels=240,
    )


@pytest.mark.asyncio
async def test_semantic_fallback_emits_transition_for_confirmed_roi_episode() -> None:
    checker = FakeChecker("有")
    tracker = SemanticFallbackTracker(
        rule=build_rule(),
        threshold_seconds=5,
        sample_interval_seconds=1.0,
        max_samples=9,
        consecutive_yolo_failures=2,
        retry_cooldown_seconds=1.0,
        max_attempts_per_episode=2,
        checker=checker,
    )
    start = datetime(2026, 4, 13, 8, 0, tzinfo=UTC)

    for second in range(6):
        transition = await tracker.observe(
            observed_at=start + timedelta(seconds=second),
            roi_observation=occupied_roi(start + timedelta(seconds=second)),
            image_bytes=f"frame-{second}".encode(),
            yolo_confidence=None,
            yolo_threshold_observed=False,
        )
        assert transition is None

    completed = tracker.force_clear(observed_at=start + timedelta(seconds=7))

    assert completed is not None
    assert completed.dwell_seconds == 5
    assert completed.semantic_result.verdict == "有"
    assert completed.confidence.source == "roi_vlm_fallback"
    assert len(completed.evidence_samples) == 3
    assert checker.calls == 1


@pytest.mark.asyncio
async def test_semantic_fallback_is_suppressed_after_yolo_threshold_observed() -> None:
    checker = FakeChecker("有")
    tracker = SemanticFallbackTracker(
        rule=build_rule(),
        threshold_seconds=5,
        sample_interval_seconds=1.0,
        max_samples=9,
        consecutive_yolo_failures=1,
        retry_cooldown_seconds=1.0,
        max_attempts_per_episode=2,
        checker=checker,
    )
    observed_at = datetime(2026, 4, 13, 8, 10, tzinfo=UTC)

    await tracker.observe(
        observed_at=observed_at,
        roi_observation=occupied_roi(observed_at),
        image_bytes=b"frame-0",
        yolo_confidence=None,
        yolo_threshold_observed=False,
    )

    completed = tracker.force_clear(
        observed_at=observed_at + timedelta(seconds=6),
        yolo_threshold_observed=True,
    )

    assert completed is None

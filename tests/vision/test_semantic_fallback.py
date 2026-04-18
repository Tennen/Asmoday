from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

import vision_service.vision.semantic_runtime as semantic_runtime
from vision_service.contracts import (
    CameraIdentity,
    EntitySelector,
    RTSPSource,
    VisionRule,
    ZoneRect,
)
from vision_service.settings import Settings
from vision_service.vision.entities import (
    ProcessedFrame,
    TransitionContext,
    ZoneObservation,
)
from vision_service.vision.semantic import SemanticCheckResult
from vision_service.vision.semantic_fallback import SemanticFallbackTracker
from vision_service.vision.semantic_runtime import observe_semantic_fallback_safely
from vision_service.vision.roi.models import ROIOccupancyObservation


class FakeChecker:
    def __init__(self, verdicts: list[str]) -> None:
        self._verdicts = verdicts
        self.calls = 0
        self.image_bytes: list[bytes] = []

    async def check(self, *, image_bytes: bytes, rule: VisionRule) -> SemanticCheckResult:
        self.image_bytes.append(image_bytes)
        verdict = self._verdicts[min(self.calls, len(self._verdicts) - 1)]
        self.calls += 1
        return SemanticCheckResult(
            verdict=verdict,  # type: ignore[arg-type]
            raw_output=verdict,
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
    checker = FakeChecker(["有", "有", "无法确定"])
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
            evidence_image_bytes=f"frame-{second}".encode(),
            semantic_image_bytes=f"frame-{second}".encode(),
            yolo_confidence=None,
            yolo_threshold_observed=False,
        )
        assert transition is None

    completed = tracker.force_clear(observed_at=start + timedelta(seconds=7))

    assert completed is not None
    assert completed.dwell_seconds == 5
    assert completed.semantic_result.verdict == "有"
    assert completed.vote_summary.attempts == 2
    assert completed.vote_summary.positive_votes == 2
    assert completed.confidence.source == "roi_vlm_fallback"
    assert len(completed.evidence_samples) == 3
    assert checker.calls == 2


@pytest.mark.asyncio
async def test_semantic_fallback_uses_crop_for_vlm_but_stores_raw_evidence() -> None:
    checker = FakeChecker(["有"])
    tracker = SemanticFallbackTracker(
        rule=build_rule(),
        threshold_seconds=1,
        sample_interval_seconds=0.0,
        max_samples=9,
        consecutive_yolo_failures=1,
        retry_cooldown_seconds=1.0,
        max_attempts_per_episode=1,
        checker=checker,
    )
    start = datetime(2026, 4, 13, 8, 1, tzinfo=UTC)

    for second in range(2):
        await tracker.observe(
            observed_at=start + timedelta(seconds=second),
            roi_observation=occupied_roi(start + timedelta(seconds=second)),
            evidence_image_bytes=f"raw-{second}".encode(),
            semantic_image_bytes=f"crop-{second}".encode(),
            yolo_confidence=None,
            yolo_threshold_observed=False,
        )

    completed = tracker.force_clear(observed_at=start + timedelta(seconds=2))

    assert completed is not None
    assert checker.image_bytes == [b"crop-0"]
    assert completed.evidence_samples
    assert all(
        sample.image_bytes in {b"raw-0", b"raw-1"}
        for sample in completed.evidence_samples
    )
    assert all(
        not sample.image_bytes.startswith(b"crop")
        for sample in completed.evidence_samples
    )


@pytest.mark.asyncio
async def test_semantic_fallback_requests_full_frame_evidence_only_on_sample_interval() -> None:
    checker = FakeChecker(["无法确定"])
    tracker = SemanticFallbackTracker(
        rule=build_rule(),
        threshold_seconds=5,
        sample_interval_seconds=10.0,
        max_samples=9,
        consecutive_yolo_failures=1,
        retry_cooldown_seconds=1.0,
        max_attempts_per_episode=1,
        checker=checker,
    )
    start = datetime(2026, 4, 13, 8, 1, tzinfo=UTC)

    assert tracker.should_capture_evidence_sample(
        observed_at=start,
        roi_observation=occupied_roi(start),
    )
    await tracker.observe(
        observed_at=start,
        roi_observation=occupied_roi(start),
        evidence_image_bytes=b"raw-0",
        semantic_image_bytes=b"crop-0",
        yolo_confidence=None,
        yolo_threshold_observed=False,
    )

    assert not tracker.should_capture_evidence_sample(
        observed_at=start + timedelta(seconds=1),
        roi_observation=occupied_roi(start + timedelta(seconds=1)),
    )
    assert tracker.should_capture_evidence_sample(
        observed_at=start + timedelta(seconds=10),
        roi_observation=occupied_roi(start + timedelta(seconds=10)),
    )


@pytest.mark.asyncio
async def test_semantic_fallback_rejects_single_positive_vote_episode() -> None:
    checker = FakeChecker(["有", "无法确定", "无法确定"])
    tracker = SemanticFallbackTracker(
        rule=build_rule(),
        threshold_seconds=5,
        sample_interval_seconds=1.0,
        max_samples=9,
        consecutive_yolo_failures=2,
        retry_cooldown_seconds=1.0,
        max_attempts_per_episode=3,
        checker=checker,
    )
    start = datetime(2026, 4, 13, 8, 2, tzinfo=UTC)

    for second in range(7):
        transition = await tracker.observe(
            observed_at=start + timedelta(seconds=second),
            roi_observation=occupied_roi(start + timedelta(seconds=second)),
            evidence_image_bytes=f"frame-{second}".encode(),
            semantic_image_bytes=f"frame-{second}".encode(),
            yolo_confidence=None,
            yolo_threshold_observed=False,
        )
        assert transition is None

    completed = tracker.force_clear(observed_at=start + timedelta(seconds=8))

    assert completed is None
    assert checker.calls == 3


@pytest.mark.asyncio
async def test_semantic_fallback_spreads_checks_across_threshold_window() -> None:
    checker = FakeChecker(["无法确定", "无法确定", "无法确定"])
    tracker = SemanticFallbackTracker(
        rule=build_rule(),
        threshold_seconds=6,
        sample_interval_seconds=1.0,
        max_samples=9,
        consecutive_yolo_failures=2,
        retry_cooldown_seconds=0.5,
        max_attempts_per_episode=3,
        checker=checker,
    )
    start = datetime(2026, 4, 13, 8, 4, tzinfo=UTC)
    call_counts: list[int] = []

    for second in range(6):
        await tracker.observe(
            observed_at=start + timedelta(seconds=second),
            roi_observation=occupied_roi(start + timedelta(seconds=second)),
            evidence_image_bytes=f"frame-{second}".encode(),
            semantic_image_bytes=f"frame-{second}".encode(),
            yolo_confidence=None,
            yolo_threshold_observed=False,
        )
        call_counts.append(checker.calls)

    assert call_counts == [0, 1, 1, 2, 2, 3]


@pytest.mark.asyncio
async def test_semantic_fallback_is_suppressed_after_yolo_threshold_observed() -> None:
    checker = FakeChecker(["有", "有"])
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
        evidence_image_bytes=b"frame-0",
        semantic_image_bytes=b"frame-0",
        yolo_confidence=None,
        yolo_threshold_observed=False,
    )

    completed = tracker.force_clear(
        observed_at=observed_at + timedelta(seconds=6),
        yolo_threshold_observed=True,
    )

    assert completed is None


@pytest.mark.asyncio
async def test_semantic_runtime_passes_full_frame_evidence_and_zone_crop_vlm_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class CapturingSemanticFallback:
        def __init__(self) -> None:
            self.observed: dict[str, object] | None = None

        def should_capture_evidence_sample(self, **kwargs):  # noqa: ANN003, ANN201
            return True

        async def observe(self, **kwargs):  # noqa: ANN003, ANN201
            self.observed = kwargs
            return None

    def fake_encode_frame(*, frame: np.ndarray, jpeg_quality: int) -> bytes:
        return f"{frame.shape[0]}x{frame.shape[1]}".encode()

    fallback = CapturingSemanticFallback()
    monkeypatch.setattr(semantic_runtime, "encode_frame", fake_encode_frame)

    transition, error = await semantic_runtime.observe_semantic_fallback_safely(
        rule=build_rule(),
        settings=Settings(),
        semantic_fallback=fallback,  # type: ignore[arg-type]
        frame=np.zeros((10, 20, 3), dtype=np.uint8),
        observed_at=datetime(2026, 4, 13, 8, 12, tzinfo=UTC),
        processed=ProcessedFrame(
            transition=None,
            context=TransitionContext(),
            zone_observation=ZoneObservation(
                visible_tracks={},
                track_entities={},
                track_confidences={},
                entities=(),
            ),
            roi_observation=occupied_roi(datetime(2026, 4, 13, 8, 12, tzinfo=UTC)),
        ),
        yolo_threshold_observed=False,
    )

    assert transition is None
    assert error is None
    assert fallback.observed is not None
    assert fallback.observed["evidence_image_bytes"] == b"10x20"
    assert fallback.observed["semantic_image_bytes"] == b"2x4"


@pytest.mark.asyncio
async def test_semantic_runtime_still_checks_vlm_when_full_frame_evidence_encode_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class CapturingSemanticFallback:
        def __init__(self) -> None:
            self.observed: dict[str, object] | None = None

        def should_capture_evidence_sample(self, **kwargs):  # noqa: ANN003, ANN201
            return True

        async def observe(self, **kwargs):  # noqa: ANN003, ANN201
            self.observed = kwargs
            return None

    def fake_encode_frame(*, frame: np.ndarray, jpeg_quality: int) -> bytes:
        if frame.shape[:2] == (10, 20):
            raise RuntimeError("full frame encode failed")
        return f"{frame.shape[0]}x{frame.shape[1]}".encode()

    fallback = CapturingSemanticFallback()
    monkeypatch.setattr(semantic_runtime, "encode_frame", fake_encode_frame)

    transition, error = await semantic_runtime.observe_semantic_fallback_safely(
        rule=build_rule(),
        settings=Settings(),
        semantic_fallback=fallback,  # type: ignore[arg-type]
        frame=np.zeros((10, 20, 3), dtype=np.uint8),
        observed_at=datetime(2026, 4, 13, 8, 12, tzinfo=UTC),
        processed=ProcessedFrame(
            transition=None,
            context=TransitionContext(),
            zone_observation=ZoneObservation(
                visible_tracks={},
                track_entities={},
                track_confidences={},
                entities=(),
            ),
            roi_observation=occupied_roi(datetime(2026, 4, 13, 8, 12, tzinfo=UTC)),
        ),
        yolo_threshold_observed=False,
    )

    assert transition is None
    assert error is not None
    assert "full-frame semantic fallback evidence" in error
    assert fallback.observed is not None
    assert fallback.observed["evidence_image_bytes"] is None
    assert fallback.observed["semantic_image_bytes"] == b"2x4"


@pytest.mark.asyncio
async def test_semantic_runtime_returns_error_for_unexpected_checker_failure() -> None:
    class FailingSemanticFallback:
        async def observe(self, **kwargs):  # noqa: ANN003, ANN201
            raise RuntimeError("model server crashed")

    transition, error = await observe_semantic_fallback_safely(
        rule=build_rule(),
        settings=Settings(),
        semantic_fallback=FailingSemanticFallback(),  # type: ignore[arg-type]
        frame=np.zeros((4, 4, 3), dtype=np.uint8),
        observed_at=datetime(2026, 4, 13, 8, 12, tzinfo=UTC),
        processed=ProcessedFrame(
            transition=None,
            context=TransitionContext(),
            zone_observation=ZoneObservation(
                visible_tracks={},
                track_entities={},
                track_confidences={},
                entities=(),
            ),
            roi_observation=None,
        ),
        yolo_threshold_observed=False,
    )

    assert transition is None
    assert error is not None
    assert "model server crashed" in error

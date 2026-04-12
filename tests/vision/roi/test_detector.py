from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from vision_service.contracts import (
    CameraIdentity,
    EntitySelector,
    RTSPSource,
    VisionRule,
    ZoneRect,
)
from vision_service.settings import Settings
from vision_service.vision.roi.detector import ROIOccupancyDetector


class FakeSubtractor:
    def __init__(self, mask: np.ndarray) -> None:
        self._mask = mask
        self.learning_rates: list[float] = []

    def apply(self, frame, learningRate):  # noqa: ANN001, ANN201, N803
        self.learning_rates.append(float(learningRate))
        return self._mask.copy()


def build_rule() -> VisionRule:
    return VisionRule(
        id="rule-1",
        name="Rule 1",
        enabled=True,
        camera=CameraIdentity(device_id="camera-1"),
        rtsp_source=RTSPSource(url="rtsp://camera/test"),
        entity_selector=EntitySelector(value="cat"),
        zone=ZoneRect(x=0.0, y=0.0, width=1.0, height=1.0),
        stay_threshold_seconds=2,
    )


def test_roi_detector_freezes_learning_rate_once_presence_is_active(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frame = np.zeros((40, 40, 3), dtype=np.uint8)
    fake_subtractor = FakeSubtractor(
        np.full((40, 40), 255, dtype=np.uint8),
    )
    detector = ROIOccupancyDetector(
        rule=build_rule(),
        settings=Settings(
            roi_warmup_seconds=0.0,
            roi_occupancy_ratio_threshold=0.08,
            roi_min_largest_blob_area=10,
        ),
    )
    monkeypatch.setattr(detector, "_create_subtractor", lambda: fake_subtractor)
    start = datetime(2026, 4, 12, 8, 0, tzinfo=UTC)

    first = detector.observe(frame=frame, observed_at=start)
    second = detector.observe(
        frame=frame,
        observed_at=start + timedelta(seconds=2),
    )

    assert first.state == "candidate_occupied"
    assert second.state == "occupied"
    assert fake_subtractor.learning_rates == [-1.0, 0.0]

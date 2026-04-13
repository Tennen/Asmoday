from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np

from vision_service.contracts import VisionRule
from vision_service.settings import Settings
from vision_service.vision.roi.models import ROIOccupancyObservation
from vision_service.vision.roi.state import ROIOccupancyStateMachine
from vision_service.vision.zone import crop_zone_frame


@dataclass(slots=True, frozen=True)
class _PreparedROIFrame:
    frame: np.ndarray[Any, Any]
    mask: np.ndarray[Any, Any]


class ROIOccupancyDetector:
    def __init__(
        self,
        *,
        rule: VisionRule,
        settings: Settings,
    ) -> None:
        self._rule = rule.model_copy(deep=True)
        self._settings = settings
        self._state_machine = ROIOccupancyStateMachine(
            stay_threshold_seconds=self._rule.stay_threshold_seconds,
            warmup_seconds=self._settings.roi_warmup_seconds,
            clear_hold_seconds=self._settings.roi_clear_hold_seconds,
        )
        self._subtractor: Any | None = None
        self._open_kernel: np.ndarray[Any, Any] | None = None
        self._close_kernel: np.ndarray[Any, Any] | None = None
        self._working_shape: tuple[int, int] | None = None

    @property
    def current_state(self) -> str:
        return self._state_machine.state

    def observe(
        self,
        *,
        frame: np.ndarray[Any, Any],
        observed_at: datetime,
    ) -> ROIOccupancyObservation:
        import cv2

        prepared = self._prepare_frame(frame)
        if self._working_shape != prepared.frame.shape[:2]:
            self._working_shape = prepared.frame.shape[:2]
            self._subtractor = self._create_subtractor()
            self._open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            self._close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            self._state_machine.reset()

        assert self._subtractor is not None
        assert self._open_kernel is not None
        assert self._close_kernel is not None

        fgmask = self._subtractor.apply(
            prepared.frame,
            learningRate=self._learning_rate(),
        )
        fgmask_binary = np.where(fgmask > 0, 255, 0).astype(np.uint8)
        masked = cv2.bitwise_and(fgmask_binary, prepared.mask)
        opened = cv2.morphologyEx(masked, cv2.MORPH_OPEN, self._open_kernel)
        cleaned = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, self._close_kernel)

        roi_area_pixels = int(cv2.countNonZero(prepared.mask))
        foreground_pixels = int(cv2.countNonZero(cleaned))
        occupancy_ratio = (
            foreground_pixels / roi_area_pixels
            if roi_area_pixels > 0
            else 0.0
        )
        _, _, stats, _ = cv2.connectedComponentsWithStats(cleaned, connectivity=8)
        largest_blob_area = (
            int(stats[1:, cv2.CC_STAT_AREA].max())
            if len(stats) > 1
            else 0
        )

        frame_present = (
            occupancy_ratio >= self._settings.roi_occupancy_ratio_threshold
            or largest_blob_area >= self._largest_blob_threshold(roi_area_pixels)
        )
        return self._state_machine.observe(
            observed_at=observed_at,
            frame_present=frame_present,
            occupancy_ratio=occupancy_ratio,
            largest_blob_area=largest_blob_area,
            roi_area_pixels=roi_area_pixels,
            foreground_pixels=foreground_pixels,
        )

    def _prepare_frame(self, frame: np.ndarray[Any, Any]) -> _PreparedROIFrame:
        resized = crop_zone_frame(
            rule=self._rule,
            frame=frame,
            max_side_px=self._settings.roi_max_side_px,
        )

        mask = np.full(resized.shape[:2], 255, dtype=np.uint8)
        return _PreparedROIFrame(frame=resized, mask=mask)

    def _create_subtractor(self) -> Any:
        import cv2

        return cv2.createBackgroundSubtractorMOG2(
            history=self._settings.roi_mog2_history,
            varThreshold=self._settings.roi_mog2_var_threshold,
            detectShadows=False,
        )

    def _learning_rate(self) -> float:
        if self._state_machine.state in {"candidate_occupied", "occupied"}:
            return 0.0
        return -1.0

    def _largest_blob_threshold(self, roi_area_pixels: int) -> int:
        return max(
            self._settings.roi_min_largest_blob_area,
            int(
                roi_area_pixels
                * self._settings.roi_largest_blob_area_ratio_threshold
            ),
        )

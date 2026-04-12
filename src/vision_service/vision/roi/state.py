from datetime import datetime

from vision_service.vision.roi.models import (
    ROIOccupancyObservation,
    ROIOccupancyState,
)


class ROIOccupancyStateMachine:
    def __init__(
        self,
        *,
        stay_threshold_seconds: int,
        warmup_seconds: float,
        clear_hold_seconds: float,
    ) -> None:
        self._stay_threshold_seconds = stay_threshold_seconds
        self._warmup_seconds = warmup_seconds
        self._clear_hold_seconds = clear_hold_seconds
        self._state: ROIOccupancyState = "warming_up"
        self._started_at: datetime | None = None
        self._presence_started_at: datetime | None = None
        self._absence_started_at: datetime | None = None

    @property
    def state(self) -> ROIOccupancyState:
        return self._state

    def reset(self) -> None:
        self._state = "warming_up"
        self._started_at = None
        self._presence_started_at = None
        self._absence_started_at = None

    def observe(
        self,
        *,
        observed_at: datetime,
        frame_present: bool,
        occupancy_ratio: float,
        largest_blob_area: int,
        roi_area_pixels: int,
        foreground_pixels: int,
    ) -> ROIOccupancyObservation:
        if self._started_at is None:
            self._started_at = observed_at

        if (
            self._warmup_seconds > 0
            and (observed_at - self._started_at).total_seconds() < self._warmup_seconds
        ):
            self._state = "warming_up"
            self._presence_started_at = None
            self._absence_started_at = None
            return self._build_observation(
                observed_at=observed_at,
                frame_present=frame_present,
                occupancy_ratio=occupancy_ratio,
                largest_blob_area=largest_blob_area,
                roi_area_pixels=roi_area_pixels,
                foreground_pixels=foreground_pixels,
            )

        if frame_present:
            if self._presence_started_at is None:
                self._presence_started_at = observed_at
            self._absence_started_at = None
            dwell_seconds = max(
                0,
                int((observed_at - self._presence_started_at).total_seconds()),
            )
            self._state = (
                "occupied"
                if dwell_seconds >= self._stay_threshold_seconds
                else "candidate_occupied"
            )
        elif self._state in {"candidate_occupied", "occupied"}:
            if self._absence_started_at is None:
                self._absence_started_at = observed_at
            if (
                observed_at - self._absence_started_at
            ).total_seconds() >= self._clear_hold_seconds:
                self._state = "empty"
                self._presence_started_at = None
                self._absence_started_at = None
        else:
            self._state = "empty"
            self._presence_started_at = None
            self._absence_started_at = None

        return self._build_observation(
            observed_at=observed_at,
            frame_present=frame_present,
            occupancy_ratio=occupancy_ratio,
            largest_blob_area=largest_blob_area,
            roi_area_pixels=roi_area_pixels,
            foreground_pixels=foreground_pixels,
        )

    def _build_observation(
        self,
        *,
        observed_at: datetime,
        frame_present: bool,
        occupancy_ratio: float,
        largest_blob_area: int,
        roi_area_pixels: int,
        foreground_pixels: int,
    ) -> ROIOccupancyObservation:
        return ROIOccupancyObservation(
            observed_at=observed_at,
            state=self._state,
            frame_present=frame_present,
            occupancy_ratio=occupancy_ratio,
            largest_blob_area=largest_blob_area,
            roi_area_pixels=roi_area_pixels,
            foreground_pixels=foreground_pixels,
        )

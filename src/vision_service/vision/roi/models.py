from dataclasses import dataclass
from datetime import datetime
from typing import Literal


ROIOccupancyState = Literal[
    "warming_up",
    "empty",
    "candidate_occupied",
    "occupied",
]


@dataclass(slots=True, frozen=True)
class ROIOccupancyObservation:
    observed_at: datetime
    state: ROIOccupancyState
    frame_present: bool
    occupancy_ratio: float
    largest_blob_area: int
    roi_area_pixels: int
    foreground_pixels: int

    @property
    def presence_active(self) -> bool:
        return self.state in {"candidate_occupied", "occupied"}

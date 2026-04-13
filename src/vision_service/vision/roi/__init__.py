"""ROI occupancy detection helpers."""

__all__ = [
    "ROIOccupancyDetector",
    "ROIOccupancyObservation",
    "ROIOccupancyState",
    "ROIOccupancyStateMachine",
]


def __getattr__(name: str):
    if name == "ROIOccupancyDetector":
        from vision_service.vision.roi.detector import ROIOccupancyDetector

        return ROIOccupancyDetector
    if name in {"ROIOccupancyObservation", "ROIOccupancyState"}:
        from vision_service.vision.roi.models import (
            ROIOccupancyObservation,
            ROIOccupancyState,
        )

        return {
            "ROIOccupancyObservation": ROIOccupancyObservation,
            "ROIOccupancyState": ROIOccupancyState,
        }[name]
    if name == "ROIOccupancyStateMachine":
        from vision_service.vision.roi.state import ROIOccupancyStateMachine

        return ROIOccupancyStateMachine
    raise AttributeError(name)

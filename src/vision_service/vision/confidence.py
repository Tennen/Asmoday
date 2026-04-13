from dataclasses import dataclass

from vision_service.vision.roi.models import ROIOccupancyObservation


@dataclass(slots=True, frozen=True)
class ConfidenceAssessment:
    source: str
    score: float
    breakdown: dict[str, float]


def roi_signal_confidence(
    observation: ROIOccupancyObservation | None,
) -> float:
    if observation is None:
        return 0.45

    occupancy_strength = min(1.0, observation.occupancy_ratio / 0.18)
    blob_ratio = (
        observation.largest_blob_area / observation.roi_area_pixels
        if observation.roi_area_pixels > 0
        else 0.0
    )
    blob_strength = min(1.0, blob_ratio / 0.10)
    presence_bonus = 0.10 if observation.presence_active else 0.0
    return round(
        _clamp(
            0.35
            + (0.25 * occupancy_strength)
            + (0.20 * blob_strength)
            + presence_bonus,
            lower=0.35,
            upper=0.85,
        ),
        3,
    )


def score_yolo_event(
    *,
    yolo_confidence: float | None,
    roi_observation: ROIOccupancyObservation | None,
    dwell_seconds: int,
    threshold_seconds: int,
) -> ConfidenceAssessment:
    detection_score = _clamp(yolo_confidence or 0.78, lower=0.35, upper=0.99)
    roi_score = roi_signal_confidence(roi_observation)
    dwell_multiplier = min(
        2.0,
        dwell_seconds / max(float(threshold_seconds), 1.0),
    )
    dwell_score = 0.70 + (0.30 * (dwell_multiplier / 2.0))
    score = _clamp(
        (0.75 * detection_score) + (0.10 * roi_score) + (0.15 * dwell_score),
        lower=0.50,
        upper=0.99,
    )
    return ConfidenceAssessment(
        source="yolo_track",
        score=round(score, 3),
        breakdown={
            "yolo": round(detection_score, 3),
            "roi": round(roi_score, 3),
        },
    )


def score_semantic_fallback(
    *,
    verdict: str,
    roi_observation: ROIOccupancyObservation | None,
    yolo_support_confidence: float | None,
    consecutive_yolo_failures: int,
) -> ConfidenceAssessment:
    vlm_score = {
        "有": 0.82,
        "疑似有": 0.64,
        "无法确定": 0.40,
    }[verdict]
    roi_score = roi_signal_confidence(roi_observation)
    yolo_score = (
        _clamp(0.20 + (0.60 * yolo_support_confidence), lower=0.20, upper=0.75)
        if yolo_support_confidence is not None
        else 0.18
    )
    failure_penalty = _clamp(
        1.0 - max(0, consecutive_yolo_failures - 3) * 0.02,
        lower=0.85,
        upper=1.0,
    )
    score = _clamp(
        (
            (0.55 * vlm_score)
            + (0.30 * roi_score)
            + (0.15 * yolo_score)
        )
        * failure_penalty,
        lower=0.55 if verdict == "有" else 0.45,
        upper=0.89,
    )
    return ConfidenceAssessment(
        source="roi_vlm_fallback",
        score=round(score, 3),
        breakdown={
            "vlm": round(vlm_score, 3),
            "roi": round(roi_score, 3),
            "yolo": round(yolo_score, 3),
        },
    )


def _clamp(value: float, *, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))

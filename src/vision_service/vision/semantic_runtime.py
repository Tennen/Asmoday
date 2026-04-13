from datetime import datetime

import numpy as np

from vision_service.contracts import VisionRule
from vision_service.settings import Settings
from vision_service.vision.entities import ProcessedFrame
from vision_service.vision.semantic import SemanticChecker
from vision_service.vision.semantic_fallback import (
    SemanticFallbackTracker,
    SemanticFallbackTransition,
)
from vision_service.vision.zone import crop_zone_frame, encode_frame


def build_semantic_fallback_tracker(
    *,
    rule: VisionRule,
    settings: Settings,
    checker: SemanticChecker | None,
    roi_enabled: bool,
) -> SemanticFallbackTracker | None:
    if not roi_enabled or checker is None:
        return None
    return SemanticFallbackTracker(
        rule=rule,
        threshold_seconds=rule.stay_threshold_seconds,
        sample_interval_seconds=settings.frame_sample_interval_seconds,
        max_samples=settings.evidence_buffer_max_samples,
        consecutive_yolo_failures=settings.semantic_checker_consecutive_yolo_failures,
        retry_cooldown_seconds=settings.semantic_checker_retry_cooldown_seconds,
        max_attempts_per_episode=settings.semantic_checker_max_attempts_per_episode,
        checker=checker,
    )


async def observe_semantic_fallback(
    *,
    rule: VisionRule,
    settings: Settings,
    semantic_fallback: SemanticFallbackTracker | None,
    frame: np.ndarray,
    observed_at: datetime,
    processed: ProcessedFrame,
    yolo_threshold_observed: bool,
) -> SemanticFallbackTransition | None:
    if semantic_fallback is None:
        return None

    image_bytes: bytes | None = None
    if (
        processed.roi_observation is not None
        and processed.roi_observation.presence_active
    ):
        zone_crop = crop_zone_frame(
            rule=rule,
            frame=frame,
            max_side_px=settings.roi_max_side_px,
        )
        image_bytes = encode_frame(
            frame=zone_crop,
            jpeg_quality=settings.jpeg_quality,
        )

    return await semantic_fallback.observe(
        observed_at=observed_at,
        roi_observation=processed.roi_observation,
        image_bytes=image_bytes,
        yolo_confidence=max(
            processed.zone_observation.track_confidences.values(),
            default=None,
        ),
        yolo_threshold_observed=yolo_threshold_observed,
    )

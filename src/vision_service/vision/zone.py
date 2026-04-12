from typing import Any

import numpy as np

from vision_service.contracts import EntityDescriptor, VisionRule
from vision_service.vision.backend import DetectionBatch
from vision_service.vision.entities import (
    ZoneObservation,
    dedupe_entities,
    entity_descriptor_for_detection,
)


def select_target_detections(
    *,
    batch: DetectionBatch,
    entity_value: str,
) -> tuple[Any, np.ndarray[Any, Any] | None]:
    detections = batch.detections
    if (
        len(detections) == 0
        or detections.class_id is None
        or entity_value == ""
    ):
        return detections, None

    mask = np.array(
        [
            batch.labels.get(int(class_id)) == entity_value
            for class_id in detections.class_id
        ],
        dtype=bool,
    )
    return detections[mask], mask


def visible_tracks_in_zone(
    *,
    rule: VisionRule,
    detections: Any,
    frame: np.ndarray[Any, Any],
    labels: dict[int, str],
    batch: DetectionBatch,
    class_mask: np.ndarray[Any, Any] | None,
    default_entity: EntityDescriptor | None,
    jpeg_quality: int,
) -> ZoneObservation:
    if len(detections) == 0 or detections.tracker_id is None:
        return ZoneObservation(
            visible_tracks={},
            track_entities={},
            entities=(),
        )

    frame_height, frame_width = frame.shape[:2]
    zone_left = rule.zone.x * frame_width
    zone_top = rule.zone.y * frame_height
    zone_right = zone_left + (rule.zone.width * frame_width)
    zone_bottom = zone_top + (rule.zone.height * frame_height)

    track_ids: list[int] = []
    track_entities: dict[int, EntityDescriptor] = {}
    class_ids = detections.class_id

    for index, (bounding_box, tracker_id) in enumerate(
        zip(detections.xyxy, detections.tracker_id)
    ):
        if tracker_id is None:
            continue
        center_x = float((bounding_box[0] + bounding_box[2]) / 2.0)
        center_y = float((bounding_box[1] + bounding_box[3]) / 2.0)
        if zone_left <= center_x <= zone_right and zone_top <= center_y <= zone_bottom:
            track_id = int(tracker_id)
            track_ids.append(track_id)
            track_entities[track_id] = entity_descriptor_for_detection(
                class_id=int(class_ids[index]) if class_ids is not None else None,
                labels=labels,
                default_entity=default_entity,
            )

    visible_tracks: dict[int, bytes | None] = {}
    if track_ids:
        encoded_frame = encode_annotated_frame(
            batch=batch,
            frame=frame,
            class_mask=class_mask,
            jpeg_quality=jpeg_quality,
        )
        visible_tracks = {track_id: encoded_frame for track_id in track_ids}

    return ZoneObservation(
        visible_tracks=visible_tracks,
        track_entities=track_entities,
        entities=dedupe_entities(track_entities.values()),
    )


def encode_annotated_frame(
    *,
    batch: DetectionBatch,
    frame: np.ndarray[Any, Any],
    class_mask: np.ndarray[Any, Any] | None,
    jpeg_quality: int,
) -> bytes:
    plot_result = batch.result
    if class_mask is not None and getattr(batch.result, "boxes", None) is not None:
        plot_result = batch.result.new()
        plot_result.boxes = batch.result.boxes[class_mask]

    annotated_frame = plot_result.plot(
        img=frame,
        boxes=True,
        labels=True,
        masks=False,
        probs=False,
    )
    return encode_frame(frame=annotated_frame, jpeg_quality=jpeg_quality)


def encode_frame(
    *,
    frame: np.ndarray[Any, Any],
    jpeg_quality: int,
) -> bytes:
    import cv2

    success, buffer = cv2.imencode(
        ".jpg",
        frame,
        [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality],
    )
    if not success:
        raise RuntimeError("failed to encode evidence frame")
    return buffer.tobytes()

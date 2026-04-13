from dataclasses import dataclass, field
from typing import Iterable

from vision_service.contracts import EntityDescriptor, VisionRule
from vision_service.runtime.dwell import DwellTransition
from vision_service.vision.roi.models import ROIOccupancyObservation


@dataclass(slots=True, frozen=True)
class TransitionContext:
    primary_entity: EntityDescriptor | None = None
    primary_confidence: float | None = None
    entities: tuple[EntityDescriptor, ...] = ()


@dataclass(slots=True, frozen=True)
class ZoneObservation:
    visible_tracks: dict[int, bytes | None]
    track_entities: dict[int, EntityDescriptor]
    entities: tuple[EntityDescriptor, ...]
    track_confidences: dict[int, float] = field(default_factory=dict)


def default_entity_for_rule(rule: VisionRule) -> EntityDescriptor | None:
    if rule.entity_selector.value == "":
        return None
    return EntityDescriptor(
        kind=rule.entity_selector.kind,
        value=rule.entity_selector.value,
        display_name=rule.entity_selector.value.replace("_", " ").title(),
    )


def entity_descriptor_for_detection(
    *,
    class_id: int | None,
    labels: dict[int, str],
    default_entity: EntityDescriptor | None,
) -> EntityDescriptor:
    if class_id is None:
        if default_entity is not None:
            return default_entity
        value = "unknown"
    else:
        value = labels.get(class_id, str(class_id))

    return EntityDescriptor(
        kind="label",
        value=value,
        display_name=value.replace("_", " ").title(),
    )


def build_transition_context(
    *,
    transition: DwellTransition | None,
    current_track_entities: dict[int, EntityDescriptor],
    current_track_confidences: dict[int, float],
    removed_track_entities: dict[int, EntityDescriptor],
    removed_track_confidences: dict[int, float],
    current_entities: tuple[EntityDescriptor, ...],
    default_entity: EntityDescriptor | None,
) -> TransitionContext:
    if transition is None:
        return TransitionContext()

    primary_entity = default_entity
    primary_confidence: float | None = None
    if transition.track_id is not None:
        primary_entity = current_track_entities.get(transition.track_id)
        primary_confidence = current_track_confidences.get(transition.track_id)
        if primary_entity is None:
            primary_entity = removed_track_entities.get(transition.track_id, default_entity)
        if primary_confidence is None:
            primary_confidence = removed_track_confidences.get(transition.track_id)

    ordered_entities: list[EntityDescriptor] = []
    if primary_entity is not None:
        ordered_entities.append(primary_entity)
    ordered_entities.extend(current_entities)
    return TransitionContext(
        primary_entity=primary_entity,
        primary_confidence=primary_confidence,
        entities=dedupe_entities(ordered_entities),
    )


@dataclass(slots=True, frozen=True)
class ProcessedFrame:
    transition: DwellTransition | None
    context: TransitionContext
    zone_observation: ZoneObservation
    roi_observation: ROIOccupancyObservation | None = None


def dedupe_entities(
    entities: Iterable[EntityDescriptor],
) -> tuple[EntityDescriptor, ...]:
    ordered: list[EntityDescriptor] = []
    seen: set[tuple[str, str]] = set()

    for entity in entities:
        key = (entity.kind, entity.value)
        if key in seen:
            continue
        ordered.append(entity)
        seen.add(key)

    return tuple(ordered)


def evidence_metadata() -> dict[str, dict[str, str]]:
    return {
        "annotations": {
            "image_kind": "annotated",
            "source": "ultralytics.plot",
        }
    }

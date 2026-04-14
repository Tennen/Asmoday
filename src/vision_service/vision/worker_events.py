from vision_service.contracts import EntityDescriptor, VisionRule
from vision_service.runtime.dwell import DwellTransition
from vision_service.runtime.events import EventEvidence, RuleEvent
from vision_service.vision.entities import TransitionContext, evidence_metadata
from vision_service.vision.key_entity_matcher import KeyEntityIdentification
from vision_service.vision.roi.models import ROIOccupancyObservation
from vision_service.vision.confidence import score_yolo_event
from vision_service.vision.semantic_fallback import (
    SemanticFallbackTransition,
    build_semantic_event_metadata,
    semantic_evidence_metadata,
)


def build_yolo_rule_event(
    *,
    rule: VisionRule,
    transition: DwellTransition,
    context: TransitionContext,
    roi_observation: ROIOccupancyObservation | None,
    key_entity_identification: KeyEntityIdentification | None = None,
) -> RuleEvent:
    evidence: tuple[EventEvidence, ...] = ()
    if transition.status == "threshold_met" and transition.evidence_samples:
        phases = ("start", "middle", "end")
        evidence = tuple(
            EventEvidence(
                phase=phase,
                captured_at=sample.captured_at,
                image_bytes=sample.image_bytes,
                metadata=evidence_metadata(sample),
            )
            for phase, sample in zip(phases, transition.evidence_samples)
        )

    decision_confidence = score_yolo_event(
        yolo_confidence=context.primary_confidence,
        roi_observation=roi_observation,
        dwell_seconds=transition.dwell_seconds,
        threshold_seconds=rule.stay_threshold_seconds,
    )
    metadata: dict[str, object] = {
        "decision": {
            "source": decision_confidence.source,
            "confidence_score": decision_confidence.score,
            "confidence_breakdown": decision_confidence.breakdown,
        }
    }
    if transition.track_id is not None:
        metadata["track_id"] = str(transition.track_id)
    if key_entity_identification is not None:
        metadata["key_entity_match"] = key_entity_identification.metadata

    return RuleEvent(
        rule_id=rule.id,
        camera_device_id=rule.camera.device_id,
        status=transition.status,
        observed_at=transition.observed_at,
        dwell_seconds=transition.dwell_seconds,
        entity_value=(
            context.primary_entity.value if context.primary_entity is not None else None
        ),
        key_entity_id=(
            key_entity_identification.key_entity_id
            if key_entity_identification is not None
            else None
        ),
        entities=context.entities,
        metadata=metadata,
        evidence=evidence,
    )


def build_semantic_rule_event(
    *,
    rule: VisionRule,
    default_entity: EntityDescriptor | None,
    transition: SemanticFallbackTransition,
) -> RuleEvent:
    evidence: tuple[EventEvidence, ...] = ()
    if transition.evidence_samples:
        phases = ("start", "middle", "end")
        evidence = tuple(
            EventEvidence(
                phase=phase,
                captured_at=sample.captured_at,
                image_bytes=sample.image_bytes,
                metadata=semantic_evidence_metadata(transition),
            )
            for phase, sample in zip(phases, transition.evidence_samples)
        )

    return RuleEvent(
        rule_id=rule.id,
        camera_device_id=rule.camera.device_id,
        status="threshold_met",
        observed_at=transition.observed_at,
        dwell_seconds=transition.dwell_seconds,
        entity_value=default_entity.value if default_entity is not None else None,
        entities=(default_entity,) if default_entity is not None else (),
        metadata=build_semantic_event_metadata(transition),
        evidence=evidence,
    )

from base64 import b64encode

from vision_service.contracts import (
    EvidenceCallbackPayload,
    EvidenceCapture,
    EventCallbackPayload,
    EventRecord,
)
from vision_service.runtime.events import RuleEvent


def build_event_callback_payload(
    *,
    event: RuleEvent,
    event_id: str,
) -> EventCallbackPayload:
    return EventCallbackPayload(
        events=[
            EventRecord(
                event_id=event_id,
                rule_id=event.rule_id,
                camera_device_id=event.camera_device_id,
                status=event.status,
                observed_at=event.observed_at,
                dwell_seconds=event.dwell_seconds,
                entity_value=event.entity_value,
                entities=list(event.entities) or None,
                metadata=event.metadata or None,
            )
        ]
    )


def build_evidence_callback_payload(
    *,
    event: RuleEvent,
    event_id: str,
) -> EvidenceCallbackPayload | None:
    if not event.evidence:
        return None

    return EvidenceCallbackPayload(
        captures=[
            EvidenceCapture(
                capture_id=f"{event_id}:{capture.phase}",
                event_id=event_id,
                rule_id=event.rule_id,
                camera_device_id=event.camera_device_id,
                phase=capture.phase,
                captured_at=capture.captured_at,
                content_type="image/jpeg",
                image_base64=b64encode(capture.image_bytes).decode("ascii"),
                metadata=capture.metadata or None,
            )
            for capture in event.evidence
        ]
    )

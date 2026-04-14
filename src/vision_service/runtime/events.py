from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from vision_service.contracts.catalog import EntityDescriptor
from vision_service.contracts.callbacks import EvidencePhase, RuleEventStatus
from vision_service.contracts.control import KeyEntityId


@dataclass(slots=True, frozen=True)
class EventEvidence:
    phase: EvidencePhase
    captured_at: datetime
    image_bytes: bytes
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class RuleEvent:
    rule_id: str
    camera_device_id: str
    status: RuleEventStatus
    observed_at: datetime
    dwell_seconds: int
    entity_value: str | None = None
    key_entity_id: KeyEntityId | None = None
    entities: tuple[EntityDescriptor, ...] = ()
    event_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    evidence: tuple[EventEvidence, ...] = ()

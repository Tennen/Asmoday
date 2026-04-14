from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from vision_service.contracts.catalog import EntityDescriptor
from vision_service.contracts.control import KeyEntityId


ServiceStatus = Literal["unknown", "healthy", "degraded", "unhealthy", "stopped"]
RuleEventStatus = Literal["threshold_met", "cleared"]
EvidencePhase = Literal["start", "middle", "end"]


class RuntimeStatusPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: ServiceStatus
    message: str
    service_version: str | None = None
    reported_at: datetime
    runtime: dict[str, Any] | None = None


class NormalizedBoundingBox(BaseModel):
    model_config = ConfigDict(extra="forbid")

    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)
    width: float = Field(gt=0.0, le=1.0)
    height: float = Field(gt=0.0, le=1.0)


class EvidenceDetection(EntityDescriptor):
    model_config = ConfigDict(extra="forbid")

    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    track_id: str | None = None
    box: NormalizedBoundingBox


class EventRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_id: str | None = None
    rule_id: str = Field(min_length=1)
    camera_device_id: str | None = None
    status: RuleEventStatus
    observed_at: datetime
    dwell_seconds: int = Field(ge=0)
    entity_value: str | None = None
    key_entity_id: KeyEntityId | None = None
    entities: list[EntityDescriptor] | None = None
    metadata: dict[str, Any] | None = None


class EventCallbackPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    events: list[EventRecord] = Field(min_length=1)


class EvidenceCapture(BaseModel):
    model_config = ConfigDict(extra="forbid")

    capture_id: str | None = None
    event_id: str = Field(min_length=1)
    rule_id: str | None = None
    camera_device_id: str | None = None
    phase: EvidencePhase
    captured_at: datetime
    content_type: str | None = "image/jpeg"
    image_base64: str = Field(min_length=1)
    metadata: dict[str, Any] | None = None


class EvidenceCallbackPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    captures: list[EvidenceCapture] = Field(min_length=1)

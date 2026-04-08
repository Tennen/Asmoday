from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


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


class EventRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_id: str | None = None
    rule_id: str = Field(min_length=1)
    camera_device_id: str | None = None
    status: RuleEventStatus
    observed_at: datetime
    dwell_seconds: int = Field(ge=0)
    entity_value: str | None = None
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


class EvidenceCallbackPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    captures: list[EvidenceCapture] = Field(min_length=1)

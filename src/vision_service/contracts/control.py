from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class CallbackPaths(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status_path: str = Field(min_length=1)
    event_path: str = Field(min_length=1)
    evidence_path: str = Field(min_length=1)


class CameraIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid")

    device_id: str = Field(min_length=1)
    plugin_id: str | None = None
    vendor_device_id: str | None = None
    name: str | None = None
    entry_id: str | None = None


class RTSPSource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: str = Field(min_length=1)


class EntitySelector(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["label"] = "label"
    value: str = Field(min_length=1)


class ZoneRect(BaseModel):
    model_config = ConfigDict(extra="forbid")

    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)
    width: float = Field(gt=0.0, le=1.0)
    height: float = Field(gt=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_bounds(self) -> "ZoneRect":
        if self.x + self.width > 1.0:
            raise ValueError("zone.x + zone.width must be <= 1.0")
        if self.y + self.height > 1.0:
            raise ValueError("zone.y + zone.height must be <= 1.0")
        return self


class VisionRule(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    enabled: bool = True
    camera: CameraIdentity
    rtsp_source: RTSPSource
    entity_selector: EntitySelector
    zone: ZoneRect
    stay_threshold_seconds: int = Field(ge=1)


class SyncRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["celestia.vision.control.v1"] = "celestia.vision.control.v1"
    sent_at: datetime
    recognition_enabled: bool = True
    callbacks: CallbackPaths
    rules: list[VisionRule]


class SyncResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool = True

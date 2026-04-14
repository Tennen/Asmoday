from datetime import datetime
from typing import Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


KeyEntityId: TypeAlias = str | int


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
    value: str


class KeyEntityImage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    base64: str = Field(min_length=1)
    content_type: str = "image/jpeg"


class KeyEntityReference(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: KeyEntityId
    image: KeyEntityImage | None = None
    description: str | None = Field(default=None, min_length=1)

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: KeyEntityId) -> KeyEntityId:
        if isinstance(value, str) and not value.strip():
            raise ValueError("key entity id must not be blank")
        return value

    @field_validator("description", mode="before")
    @classmethod
    def normalize_description(cls, value: object) -> object:
        if value is None:
            return None
        if isinstance(value, str):
            stripped = value.strip()
            return stripped or None
        return value

    @model_validator(mode="after")
    def validate_reference(self) -> "KeyEntityReference":
        if self.image is None and self.description is None:
            raise ValueError("key entity must provide image or description")
        return self


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
    behavior: str | None = Field(default=None, min_length=1)
    key_entities: list[KeyEntityReference] = Field(default_factory=list)
    zone: ZoneRect
    stay_threshold_seconds: int = Field(ge=1)

    @field_validator("behavior", mode="before")
    @classmethod
    def normalize_behavior(cls, value: object) -> object:
        if value is None:
            return None
        if isinstance(value, str):
            stripped = value.strip()
            return stripped or None
        return value

    @field_validator("key_entities")
    @classmethod
    def validate_unique_key_entity_ids(
        cls,
        value: list[KeyEntityReference],
    ) -> list[KeyEntityReference]:
        seen: set[str] = set()
        for entity in value:
            key = str(entity.id)
            if key in seen:
                raise ValueError(f"duplicate key entity id: {entity.id!r}")
            seen.add(key)
        return value


class SyncRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["celestia.vision.control.ws.v1"] = (
        "celestia.vision.control.ws.v1"
    )
    sent_at: datetime
    recognition_enabled: bool = True
    rules: list[VisionRule]

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class WebSocketEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: str = Field(min_length=1)
    request_id: str | None = None
    payload: dict[str, Any] | None = None


class SessionHelloPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["celestia.vision.ws.v1"] = "celestia.vision.ws.v1"
    service_version: str
    connected_at: datetime

    @classmethod
    def build(cls, *, service_version: str) -> "SessionHelloPayload":
        return cls(
            service_version=service_version,
            connected_at=datetime.now(tz=UTC),
        )


class ErrorPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str = Field(min_length=1)
    message: str = Field(min_length=1)


class EntityCatalogRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_name: str | None = None


class SyncAppliedPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool = True
    applied_at: datetime

    @classmethod
    def build(cls) -> "SyncAppliedPayload":
        return cls(applied_at=datetime.now(tz=UTC))

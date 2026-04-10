from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict


class ModelDescriptor(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    created_at: datetime
    is_selected: bool = False
    is_default: bool = False


class ModelListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["celestia.vision.models.v1"] = "celestia.vision.models.v1"
    service_version: str | None = None
    current_model_name: str | None = None
    default_model_name: str | None = None
    fetched_at: datetime
    models: list[ModelDescriptor]

    @classmethod
    def build(
        cls,
        *,
        models: list[ModelDescriptor],
        service_version: str | None,
        current_model_name: str | None,
        default_model_name: str | None,
    ) -> "ModelListResponse":
        return cls(
            service_version=service_version,
            current_model_name=current_model_name,
            default_model_name=default_model_name,
            fetched_at=datetime.now(tz=UTC),
            models=models,
        )


class ModelSelectionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_name: str | None = None


class ModelSelectionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ok: bool = True
    model_name: str
    changed_at: datetime

    @classmethod
    def build(cls, *, model_name: str) -> "ModelSelectionResponse":
        return cls(
            model_name=model_name,
            changed_at=datetime.now(tz=UTC),
        )

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict


class EntityDescriptor(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["label"] = "label"
    value: str
    display_name: str | None = None


class CatalogResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["celestia.vision.catalog.v1"] = "celestia.vision.catalog.v1"
    service_version: str | None = None
    model_name: str | None = None
    fetched_at: datetime
    entities: list[EntityDescriptor]

    @classmethod
    def build(
        cls,
        *,
        entities: list[EntityDescriptor],
        service_version: str | None,
        model_name: str | None,
    ) -> "CatalogResponse":
        return cls(
            service_version=service_version,
            model_name=model_name,
            fetched_at=datetime.now(tz=UTC),
            entities=entities,
        )

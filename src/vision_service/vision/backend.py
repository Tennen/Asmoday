import asyncio
from typing import Any

from vision_service.contracts import CatalogResponse, EntityDescriptor
from vision_service.settings import Settings


class VisionBackend:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model: Any | None = None
        self._lock = asyncio.Lock()

    async def get_catalog(self, *, service_version: str) -> CatalogResponse:
        model = await self._get_model()
        entities = [
            EntityDescriptor(
                kind="label",
                value=str(label_name),
                display_name=str(label_name).replace("_", " ").title(),
            )
            for _, label_name in sorted(self._iter_model_names(model).items())
        ]
        return CatalogResponse.build(
            entities=entities,
            service_version=service_version,
            model_name=self._settings.model_path,
        )

    async def _get_model(self) -> Any:
        async with self._lock:
            if self._model is None:
                self._model = await asyncio.to_thread(self._load_model)
            return self._model

    def _load_model(self) -> Any:
        from ultralytics import YOLO

        return YOLO(self._settings.model_path)

    @staticmethod
    def _iter_model_names(model: Any) -> dict[int, str]:
        names = getattr(model, "names", {})
        if isinstance(names, dict):
            return {int(index): str(label) for index, label in names.items()}
        if isinstance(names, list):
            return {index: str(label) for index, label in enumerate(names)}
        raise RuntimeError("unsupported model names format")

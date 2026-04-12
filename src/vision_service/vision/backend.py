import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from vision_service.contracts import (
    CatalogResponse,
    EntityDescriptor,
    ModelDescriptor,
    ModelListResponse,
    ModelSelectionResponse,
)
from vision_service.settings import Settings


@dataclass(slots=True, frozen=True)
class DetectionBatch:
    result: Any
    detections: Any
    labels: dict[int, str]


@dataclass(slots=True, frozen=True)
class ModelRecord:
    name: str
    path: Path
    created_at: datetime


class ModelRegistryError(RuntimeError):
    pass


class ModelNotFoundError(ModelRegistryError):
    pass


class VisionBackend:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._models: dict[str, Any] = {}
        self._selected_model_name: str | None = None
        self._lock = asyncio.Lock()

    async def get_catalog(
        self,
        *,
        service_version: str,
        model_name: str | None = None,
    ) -> CatalogResponse:
        model_record, model = await self._get_model_bundle(model_name=model_name)
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
            model_name=model_record.name,
        )

    async def list_models(self, *, service_version: str) -> ModelListResponse:
        async with self._lock:
            model_records = await asyncio.to_thread(self._discover_models)
            default_model_name = model_records[0].name
            current_model_name = self._resolve_model_name(
                model_records=model_records,
                requested_model_name=None,
                selected_model_name=self._selected_model_name,
            )

        return ModelListResponse.build(
            models=[
                ModelDescriptor(
                    name=model_record.name,
                    created_at=model_record.created_at,
                    is_selected=model_record.name == current_model_name,
                    is_default=model_record.name == default_model_name,
                )
                for model_record in model_records
            ],
            service_version=service_version,
            current_model_name=current_model_name,
            default_model_name=default_model_name,
        )

    async def select_model(self, model_name: str | None) -> ModelSelectionResponse:
        async with self._lock:
            model_records = await asyncio.to_thread(self._discover_models)
            if model_name is None:
                self._selected_model_name = None
                return ModelSelectionResponse.build(
                    model_name=model_records[0].name,
                )

            model_record = self._find_model_record(
                model_records=model_records,
                model_name=model_name,
            )
            model = self._models.get(model_record.name)
            if model is None:
                model = await asyncio.to_thread(
                    self._load_model_from_path,
                    model_record.path,
                )
                self._models[model_record.name] = model
            self._selected_model_name = model_record.name

        return ModelSelectionResponse.build(model_name=model_record.name)

    async def reset_model_selection(self) -> None:
        async with self._lock:
            self._selected_model_name = None

    async def current_model_name(self) -> str:
        async with self._lock:
            model_records = await asyncio.to_thread(self._discover_models)
            return self._resolve_model_name(
                model_records=model_records,
                requested_model_name=None,
                selected_model_name=self._selected_model_name,
            )

    async def detect(self, frame: Any) -> DetectionBatch:
        _, model = await self._get_model_bundle(model_name=None)
        result = await asyncio.to_thread(self._predict, model, frame)

        import supervision as sv

        return DetectionBatch(
            result=result,
            detections=sv.Detections.from_ultralytics(result),
            labels=self._iter_model_names(model),
        )

    async def _get_model_bundle(
        self,
        *,
        model_name: str | None,
    ) -> tuple[ModelRecord, Any]:
        async with self._lock:
            model_records = await asyncio.to_thread(self._discover_models)
            resolved_model_name = self._resolve_model_name(
                model_records=model_records,
                requested_model_name=model_name,
                selected_model_name=self._selected_model_name,
            )
            model_record = self._find_model_record(
                model_records=model_records,
                model_name=resolved_model_name,
            )
            model = self._models.get(model_record.name)
            if model is None:
                model = await asyncio.to_thread(
                    self._load_model_from_path,
                    model_record.path,
                )
                self._models[model_record.name] = model
            return model_record, model

    def _discover_models(self) -> list[ModelRecord]:
        model_directory = self._settings.model_directory.expanduser().resolve()
        if not model_directory.exists():
            raise ModelRegistryError(
                f"model directory does not exist: {model_directory}"
            )
        if not model_directory.is_dir():
            raise ModelRegistryError(
                f"model directory is not a directory: {model_directory}"
            )

        model_records = [
            ModelRecord(
                name=path.name,
                path=path,
                created_at=self._model_created_at(path),
            )
            for path in model_directory.iterdir()
            if not path.name.startswith(".") and (path.is_file() or path.is_dir())
        ]
        if not model_records:
            raise ModelRegistryError(
                f"no models found in directory: {model_directory}"
            )
        return sorted(model_records, key=lambda item: (item.created_at, item.name))

    @staticmethod
    def _model_created_at(path: Path) -> datetime:
        stat_result = path.stat()
        created_at = getattr(stat_result, "st_birthtime", stat_result.st_ctime)
        return datetime.fromtimestamp(created_at, tz=UTC)

    def _load_model_from_path(self, model_path: Path) -> Any:
        from ultralytics import YOLO

        return YOLO(str(model_path))

    @staticmethod
    def _resolve_model_name(
        *,
        model_records: list[ModelRecord],
        requested_model_name: str | None,
        selected_model_name: str | None,
    ) -> str:
        if requested_model_name is not None:
            return VisionBackend._find_model_record(
                model_records=model_records,
                model_name=requested_model_name,
            ).name
        if selected_model_name is not None:
            return VisionBackend._find_model_record(
                model_records=model_records,
                model_name=selected_model_name,
            ).name
        return model_records[0].name

    @staticmethod
    def _find_model_record(
        *,
        model_records: list[ModelRecord],
        model_name: str,
    ) -> ModelRecord:
        for model_record in model_records:
            if model_record.name == model_name:
                return model_record
        raise ModelNotFoundError(
            f"model '{model_name}' was not found in directory "
            f"{model_records[0].path.parent}"
        )

    def _predict(self, model: Any, frame: Any) -> Any:
        results = model.predict(
            source=frame,
            conf=self._settings.model_confidence_threshold,
            device=self._settings.model_device,
            verbose=False,
        )
        if not results:
            raise RuntimeError("model returned no detection results")
        return results[0]

    @staticmethod
    def _iter_model_names(model: Any) -> dict[int, str]:
        names = getattr(model, "names", {})
        if isinstance(names, dict):
            return {int(index): str(label) for index, label in names.items()}
        if isinstance(names, list):
            return {index: str(label) for index, label in enumerate(names)}
        raise RuntimeError("unsupported model names format")

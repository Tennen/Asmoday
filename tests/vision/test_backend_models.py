from datetime import UTC, datetime
from pathlib import Path

import pytest

from vision_service.settings import Settings
from vision_service.vision.backend import ModelNotFoundError, VisionBackend


class FakeModel:
    def __init__(self, names: dict[int, str]) -> None:
        self.names = names


def _build_backend(
    *,
    model_directory: Path,
    created_at_by_name: dict[str, datetime],
    labels_by_name: dict[str, dict[int, str]],
    monkeypatch: pytest.MonkeyPatch,
) -> VisionBackend:
    monkeypatch.setattr(
        VisionBackend,
        "_model_created_at",
        staticmethod(lambda path: created_at_by_name[path.name]),
    )
    monkeypatch.setattr(
        VisionBackend,
        "_load_model_from_path",
        lambda self, model_path: FakeModel(labels_by_name[model_path.name]),
    )
    return VisionBackend(Settings(model_directory=model_directory))


@pytest.mark.asyncio
async def test_get_catalog_uses_oldest_model_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "newer.pt").write_bytes(b"newer")
    (tmp_path / "older.pt").write_bytes(b"older")
    created_at_by_name = {
        "newer.pt": datetime(2026, 4, 10, 8, 1, tzinfo=UTC),
        "older.pt": datetime(2026, 4, 10, 8, 0, tzinfo=UTC),
    }
    labels_by_name = {
        "newer.pt": {0: "dog"},
        "older.pt": {0: "cat"},
    }
    backend = _build_backend(
        model_directory=tmp_path,
        created_at_by_name=created_at_by_name,
        labels_by_name=labels_by_name,
        monkeypatch=monkeypatch,
    )

    response = await backend.get_catalog(service_version="0.1.0")

    assert response.model_name == "older.pt"
    assert [entity.value for entity in response.entities] == ["cat"]


@pytest.mark.asyncio
async def test_select_model_marks_current_selection_in_model_list(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "a.pt").write_bytes(b"a")
    (tmp_path / "b.pt").write_bytes(b"b")
    created_at_by_name = {
        "a.pt": datetime(2026, 4, 10, 8, 0, tzinfo=UTC),
        "b.pt": datetime(2026, 4, 10, 8, 1, tzinfo=UTC),
    }
    labels_by_name = {
        "a.pt": {0: "cat"},
        "b.pt": {0: "dog"},
    }
    backend = _build_backend(
        model_directory=tmp_path,
        created_at_by_name=created_at_by_name,
        labels_by_name=labels_by_name,
        monkeypatch=monkeypatch,
    )

    selection = await backend.select_model("b.pt")
    model_list = await backend.list_models(service_version="0.1.0")

    assert selection.model_name == "b.pt"
    assert model_list.current_model_name == "b.pt"
    assert model_list.default_model_name == "a.pt"
    assert [model.name for model in model_list.models] == ["a.pt", "b.pt"]
    assert [model.is_selected for model in model_list.models] == [False, True]
    assert [model.is_default for model in model_list.models] == [True, False]


@pytest.mark.asyncio
async def test_requested_model_must_exist_in_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "only.pt").write_bytes(b"only")
    created_at_by_name = {
        "only.pt": datetime(2026, 4, 10, 8, 0, tzinfo=UTC),
    }
    labels_by_name = {
        "only.pt": {0: "cat"},
    }
    backend = _build_backend(
        model_directory=tmp_path,
        created_at_by_name=created_at_by_name,
        labels_by_name=labels_by_name,
        monkeypatch=monkeypatch,
    )

    with pytest.raises(ModelNotFoundError, match="missing.pt"):
        await backend.get_catalog(
            service_version="0.1.0",
            model_name="missing.pt",
        )

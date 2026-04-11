import logging
from dataclasses import dataclass
from datetime import UTC, datetime

import pytest

from vision_service.contracts import (
    CameraIdentity,
    EntitySelector,
    RTSPSource,
    SyncRequest,
    VisionRule,
    ZoneRect,
)
from vision_service.runtime.manager import RuntimeManager
from vision_service.settings import Settings
from vision_service.vision.stream import StreamReadResult, StreamSnapshot


class FakeBackend:
    async def detect(self, frame):  # noqa: ANN001, ANN201
        raise AssertionError("not used in manager tests")


@dataclass
class FakeStream:
    url: str
    started: int = 0
    stopped: int = 0

    async def start(self) -> None:
        self.started += 1

    async def stop(self) -> None:
        self.stopped += 1

    async def wait_for_result(self, *, after_token: int | None) -> StreamReadResult | None:
        return None

    def snapshot(self) -> StreamSnapshot:
        return StreamSnapshot(
            url=self.url,
            state="running",
            last_frame_at=datetime.now(tz=UTC),
            last_error=None,
        )


@dataclass
class FakeWorker:
    rule: VisionRule
    frame_stream: FakeStream
    started: int = 0
    stopped: int = 0

    @property
    def rule_id(self) -> str:
        return self.rule.id

    def matches(self, rule: VisionRule) -> bool:
        return self.rule.model_dump(mode="json") == rule.model_dump(mode="json")

    def snapshot(self):  # noqa: ANN202
        from vision_service.vision.pipeline import WorkerSnapshot

        return WorkerSnapshot(
            rule_id=self.rule.id,
            camera_device_id=self.rule.camera.device_id,
            state="running",
            active=False,
            last_frame_at=None,
            last_error=None,
            emitted_threshold_events=0,
        )

    async def start(self) -> None:
        self.started += 1

    async def stop(self) -> None:
        self.stopped += 1


class RecordingRuntimeManager(RuntimeManager):
    def __init__(self) -> None:
        super().__init__(
            settings=Settings(),
            backend=FakeBackend(),
        )
        self.created_streams: list[FakeStream] = []
        self.created_workers: list[FakeWorker] = []

    def _create_stream(self, *, url: str) -> FakeStream:  # type: ignore[override]
        stream = FakeStream(url=url)
        self.created_streams.append(stream)
        return stream

    def _create_worker(  # type: ignore[override]
        self,
        *,
        rule: VisionRule,
        frame_stream: FakeStream,
    ) -> FakeWorker:
        worker = FakeWorker(rule=rule, frame_stream=frame_stream)
        self.created_workers.append(worker)
        return worker


def build_rule(*, rule_id: str, url: str) -> VisionRule:
    return VisionRule(
        id=rule_id,
        name=rule_id,
        enabled=True,
        camera=CameraIdentity(device_id=f"camera:{rule_id}"),
        rtsp_source=RTSPSource(url=url),
        entity_selector=EntitySelector(value="cat"),
        zone=ZoneRect(x=0.1, y=0.1, width=0.2, height=0.2),
        stay_threshold_seconds=5,
    )


def build_payload(
    *,
    rules: list[VisionRule],
    recognition_enabled: bool = True,
) -> SyncRequest:
    return SyncRequest(
        schema_version="celestia.vision.control.ws.v1",
        sent_at=datetime.now(tz=UTC),
        recognition_enabled=recognition_enabled,
        rules=rules,
    )


@pytest.mark.asyncio
async def test_manager_reuses_single_stream_for_rules_with_same_url() -> None:
    manager = RecordingRuntimeManager()
    payload = build_payload(
        rules=[
            build_rule(rule_id="rule-1", url="rtsp://camera/shared"),
            build_rule(rule_id="rule-2", url="rtsp://camera/shared"),
        ],
    )

    await manager.apply_config(payload)

    assert len(manager.created_streams) == 1
    assert len(manager.created_workers) == 2
    assert manager.created_workers[0].frame_stream is manager.created_workers[1].frame_stream

    status = await manager.snapshot_status()
    assert status.runtime["active_streams"] == 1


@pytest.mark.asyncio
async def test_manager_stops_unused_streams_after_reconcile() -> None:
    manager = RecordingRuntimeManager()
    shared_url = "rtsp://camera/shared"

    await manager.apply_config(
        build_payload(
            rules=[
                build_rule(rule_id="rule-1", url=shared_url),
                build_rule(rule_id="rule-2", url=shared_url),
            ],
        )
    )

    shared_stream = manager.created_streams[0]

    await manager.apply_config(
        build_payload(
            rules=[build_rule(rule_id="rule-1", url="rtsp://camera/other")]
        )
    )

    assert shared_stream.stopped == 1
    assert len(manager.created_streams) == 2


@pytest.mark.asyncio
async def test_manager_logs_when_sync_has_no_runnable_rules(caplog) -> None:
    manager = RecordingRuntimeManager()
    caplog.set_level(logging.INFO, logger="vision_service.runtime.manager")

    await manager.apply_config(
        build_payload(
            rules=[build_rule(rule_id="rule-1", url="rtsp://camera/shared")],
            recognition_enabled=False,
        )
    )

    assert len(manager.created_streams) == 0
    assert len(manager.created_workers) == 0
    assert "reconciling config sync" in caplog.text
    assert "config sync left no runnable rules" in caplog.text


@pytest.mark.asyncio
async def test_manager_clear_session_stops_workers_and_streams() -> None:
    manager = RecordingRuntimeManager()
    payload = build_payload(
        rules=[build_rule(rule_id="rule-1", url="rtsp://camera/shared")]
    )

    await manager.apply_config(payload)
    shared_stream = manager.created_streams[0]
    worker = manager.created_workers[0]

    await manager.clear_session(reason="gateway websocket disconnected")

    assert shared_stream.stopped == 1
    assert worker.stopped == 1
    assert await manager.current_config() is None

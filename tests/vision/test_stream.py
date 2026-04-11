import logging
import threading

import numpy as np
import pytest

from vision_service.settings import Settings
from vision_service.vision.stream import READ_FAILURE_LOG_INTERVAL, SharedRTSPStream


class FakeCapture:
    def __init__(self) -> None:
        self._reads = [
            (True, np.zeros((4, 4, 3), dtype=np.uint8)),
            (False, None),
        ]
        self.released = False

    def isOpened(self) -> bool:
        return True

    def read(self):  # noqa: ANN201
        if self._reads:
            return self._reads.pop(0)
        return False, None

    def release(self) -> None:
        self.released = True


@pytest.mark.asyncio
async def test_shared_rtsp_stream_publishes_frames_and_failures(monkeypatch) -> None:
    capture = FakeCapture()

    def fake_open_rtsp_capture(*, url: str, settings: Settings) -> FakeCapture:
        assert url == "rtsp://camera/stream"
        assert settings.rtsp_transport == "tcp"
        return capture

    monkeypatch.setattr(
        "vision_service.vision.stream.open_rtsp_capture",
        fake_open_rtsp_capture,
    )

    stream = SharedRTSPStream(
        url="rtsp://camera/stream",
        settings=Settings(
            rtsp_transport="tcp",
            frame_failure_backoff_seconds=0.001,
        ),
    )
    await stream.start()

    first = await stream.wait_for_result(after_token=None)
    assert first is not None
    assert first.error is None
    assert first.frame is not None
    assert first.frame.shape == (4, 4, 3)

    second = await stream.wait_for_result(after_token=first.token)
    assert second is not None
    assert second.frame is None
    assert second.error is not None
    assert "failed to read frame" in second.error
    assert stream.snapshot().state == "degraded"

    await stream.stop()

    assert capture.released is True


@pytest.mark.asyncio
async def test_shared_rtsp_stream_uses_single_thread_for_capture_lifecycle(
    monkeypatch,
) -> None:
    class ThreadAwareCapture:
        def __init__(self) -> None:
            self.created_thread_id = threading.get_ident()
            self.read_thread_ids: list[int] = []
            self.release_thread_id: int | None = None
            self._reads = [
                (True, np.zeros((2, 2, 3), dtype=np.uint8)),
                (False, None),
            ]

        def isOpened(self) -> bool:
            return True

        def read(self):  # noqa: ANN201
            self.read_thread_ids.append(threading.get_ident())
            if self._reads:
                return self._reads.pop(0)
            return False, None

        def release(self) -> None:
            self.release_thread_id = threading.get_ident()

    created: list[ThreadAwareCapture] = []

    def fake_open_rtsp_capture(*, url: str, settings: Settings) -> ThreadAwareCapture:
        assert url == "rtsp://camera/thread-aware"
        capture = ThreadAwareCapture()
        created.append(capture)
        return capture

    monkeypatch.setattr(
        "vision_service.vision.stream.open_rtsp_capture",
        fake_open_rtsp_capture,
    )

    stream = SharedRTSPStream(
        url="rtsp://camera/thread-aware",
        settings=Settings(
            frame_failure_backoff_seconds=0.001,
        ),
    )
    await stream.start()

    first = await stream.wait_for_result(after_token=None)
    assert first is not None

    second = await stream.wait_for_result(after_token=first.token)
    assert second is not None

    await stream.stop()

    capture = created[0]
    assert capture.read_thread_ids
    assert set(capture.read_thread_ids) == {capture.created_thread_id}
    assert capture.release_thread_id == capture.created_thread_id
    assert stream.snapshot().state == "stopped"


@pytest.mark.asyncio
async def test_shared_rtsp_stream_logs_repeated_failures(monkeypatch, caplog) -> None:
    class FailingCapture:
        def __init__(self) -> None:
            self.released = False

        def isOpened(self) -> bool:
            return True

        def read(self):  # noqa: ANN201
            return False, None

        def release(self) -> None:
            self.released = True

    capture = FailingCapture()

    def fake_open_rtsp_capture(*, url: str, settings: Settings) -> FailingCapture:
        assert url == "rtsp://camera/failing"
        return capture

    monkeypatch.setattr(
        "vision_service.vision.stream.open_rtsp_capture",
        fake_open_rtsp_capture,
    )
    monkeypatch.setattr(
        "vision_service.vision.stream.READ_FAILURE_LOG_INTERVAL",
        2,
    )
    caplog.set_level(logging.WARNING, logger="vision_service.vision.stream")

    stream = SharedRTSPStream(
        url="rtsp://camera/failing",
        settings=Settings(
            frame_failure_backoff_seconds=0.001,
        ),
    )
    await stream.start()

    first = await stream.wait_for_result(after_token=None)
    second = await stream.wait_for_result(after_token=first.token if first else None)

    assert first is not None
    assert second is not None
    assert "consecutive_failures=1" in caplog.text
    assert "consecutive_failures=2" in caplog.text

    await stream.stop()

    assert capture.released is True

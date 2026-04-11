import logging
import threading
from typing import Any

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
            rtsp_reconnect_failure_threshold=99,
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
            rtsp_reconnect_failure_threshold=99,
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
            rtsp_reconnect_failure_threshold=99,
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


@pytest.mark.asyncio
async def test_shared_rtsp_stream_reconnects_after_repeated_failures(
    monkeypatch,
) -> None:
    class ReconnectCapture:
        def __init__(self, reads: list[tuple[bool, np.ndarray[Any, Any] | None]]) -> None:
            self._reads = reads
            self.released = False

        def isOpened(self) -> bool:
            return True

        def read(self):  # noqa: ANN201
            if self._reads:
                return self._reads.pop(0)
            return False, None

        def release(self) -> None:
            self.released = True

    first_capture = ReconnectCapture(
        [
            (True, np.zeros((3, 3, 3), dtype=np.uint8)),
            (False, None),
            (False, None),
        ]
    )
    second_capture = ReconnectCapture(
        [
            (True, np.ones((3, 3, 3), dtype=np.uint8)),
            (False, None),
        ]
    )
    captures = [first_capture, second_capture]

    def fake_open_rtsp_capture(*, url: str, settings: Settings) -> ReconnectCapture:
        assert url == "rtsp://camera/reconnect"
        return captures.pop(0)

    monkeypatch.setattr(
        "vision_service.vision.stream.open_rtsp_capture",
        fake_open_rtsp_capture,
    )

    stream = SharedRTSPStream(
        url="rtsp://camera/reconnect",
        settings=Settings(
            frame_failure_backoff_seconds=0.001,
            rtsp_reconnect_failure_threshold=2,
            rtsp_reconnect_backoff_seconds=0.001,
            rtsp_reconnect_max_attempts=2,
        ),
    )
    await stream.start()

    first = await stream.wait_for_result(after_token=None)
    second = await stream.wait_for_result(after_token=first.token if first else None)
    third = await stream.wait_for_result(after_token=second.token if second else None)

    assert first is not None
    assert first.frame is not None
    assert second is not None
    assert second.frame is None
    assert third is not None
    assert third.frame is None

    recovered = await stream.wait_for_result(after_token=third.token)

    assert recovered is not None
    assert recovered.frame is not None
    assert np.all(recovered.frame == 1)
    assert stream.snapshot().state == "running"

    await stream.stop()

    assert first_capture.released is True
    assert second_capture.released is True


@pytest.mark.asyncio
async def test_shared_rtsp_stream_stops_after_reconnect_attempts_are_exhausted(
    monkeypatch,
) -> None:
    class ReconnectCapture:
        def __init__(
            self,
            *,
            opened: bool,
            reads: list[tuple[bool, np.ndarray[Any, Any] | None]] | None = None,
        ) -> None:
            self._opened = opened
            self._reads = reads or []
            self.released = False

        def isOpened(self) -> bool:
            return self._opened

        def read(self):  # noqa: ANN201
            if self._reads:
                return self._reads.pop(0)
            return False, None

        def release(self) -> None:
            self.released = True

    first_capture = ReconnectCapture(
        opened=True,
        reads=[
            (True, np.zeros((2, 2, 3), dtype=np.uint8)),
            (False, None),
            (False, None),
        ],
    )
    failed_reconnect_captures = [
        ReconnectCapture(opened=False),
        ReconnectCapture(opened=False),
    ]
    captures = [first_capture, *failed_reconnect_captures]

    def fake_open_rtsp_capture(*, url: str, settings: Settings) -> ReconnectCapture:
        assert url == "rtsp://camera/exhausted"
        return captures.pop(0)

    monkeypatch.setattr(
        "vision_service.vision.stream.open_rtsp_capture",
        fake_open_rtsp_capture,
    )

    stream = SharedRTSPStream(
        url="rtsp://camera/exhausted",
        settings=Settings(
            frame_failure_backoff_seconds=0.001,
            rtsp_reconnect_failure_threshold=2,
            rtsp_reconnect_backoff_seconds=0.001,
            rtsp_reconnect_max_attempts=2,
        ),
    )
    await stream.start()

    first = await stream.wait_for_result(after_token=None)
    second = await stream.wait_for_result(after_token=first.token if first else None)
    third = await stream.wait_for_result(after_token=second.token if second else None)
    terminal = await stream.wait_for_result(after_token=third.token if third else None)

    assert first is not None
    assert second is not None
    assert third is not None
    assert terminal is not None
    assert terminal.frame is None
    assert terminal.error is not None
    assert "unable to reconnect RTSP stream after 2 attempts" in terminal.error
    assert stream.snapshot().state == "degraded"

    await stream.stop()

    assert first_capture.released is True
    assert all(capture.released is True for capture in failed_reconnect_captures)

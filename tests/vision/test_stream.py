import numpy as np
import pytest

from vision_service.settings import Settings
from vision_service.vision.stream import SharedRTSPStream


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

    await stream.stop()

    assert capture.released is True
    assert stream.snapshot().state == "stopped"

import os
import sys
from types import SimpleNamespace

from vision_service.settings import Settings
from vision_service.vision.capture import (
    configure_rtsp_capture_environment,
    open_rtsp_capture,
)


def test_configure_rtsp_capture_environment_sets_transport(monkeypatch) -> None:
    monkeypatch.delenv("OPENCV_FFMPEG_CAPTURE_OPTIONS", raising=False)

    configure_rtsp_capture_environment(Settings(rtsp_transport="tcp"))

    assert os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] == "rtsp_transport;tcp"


def test_open_rtsp_capture_uses_ffmpeg_backend_and_timeouts(monkeypatch) -> None:
    calls: list[tuple[str, int, list[int]]] = []
    fake_cv2 = SimpleNamespace(
        CAP_FFMPEG=1900,
        CAP_PROP_OPEN_TIMEOUT_MSEC=53,
        CAP_PROP_READ_TIMEOUT_MSEC=54,
    )

    def fake_video_capture(url: str, backend: int, params: list[int]) -> str:
        calls.append((url, backend, params))
        return "capture"

    fake_cv2.VideoCapture = fake_video_capture
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)
    monkeypatch.delenv("OPENCV_FFMPEG_CAPTURE_OPTIONS", raising=False)

    capture = open_rtsp_capture(
        url="rtsp://camera/stream",
        settings=Settings(
            rtsp_transport="udp",
            rtsp_open_timeout_msec=1_500,
            rtsp_read_timeout_msec=3_500,
        ),
    )

    assert capture == "capture"
    assert calls == [
        (
            "rtsp://camera/stream",
            1900,
            [53, 1_500, 54, 3_500],
        )
    ]
    assert os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] == "rtsp_transport;udp"

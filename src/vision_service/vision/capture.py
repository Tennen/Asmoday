import os
from typing import Any

from vision_service.settings import Settings


def configure_rtsp_capture_environment(settings: Settings) -> None:
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
        f"rtsp_transport;{settings.rtsp_transport}"
    )


def open_rtsp_capture(*, url: str, settings: Settings) -> Any:
    import cv2

    configure_rtsp_capture_environment(settings)
    return cv2.VideoCapture(
        url,
        cv2.CAP_FFMPEG,
        [
            cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
            settings.rtsp_open_timeout_msec,
            cv2.CAP_PROP_READ_TIMEOUT_MSEC,
            settings.rtsp_read_timeout_msec,
        ],
    )

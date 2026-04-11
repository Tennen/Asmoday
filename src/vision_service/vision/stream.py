import asyncio
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal, Protocol

import numpy as np

from vision_service.settings import Settings
from vision_service.vision.capture import open_rtsp_capture

logger = logging.getLogger(__name__)

StreamState = Literal["starting", "running", "stopped", "degraded"]


@dataclass(slots=True, frozen=True)
class StreamReadResult:
    token: int
    observed_at: datetime
    frame: np.ndarray[Any, Any] | None
    error: str | None


@dataclass(slots=True, frozen=True)
class StreamSnapshot:
    url: str
    state: StreamState
    last_frame_at: datetime | None
    last_error: str | None


class FrameStream(Protocol):
    async def wait_for_result(
        self,
        *,
        after_token: int | None,
    ) -> StreamReadResult | None: ...


class SharedRTSPStream:
    def __init__(self, *, url: str, settings: Settings) -> None:
        self._url = url
        self._settings = settings

        self._state: StreamState = "starting"
        self._last_frame_at: datetime | None = None
        self._last_error: str | None = None

        self._task: asyncio.Task[None] | None = None
        self._stop_event = asyncio.Event()
        self._condition = asyncio.Condition()
        self._latest_result: StreamReadResult | None = None
        self._result_token = 0

    @property
    def url(self) -> str:
        return self._url

    def snapshot(self) -> StreamSnapshot:
        return StreamSnapshot(
            url=self._url,
            state=self._state,
            last_frame_at=self._last_frame_at,
            last_error=self._last_error,
        )

    async def start(self) -> None:
        if self._task is not None:
            return
        self._stop_event.clear()
        self._state = "starting"
        logger.info(
            "starting RTSP stream url=%s rtsp_transport=%s",
            self._url,
            self._settings.rtsp_transport,
        )
        self._task = asyncio.create_task(self._run_wrapper())

    async def stop(self) -> None:
        if self._task is None:
            self._state = "stopped"
            await self._notify_waiters()
            logger.info("stopped RTSP stream url=%s", self._url)
            return

        self._stop_event.set()
        await self._task
        logger.info("stopped RTSP stream url=%s", self._url)

    async def wait_for_result(
        self,
        *,
        after_token: int | None,
    ) -> StreamReadResult | None:
        async with self._condition:
            await self._condition.wait_for(
                lambda: self._has_update(after_token=after_token),
            )
            result = self._latest_result
            if result is None:
                return None
            if after_token is not None and result.token <= after_token:
                return None
            return StreamReadResult(
                token=result.token,
                observed_at=result.observed_at,
                frame=None if result.frame is None else result.frame.copy(),
                error=result.error,
            )

    def _has_update(self, *, after_token: int | None) -> bool:
        result = self._latest_result
        if result is not None and (after_token is None or result.token > after_token):
            return True
        return self._task is None and self._state in {"stopped", "degraded"}

    async def _run_wrapper(self) -> None:
        try:
            await self._run()
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "RTSP stream degraded url=%s rtsp_transport=%s",
                self._url,
                self._settings.rtsp_transport,
            )
            await self._publish_terminal_error(str(exc))
        else:
            self._state = "stopped"
        finally:
            self._task = None
            await self._notify_waiters()

    async def _run(self) -> None:
        capture = await asyncio.to_thread(
            open_rtsp_capture,
            url=self._url,
            settings=self._settings,
        )
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(
                "unable to open RTSP stream with "
                f"rtsp_transport={self._settings.rtsp_transport}"
            )

        self._state = "running"
        logger.info(
            "RTSP stream running url=%s rtsp_transport=%s",
            self._url,
            self._settings.rtsp_transport,
        )
        await self._notify_waiters()
        read_failure_active = False

        try:
            while not self._stop_event.is_set():
                success, frame = await asyncio.to_thread(capture.read)
                observed_at = datetime.now(tz=UTC)
                if success:
                    if read_failure_active:
                        logger.info(
                            "RTSP stream recovered url=%s rtsp_transport=%s",
                            self._url,
                            self._settings.rtsp_transport,
                        )
                        read_failure_active = False
                    await self._publish_result(
                        observed_at=observed_at,
                        frame=frame,
                        error=None,
                    )
                    continue

                if not read_failure_active:
                    logger.warning(
                        "RTSP frame read failed url=%s rtsp_transport=%s",
                        self._url,
                        self._settings.rtsp_transport,
                    )
                    read_failure_active = True
                await self._publish_result(
                    observed_at=observed_at,
                    frame=None,
                    error=(
                        "failed to read frame from RTSP stream with "
                        f"rtsp_transport={self._settings.rtsp_transport}"
                    ),
                )
                await asyncio.sleep(self._settings.frame_failure_backoff_seconds)
        finally:
            await asyncio.to_thread(capture.release)

    async def _publish_result(
        self,
        *,
        observed_at: datetime,
        frame: np.ndarray[Any, Any] | None,
        error: str | None,
    ) -> None:
        async with self._condition:
            self._result_token += 1
            if frame is not None:
                self._last_frame_at = observed_at
            self._last_error = error
            self._latest_result = StreamReadResult(
                token=self._result_token,
                observed_at=observed_at,
                frame=frame,
                error=error,
            )
            self._condition.notify_all()

    async def _publish_terminal_error(self, message: str) -> None:
        self._state = "degraded"
        await self._publish_result(
            observed_at=datetime.now(tz=UTC),
            frame=None,
            error=message,
        )

    async def _notify_waiters(self) -> None:
        async with self._condition:
            self._condition.notify_all()

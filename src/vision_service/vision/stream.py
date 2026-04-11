import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from functools import partial
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal, Protocol

import numpy as np

from vision_service.settings import Settings
from vision_service.vision.capture import open_rtsp_capture

logger = logging.getLogger(__name__)
READ_FAILURE_LOG_INTERVAL = 30

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
        self._executor: ThreadPoolExecutor | None = None
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
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="vision-rtsp",
            )
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
            await self._shutdown_executor()

    async def _run(self) -> None:
        capture = await self._open_capture()

        self._state = "running"
        logger.info(
            "RTSP stream running url=%s rtsp_transport=%s",
            self._url,
            self._settings.rtsp_transport,
        )
        await self._notify_waiters()
        consecutive_read_failures = 0
        outage_read_failures = 0
        outage_reconnect_attempts = 0
        first_failure_at: datetime | None = None

        try:
            while not self._stop_event.is_set():
                success, frame = await self._run_capture_call(capture.read)
                observed_at = datetime.now(tz=UTC)
                if success:
                    if self._last_frame_at is None:
                        logger.info(
                            "RTSP first frame received url=%s rtsp_transport=%s "
                            "frame_height=%s frame_width=%s",
                            self._url,
                            self._settings.rtsp_transport,
                            frame.shape[0],
                            frame.shape[1],
                        )
                    if outage_read_failures > 0 or outage_reconnect_attempts > 0:
                        logger.info(
                            "RTSP stream recovered url=%s rtsp_transport=%s "
                            "read_failures=%s reconnect_attempts=%s "
                            "outage_seconds=%.3f",
                            self._url,
                            self._settings.rtsp_transport,
                            outage_read_failures,
                            outage_reconnect_attempts,
                            self._failure_duration_seconds(
                                first_failure_at=first_failure_at,
                                observed_at=observed_at,
                            ),
                        )
                    consecutive_read_failures = 0
                    outage_read_failures = 0
                    outage_reconnect_attempts = 0
                    first_failure_at = None
                    self._state = "running"
                    await self._publish_result(
                        observed_at=observed_at,
                        frame=frame,
                        error=None,
                    )
                    continue

                consecutive_read_failures += 1
                outage_read_failures += 1
                if first_failure_at is None:
                    first_failure_at = observed_at
                self._state = "degraded"
                error_message = (
                    "failed to read frame from RTSP stream with "
                    f"rtsp_transport={self._settings.rtsp_transport}"
                )
                if (
                    consecutive_read_failures == 1
                    or consecutive_read_failures % READ_FAILURE_LOG_INTERVAL == 0
                ):
                    logger.warning(
                        "RTSP frame read failed url=%s rtsp_transport=%s "
                        "consecutive_failures=%s seconds_since_last_frame=%s",
                        self._url,
                        self._settings.rtsp_transport,
                        consecutive_read_failures,
                        self._seconds_since_last_frame(observed_at=observed_at),
                    )
                await self._publish_result(
                    observed_at=observed_at,
                    frame=None,
                    error=error_message,
                )
                if (
                    consecutive_read_failures
                    < self._settings.rtsp_reconnect_failure_threshold
                ):
                    await asyncio.sleep(self._settings.frame_failure_backoff_seconds)
                    continue

                capture, outage_reconnect_attempts = await self._reconnect_capture(
                    capture=capture,
                    consecutive_read_failures=consecutive_read_failures,
                    outage_read_failures=outage_read_failures,
                    outage_reconnect_attempts=outage_reconnect_attempts,
                    first_failure_at=first_failure_at,
                    observed_at=observed_at,
                )
                consecutive_read_failures = 0
                await self._notify_waiters()
                continue
        finally:
            with suppress(Exception):
                await self._release_capture(capture)

    async def _open_capture(self) -> Any:
        capture, error = await self._attempt_open_capture()
        if capture is None:
            raise RuntimeError(error)
        return capture

    async def _attempt_open_capture(self) -> tuple[Any | None, str]:
        try:
            capture = await self._run_capture_call(
                open_rtsp_capture,
                url=self._url,
                settings=self._settings,
            )
        except Exception as exc:  # noqa: BLE001
            return None, str(exc)

        if capture.isOpened():
            return capture, ""

        await self._release_capture(capture)
        return (
            None,
            "unable to open RTSP stream with "
            f"rtsp_transport={self._settings.rtsp_transport}",
        )

    async def _reconnect_capture(
        self,
        *,
        capture: Any,
        consecutive_read_failures: int,
        outage_read_failures: int,
        outage_reconnect_attempts: int,
        first_failure_at: datetime | None,
        observed_at: datetime,
    ) -> tuple[Any, int]:
        await self._release_capture(capture)
        reconnect_attempt = outage_reconnect_attempts

        while reconnect_attempt < self._settings.rtsp_reconnect_max_attempts:
            reconnect_attempt += 1
            logger.warning(
                "RTSP reconnecting url=%s rtsp_transport=%s attempt=%s "
                "max_attempts=%s consecutive_failures=%s outage_read_failures=%s "
                "outage_seconds=%.3f",
                self._url,
                self._settings.rtsp_transport,
                reconnect_attempt,
                self._settings.rtsp_reconnect_max_attempts,
                consecutive_read_failures,
                outage_read_failures,
                self._failure_duration_seconds(
                    first_failure_at=first_failure_at,
                    observed_at=observed_at,
                ),
            )
            await asyncio.sleep(self._settings.rtsp_reconnect_backoff_seconds)
            if self._stop_event.is_set():
                return capture, reconnect_attempt

            reopened_capture, error = await self._attempt_open_capture()
            if reopened_capture is not None:
                logger.info(
                    "RTSP reconnect opened stream url=%s rtsp_transport=%s "
                    "attempt=%s outage_seconds=%.3f",
                    self._url,
                    self._settings.rtsp_transport,
                    reconnect_attempt,
                    self._failure_duration_seconds(
                        first_failure_at=first_failure_at,
                        observed_at=datetime.now(tz=UTC),
                    ),
                )
                return reopened_capture, reconnect_attempt

            logger.warning(
                "RTSP reconnect attempt failed url=%s rtsp_transport=%s "
                "attempt=%s max_attempts=%s error=%s",
                self._url,
                self._settings.rtsp_transport,
                reconnect_attempt,
                self._settings.rtsp_reconnect_max_attempts,
                error,
            )

        raise RuntimeError(
            "unable to reconnect RTSP stream after "
            f"{self._settings.rtsp_reconnect_max_attempts} attempts with "
            f"rtsp_transport={self._settings.rtsp_transport}"
        )

    async def _release_capture(self, capture: Any) -> None:
        await self._run_capture_call(capture.release)

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

    def _seconds_since_last_frame(self, *, observed_at: datetime) -> str:
        if self._last_frame_at is None:
            return "never"
        return f"{(observed_at - self._last_frame_at).total_seconds():.3f}"

    @staticmethod
    def _failure_duration_seconds(
        *,
        first_failure_at: datetime | None,
        observed_at: datetime,
    ) -> float:
        if first_failure_at is None:
            return 0.0
        return max(0.0, (observed_at - first_failure_at).total_seconds())

    async def _run_capture_call(self, func, /, *args, **kwargs):  # noqa: ANN202
        loop = asyncio.get_running_loop()
        executor = self._executor
        if executor is None:
            raise RuntimeError("RTSP stream executor is not initialized")
        return await loop.run_in_executor(
            executor,
            partial(func, *args, **kwargs),
        )

    async def _shutdown_executor(self) -> None:
        executor = self._executor
        if executor is None:
            return
        self._executor = None
        await asyncio.to_thread(executor.shutdown, True)

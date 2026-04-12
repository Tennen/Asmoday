import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol

import numpy as np

from vision_service.vision.backend import DetectionBatch, VisionBackend
from vision_service.vision.stream import FrameStream, StreamReadResult


@dataclass(slots=True, frozen=True)
class AnalyzedFrameResult:
    token: int
    observed_at: datetime
    frame: np.ndarray[Any, Any] | None
    batch: DetectionBatch | None
    error: str | None


class AnalyzedFrameStream(Protocol):
    async def wait_for_result(
        self,
        *,
        after_token: int | None,
    ) -> AnalyzedFrameResult | None: ...


class SharedInferenceStream:
    def __init__(
        self,
        *,
        frame_stream: FrameStream,
        backend: VisionBackend,
    ) -> None:
        self._frame_stream = frame_stream
        self._backend = backend
        self._lock = asyncio.Lock()
        self._inflight_by_token: dict[int, asyncio.Future[AnalyzedFrameResult]] = {}
        self._latest_result: AnalyzedFrameResult | None = None
        self._latest_failure_token: int | None = None
        self._latest_failure: Exception | None = None

    async def wait_for_result(
        self,
        *,
        after_token: int | None,
    ) -> AnalyzedFrameResult | None:
        stream_result = await self._frame_stream.wait_for_result(after_token=after_token)
        if stream_result is None:
            return None
        if stream_result.frame is None:
            return AnalyzedFrameResult(
                token=stream_result.token,
                observed_at=stream_result.observed_at,
                frame=None,
                batch=None,
                error=stream_result.error,
            )

        future, should_analyze = await self._future_for_token(stream_result)
        if should_analyze:
            await self._analyze(stream_result=stream_result, future=future)
        analyzed = await future
        return self._copy_result(analyzed)

    async def _future_for_token(
        self,
        stream_result: StreamReadResult,
    ) -> tuple[asyncio.Future[AnalyzedFrameResult], bool]:
        async with self._lock:
            latest_result = self._latest_result
            if latest_result is not None and latest_result.token == stream_result.token:
                future: asyncio.Future[AnalyzedFrameResult] = asyncio.get_running_loop().create_future()
                future.set_result(latest_result)
                return future, False

            if self._latest_failure_token == stream_result.token and self._latest_failure is not None:
                future = asyncio.get_running_loop().create_future()
                future.set_exception(self._latest_failure)
                return future, False

            future = self._inflight_by_token.get(stream_result.token)
            if future is not None:
                return future, False

            future = asyncio.get_running_loop().create_future()
            self._inflight_by_token[stream_result.token] = future
            return future, True

    async def _analyze(
        self,
        *,
        stream_result: StreamReadResult,
        future: asyncio.Future[AnalyzedFrameResult],
    ) -> None:
        try:
            batch = await self._backend.detect(stream_result.frame)
            analyzed = AnalyzedFrameResult(
                token=stream_result.token,
                observed_at=stream_result.observed_at,
                frame=stream_result.frame,
                batch=batch,
                error=None,
            )
        except Exception as exc:
            async with self._lock:
                self._latest_failure_token = stream_result.token
                self._latest_failure = exc
                self._inflight_by_token.pop(stream_result.token, None)
            future.set_exception(exc)
            return

        async with self._lock:
            self._latest_result = analyzed
            self._latest_failure_token = None
            self._latest_failure = None
            self._inflight_by_token.pop(stream_result.token, None)
        future.set_result(analyzed)

    @staticmethod
    def _copy_result(result: AnalyzedFrameResult) -> AnalyzedFrameResult:
        return AnalyzedFrameResult(
            token=result.token,
            observed_at=result.observed_at,
            frame=None if result.frame is None else result.frame.copy(),
            batch=None if result.batch is None else result.batch.clone(),
            error=result.error,
        )

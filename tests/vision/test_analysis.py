import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np
import pytest

from vision_service.vision.analysis import SharedInferenceStream
from vision_service.vision.stream import StreamReadResult


@dataclass
class FakeBatch:
    token: int

    def clone(self) -> "FakeBatch":
        return FakeBatch(token=self.token)


class FakeFrameStream:
    def __init__(self) -> None:
        self._result = StreamReadResult(
            token=1,
            observed_at=datetime(2026, 4, 12, 16, 0, tzinfo=UTC),
            frame=np.ones((4, 4, 3), dtype=np.uint8),
            error=None,
        )

    async def wait_for_result(
        self,
        *,
        after_token: int | None,
        require_detection: bool = True,
    ) -> StreamReadResult | None:
        return StreamReadResult(
            token=self._result.token,
            observed_at=self._result.observed_at,
            frame=self._result.frame.copy(),
            error=self._result.error,
        )


class CountingBackend:
    def __init__(self) -> None:
        self.detect_calls = 0

    async def detect(self, frame):  # noqa: ANN001, ANN201
        self.detect_calls += 1
        await asyncio.sleep(0.01)
        return FakeBatch(token=self.detect_calls)


@pytest.mark.asyncio
async def test_shared_inference_stream_runs_single_detection_for_same_frame() -> None:
    backend = CountingBackend()
    stream = SharedInferenceStream(
        frame_stream=FakeFrameStream(),
        backend=backend,  # type: ignore[arg-type]
    )

    first, second = await asyncio.gather(
        stream.wait_for_result(after_token=None),
        stream.wait_for_result(after_token=None),
    )

    assert first is not None
    assert second is not None
    assert backend.detect_calls == 1
    assert first.token == second.token == 1
    assert first.batch is not None
    assert second.batch is not None
    assert first.batch is not second.batch
    assert first.frame is not None
    assert second.frame is not None
    assert first.frame is not second.frame


@pytest.mark.asyncio
async def test_shared_inference_stream_reuses_cached_detection_for_same_token() -> None:
    backend = CountingBackend()
    stream = SharedInferenceStream(
        frame_stream=FakeFrameStream(),
        backend=backend,  # type: ignore[arg-type]
    )

    first = await stream.wait_for_result(after_token=None)
    second = await stream.wait_for_result(after_token=None)

    assert first is not None
    assert second is not None
    assert backend.detect_calls == 1


@pytest.mark.asyncio
async def test_shared_inference_stream_skips_detection_until_requested() -> None:
    backend = CountingBackend()
    stream = SharedInferenceStream(
        frame_stream=FakeFrameStream(),
        backend=backend,  # type: ignore[arg-type]
    )

    first = await stream.wait_for_result(after_token=None, require_detection=False)
    second = await stream.wait_for_result(after_token=None, require_detection=True)

    assert first is not None
    assert first.batch is None
    assert second is not None
    assert second.batch is not None
    assert backend.detect_calls == 1

import asyncio
import logging
from collections.abc import Callable
from typing import TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class LocalModelRequestQueue:
    def __init__(self) -> None:
        self._lock: asyncio.Lock | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

    async def run(
        self,
        *,
        operation: str,
        call: Callable[[], T],
    ) -> T:
        lock = self._lock_for_running_loop()
        if lock.locked():
            logger.info("local model request queued operation=%s", operation)

        async with lock:
            logger.debug("local model request started operation=%s", operation)
            request_task = asyncio.create_task(asyncio.to_thread(call))
            try:
                return await asyncio.shield(request_task)
            except asyncio.CancelledError:
                try:
                    await request_task
                except Exception:  # noqa: BLE001
                    logger.debug(
                        "local model request failed after cancellation operation=%s",
                        operation,
                        exc_info=True,
                    )
                raise
            finally:
                logger.debug("local model request finished operation=%s", operation)

    def _lock_for_running_loop(self) -> asyncio.Lock:
        loop = asyncio.get_running_loop()
        if self._lock is None or self._loop is not loop:
            if self._lock is not None and self._lock.locked():
                raise RuntimeError(
                    "local model request queue is active on another event loop"
                )
            self._lock = asyncio.Lock()
            self._loop = loop
        return self._lock


_local_model_request_queue = LocalModelRequestQueue()


def get_local_model_request_queue() -> LocalModelRequestQueue:
    return _local_model_request_queue

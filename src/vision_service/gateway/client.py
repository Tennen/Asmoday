import asyncio
import logging
from typing import Any

import httpx

from vision_service.contracts import (
    EvidenceCallbackPayload,
    EventCallbackPayload,
    RuntimeStatusPayload,
)
from vision_service.settings import Settings

logger = logging.getLogger(__name__)


class CallbackDeliveryError(RuntimeError):
    pass


class GatewayCallbackClient:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client: httpx.AsyncClient | None = None

    async def start(self) -> None:
        if self._client is not None:
            return
        self._client = httpx.AsyncClient(
            base_url=self._settings.gateway_base_url.rstrip("/") + "/",
            timeout=self._settings.callback_timeout_seconds,
        )

    async def stop(self) -> None:
        if self._client is None:
            return
        await self._client.aclose()
        self._client = None

    async def post_status(
        self,
        callback_path: str,
        payload: RuntimeStatusPayload,
    ) -> None:
        await self._post_payload(
            callback_path=callback_path,
            payload=payload.model_dump(mode="json", exclude_none=True),
            label="status",
        )

    async def post_events(
        self,
        callback_path: str,
        payload: EventCallbackPayload,
    ) -> None:
        await self._post_payload(
            callback_path=callback_path,
            payload=payload.model_dump(mode="json", exclude_none=True),
            label="event",
        )

    async def post_evidence(
        self,
        callback_path: str,
        payload: EvidenceCallbackPayload,
    ) -> None:
        await self._post_payload(
            callback_path=callback_path,
            payload=payload.model_dump(mode="json", exclude_none=True),
            label="evidence",
        )

    async def _post_payload(
        self,
        *,
        callback_path: str,
        payload: dict[str, Any],
        label: str,
    ) -> None:
        client = self._require_client()
        normalized_path = callback_path.lstrip("/")

        last_error: Exception | None = None
        for attempt in range(1, self._settings.callback_max_attempts + 1):
            try:
                response = await client.post(normalized_path, json=payload)
                response.raise_for_status()
                if attempt > 1:
                    logger.info(
                        "gateway callback recovered label=%s callback_path=%s "
                        "attempt=%s",
                        label,
                        callback_path,
                        attempt,
                    )
                return
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                logger.warning(
                    "gateway callback failed label=%s callback_path=%s "
                    "attempt=%s/%s error=%s",
                    label,
                    callback_path,
                    attempt,
                    self._settings.callback_max_attempts,
                    exc,
                )
                if attempt == self._settings.callback_max_attempts:
                    break
                await asyncio.sleep(
                    self._settings.callback_retry_backoff_seconds * attempt,
                )

        raise CallbackDeliveryError(
            f"{label} callback delivery failed after "
            f"{self._settings.callback_max_attempts} attempt(s)"
        ) from last_error

    def _require_client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError("gateway callback client has not been started")
        return self._client

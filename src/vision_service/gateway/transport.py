import asyncio
from typing import Protocol

from fastapi import WebSocket

from vision_service.contracts import (
    EvidenceCallbackPayload,
    EventCallbackPayload,
    RuntimeStatusPayload,
)


class GatewayTransportError(RuntimeError):
    pass


class GatewayTransport(Protocol):
    async def send_status(self, payload: RuntimeStatusPayload) -> None: ...

    async def send_events(self, payload: EventCallbackPayload) -> None: ...

    async def send_evidence(self, payload: EvidenceCallbackPayload) -> None: ...


class GatewayWebSocketTransport:
    def __init__(self, websocket: WebSocket) -> None:
        self._websocket = websocket
        self._send_lock = asyncio.Lock()

    async def send_hello(self, payload: dict[str, object]) -> None:
        await self.send_message(type="hello", payload=payload)

    async def send_error(
        self,
        *,
        code: str,
        message: str,
        request_id: str | None = None,
    ) -> None:
        await self.send_message(
            type="error",
            request_id=request_id,
            payload={
                "code": code,
                "message": message,
            },
        )

    async def send_message(
        self,
        *,
        type: str,
        payload: dict[str, object] | None = None,
        request_id: str | None = None,
    ) -> None:
        message: dict[str, object] = {"type": type}
        if request_id is not None:
            message["request_id"] = request_id
        if payload is not None:
            message["payload"] = payload

        async with self._send_lock:
            try:
                await self._websocket.send_json(message)
            except Exception as exc:  # noqa: BLE001
                raise GatewayTransportError(str(exc)) from exc

    async def send_status(self, payload: RuntimeStatusPayload) -> None:
        await self.send_message(
            type="runtime_status",
            payload=payload.model_dump(mode="json", exclude_none=True),
        )

    async def send_events(self, payload: EventCallbackPayload) -> None:
        await self.send_message(
            type="rule_events",
            payload=payload.model_dump(mode="json", exclude_none=True),
        )

    async def send_evidence(self, payload: EvidenceCallbackPayload) -> None:
        await self.send_message(
            type="evidence",
            payload=payload.model_dump(mode="json", exclude_none=True),
        )

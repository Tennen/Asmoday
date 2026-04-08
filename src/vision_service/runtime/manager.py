import asyncio
from datetime import UTC, datetime

from vision_service.contracts import RuntimeStatusPayload, SyncRequest
from vision_service.settings import Settings


class RuntimeManager:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._lock = asyncio.Lock()
        self._desired_state: SyncRequest | None = None
        self._last_error: str | None = None

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    async def apply_config(self, payload: SyncRequest) -> None:
        async with self._lock:
            self._desired_state = payload.model_copy(deep=True)
            self._last_error = None

    async def current_config(self) -> SyncRequest | None:
        async with self._lock:
            if self._desired_state is None:
                return None
            return self._desired_state.model_copy(deep=True)

    async def snapshot_status(self) -> RuntimeStatusPayload:
        async with self._lock:
            desired_state = self._desired_state
            last_error = self._last_error

        if desired_state is None:
            return RuntimeStatusPayload(
                status="stopped",
                message="awaiting configuration from Gateway",
                service_version=self._settings.service_version,
                reported_at=datetime.now(tz=UTC),
                runtime={
                    "configured_rules": 0,
                    "enabled_rules": 0,
                    "active_streams": 0,
                },
            )

        enabled_rules = [rule for rule in desired_state.rules if rule.enabled]
        recognition_enabled = desired_state.recognition_enabled

        if not recognition_enabled:
            status = "stopped"
            message = "recognition disabled by Gateway"
        elif last_error:
            status = "degraded"
            message = last_error
        else:
            status = "healthy"
            message = f"configured {len(enabled_rules)} active rule(s)"

        return RuntimeStatusPayload(
            status=status,
            message=message,
            service_version=self._settings.service_version,
            reported_at=datetime.now(tz=UTC),
            runtime={
                "configured_rules": len(desired_state.rules),
                "enabled_rules": len(enabled_rules),
                "active_streams": len(enabled_rules) if recognition_enabled else 0,
            },
        )

    async def set_runtime_error(self, message: str) -> None:
        async with self._lock:
            self._last_error = message

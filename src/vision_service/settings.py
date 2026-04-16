from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _default_model_directory() -> Path:
    return Path(__file__).resolve().parents[2] / "models"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="VISION_SERVICE_",
        env_file=(".env", ".env.local"),
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    service_name: str = "vision-service"
    service_version: str = "0.1.0"
    control_ws_path: str = "/ws/control"

    host: str = "0.0.0.0"
    port: int = 8081
    log_level: Literal["critical", "error", "warning", "info", "debug", "trace"] = (
        "info"
    )
    websocket_max_message_bytes: int = Field(default=16_777_216, ge=1)
    status_interval_seconds: float = Field(default=30.0, gt=0)
    rtsp_transport: Literal["tcp", "udp"] = "tcp"
    rtsp_open_timeout_msec: int = Field(default=10_000, ge=1)
    rtsp_read_timeout_msec: int = Field(default=10_000, ge=1)
    rtsp_reconnect_failure_threshold: int = Field(default=5, ge=1)
    rtsp_reconnect_backoff_seconds: float = Field(default=1.0, gt=0)
    rtsp_reconnect_max_attempts: int = Field(default=3, ge=1)

    model_directory: Path = Field(
        default_factory=_default_model_directory,
        validation_alias=AliasChoices(
            "VISION_SERVICE_MODEL_DIRECTORY",
            "VISION_SERVICE_MODEL_PATH",
        ),
    )
    model_device: str = "cpu"
    model_confidence_threshold: float = Field(default=0.35, gt=0, lt=1)
    yolo_run_mode: Literal["always", "roi_triggered"] = Field(
        default="always",
        validation_alias=AliasChoices(
            "VISION_SERVICE_YOLO_RUN_MODE",
            "YOLO_RUN_MODE",
        ),
    )

    frame_sample_interval_seconds: float = Field(default=0.25, gt=0)
    frame_failure_backoff_seconds: float = Field(default=1.0, gt=0)
    idle_sleep_seconds: float = Field(default=0.05, gt=0)
    tracker_lost_track_buffer: int = Field(default=30, ge=1)
    evidence_buffer_max_samples: int = Field(default=32, ge=3)
    jpeg_quality: int = Field(default=85, ge=1, le=100)
    roi_enabled: bool = False
    roi_warmup_seconds: float = Field(default=3.0, ge=0)
    roi_clear_hold_seconds: float = Field(default=1.0, ge=0)
    roi_max_side_px: int = Field(default=320, ge=32)
    roi_occupancy_ratio_threshold: float = Field(default=0.08, gt=0, lt=1)
    roi_largest_blob_area_ratio_threshold: float = Field(default=0.03, gt=0, lt=1)
    roi_min_largest_blob_area: int = Field(default=400, ge=1)
    roi_mog2_history: int = Field(default=300, ge=1)
    roi_mog2_var_threshold: float = Field(default=16.0, gt=0)
    semantic_checker_base_url: str | None = None
    semantic_checker_model_name: str | None = None
    semantic_checker_api_key: str | None = None
    semantic_checker_timeout_seconds: float = Field(default=20.0, gt=0)
    semantic_checker_consecutive_yolo_failures: int = Field(default=6, ge=1)
    semantic_checker_retry_cooldown_seconds: float = Field(default=2.0, gt=0)
    semantic_checker_max_attempts_per_episode: int = Field(default=3, ge=1)

    event_id_prefix: str = "vision-evt"

    @property
    def semantic_checker_enabled(self) -> bool:
        return bool(self.semantic_checker_base_url and self.semantic_checker_model_name)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

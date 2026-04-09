from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="VISION_SERVICE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    service_name: str = "vision-service"
    service_version: str = "0.1.0"
    api_prefix: str = "/api/v1/capabilities/vision_entity_stay_zone"

    host: str = "0.0.0.0"
    port: int = 8081

    gateway_base_url: str = "http://127.0.0.1:8080"
    callback_timeout_seconds: float = Field(default=10.0, gt=0)
    callback_max_attempts: int = Field(default=3, ge=1)
    callback_retry_backoff_seconds: float = Field(default=1.0, gt=0)
    status_interval_seconds: float = Field(default=30.0, gt=0)

    model_path: str = "yolo11n.pt"
    model_device: str = "cpu"
    model_confidence_threshold: float = Field(default=0.35, gt=0, lt=1)

    frame_sample_interval_seconds: float = Field(default=0.25, gt=0)
    frame_failure_backoff_seconds: float = Field(default=1.0, gt=0)
    idle_sleep_seconds: float = Field(default=0.05, gt=0)
    tracker_lost_track_buffer: int = Field(default=30, ge=1)
    evidence_buffer_max_samples: int = Field(default=32, ge=3)
    jpeg_quality: int = Field(default=85, ge=1, le=100)

    event_id_prefix: str = "vision-evt"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

# Vision Service

Python Vision Service for Gateway's `vision_entity_stay_zone` capability.

## Scope

The service:

- exposes the entity catalog endpoint used by Gateway
- accepts desired-state sync payloads
- reads RTSP streams directly
- detects entities with Ultralytics YOLO
- tracks them with ByteTrack through supervision
- emits rule-level dwell events and evidence back to Gateway

## Implemented Contract Surface

- `GET /api/v1/capabilities/vision_entity_stay_zone/entities`
- `GET /api/v1/capabilities/vision_entity_stay_zone/models`
- `PUT /api/v1/capabilities/vision_entity_stay_zone/model`
- `PUT /api/v1/capabilities/vision_entity_stay_zone`
- Gateway status callback delivery
- Gateway event callback delivery
- Gateway evidence callback delivery

The service treats the sync payload as full desired state and reconciles rule workers accordingly.

## Quick Start

```bash
uv sync --extra dev
uv run vision-service
```

Environment variables are loaded with the `VISION_SERVICE_` prefix.
The service defaults to the repository-level `models/` directory and picks the oldest-created model entry when no explicit model is selected.

## Important Settings

- `VISION_SERVICE_GATEWAY_BASE_URL`
  Base URL for Gateway callback delivery.
- `VISION_SERVICE_RTSP_TRANSPORT`
  RTSP lower transport for OpenCV FFmpeg capture. Defaults to `tcp` because the camera stream is noticeably more loss-sensitive over `udp`.
- `VISION_SERVICE_RTSP_OPEN_TIMEOUT_MSEC`
  Open timeout passed to `cv2.VideoCapture(...)`.
- `VISION_SERVICE_RTSP_READ_TIMEOUT_MSEC`
  Read timeout passed to `cv2.VideoCapture(...)`.
- `VISION_SERVICE_MODEL_PATH`
  Optional model directory path for backward compatibility. If unset, the service uses `./models`.
- `VISION_SERVICE_MODEL_DIRECTORY`
  Preferred model directory setting. The service enumerates this directory, sorts entries by creation time, and uses the first model by default.
- `VISION_SERVICE_MODEL_DEVICE`
  Device passed to Ultralytics. Defaults to `mps` for macOS.
- `VISION_SERVICE_PORT`
  HTTP listen port for the service.
- `VISION_SERVICE_STATUS_INTERVAL_SECONDS`
  Periodic status callback interval.

## Model Selection

- `GET /api/v1/capabilities/vision_entity_stay_zone/models`
  Returns the available models, the current active model, and the default model chosen by creation time ordering.
- `PUT /api/v1/capabilities/vision_entity_stay_zone/model`
  Selects the active runtime model. If `model_name` is omitted or `null`, the service resets to the default model.
- `GET /api/v1/capabilities/vision_entity_stay_zone/entities?model_name=foo.pt`
  Returns the catalog for a specific model without changing the active runtime selection.

## Runtime Model

- Gateway remains the source of truth for desired configuration.
- Enabled rules still run as independent dwell workers, but rules that share the same `rtsp_source.url` now reuse a single RTSP capture task.
- Each worker consumes the latest shared frame, runs YOLO detection with the currently selected model, applies ByteTrack tracking, filters by the configured entity label, and checks whether the tracked box center remains inside the configured normalized zone.
- `threshold_met` is emitted once per dwell episode.
- `cleared` is emitted once when the active episode is no longer present.
- Evidence uploads use `start`, `middle`, and `end` JPEG frames captured from the episode buffer.

## Validation

Static validation completed:

- `python3 -m compileall src tests`
- `.venv/bin/pytest`
- FastAPI app factory smoke via `.venv/bin/python -c "from vision_service.app import create_app; create_app()"`

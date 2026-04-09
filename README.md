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

## Important Settings

- `VISION_SERVICE_GATEWAY_BASE_URL`
  Base URL for Gateway callback delivery.
- `VISION_SERVICE_MODEL_PATH`
  Ultralytics model path or weight file, for example `yolo11n.pt`.
- `VISION_SERVICE_MODEL_DEVICE`
  Device passed to Ultralytics, for example `cpu` or `mps`.
- `VISION_SERVICE_PORT`
  HTTP listen port for the service.
- `VISION_SERVICE_STATUS_INTERVAL_SECONDS`
  Periodic status callback interval.

## Runtime Model

- Gateway remains the source of truth for desired configuration.
- Each enabled rule currently maps to one RTSP worker.
- Each worker reads the RTSP source directly, runs YOLO detection, applies ByteTrack tracking, filters by the configured entity label, and checks whether the tracked box center remains inside the configured normalized zone.
- `threshold_met` is emitted once per dwell episode.
- `cleared` is emitted once when the active episode is no longer present.
- Evidence uploads use `start`, `middle`, and `end` JPEG frames captured from the episode buffer.

## Validation

Static validation completed:

- `python3 -m compileall src tests`
- direct smoke execution of `RuleDwellTracker`

Full `pytest` execution depends on a successful `uv sync --extra dev` run because the local environment did not yet finish installing all packages during implementation.

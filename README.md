# Vision Service

Python Vision Service for Gateway-controlled RTSP recognition.

## Scope

The service:

- exposes a single Gateway WebSocket session endpoint
- accepts model queries, model selection, and desired-state sync over that session
- reads RTSP streams directly only while the Gateway session is connected
- detects entities with Ultralytics YOLO
- tracks them with ByteTrack through supervision
- emits runtime status, rule events, and evidence back to Gateway over the same WebSocket session

The service is intentionally stateless with respect to Gateway configuration:

- it does not persist synced rules locally
- it does not buffer evidence for later delivery after Gateway disconnects
- when the Gateway WebSocket disconnects, the service stops RTSP ingestion, recognition workers, and telemetry delivery immediately
- when Gateway reconnects, it must resend model selection and `sync_config`

The integration contract is documented in [vision-service-contract.md](./vision-service-contract.md).

## Quick Start

```bash
uv sync --extra dev
uv run vision-service
```

Environment variables use the `VISION_SERVICE_` prefix.

## Important Settings

- `VISION_SERVICE_CONTROL_WS_PATH`
  WebSocket control route exposed to Gateway. Defaults to `/ws/control`.
- `VISION_SERVICE_STATUS_INTERVAL_SECONDS`
  Periodic runtime status push interval while a Gateway session is connected.
- `VISION_SERVICE_WEBSOCKET_MAX_MESSAGE_BYTES`
  Maximum WebSocket message size accepted by Uvicorn. This matters because evidence is sent inline as base64 JPEG payloads.
- `VISION_SERVICE_MODEL_DIRECTORY`
  Model directory. If unset, the service uses `./models`.
- `VISION_SERVICE_MODEL_DEVICE`
  Device passed to Ultralytics. Defaults to `cpu`.
- `VISION_SERVICE_RTSP_TRANSPORT`
  RTSP lower transport for OpenCV FFmpeg capture. Defaults to `tcp`.
- `VISION_SERVICE_RTSP_OPEN_TIMEOUT_MSEC`
  Open timeout passed to `cv2.VideoCapture(...)`.
- `VISION_SERVICE_RTSP_READ_TIMEOUT_MSEC`
  Read timeout passed to `cv2.VideoCapture(...)`.
- `VISION_SERVICE_PORT`
  HTTP listen port for the service.

## WebSocket Session

Gateway connects to:

```text
ws://{vision-service-host}:{port}/ws/control
```

The server accepts one Gateway session at a time. On connect it sends:

1. `hello`
2. `runtime_status`

Gateway can then send:

- `get_models`
- `select_model`
- `get_entities`
- `sync_config`

The service may asynchronously send:

- `runtime_status`
- `rule_events`
- `evidence`
- `error`

`sync_config` is the full latest desired state. Missing rules are removed from runtime. If the connection drops, runtime is torn down and Gateway must reconnect and resend state.

## Runtime Model

- Gateway remains the source of truth for desired configuration.
- Enabled rules still run as independent dwell workers, but rules that share the same `rtsp_source.url` reuse a single RTSP capture task.
- Each worker consumes the latest shared frame, runs YOLO detection with the currently selected model, applies ByteTrack tracking, filters by the configured entity label, and checks whether the tracked box center remains inside the configured normalized zone.
- `threshold_met` is emitted once after a dwell episode ends and exceeded the configured threshold.
- Evidence is sent as `start`, `middle`, and `end` JPEG frames over the WebSocket session.

## Validation

Validation completed with:

- `python3 -m compileall src tests`
- `uv run pytest`

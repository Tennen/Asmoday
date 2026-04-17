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
cp .env.example .env
uv run vision-service
```

Environment variables use the `VISION_SERVICE_` prefix.

## Environment Management

The service already loads environment files automatically through Pydantic Settings:

- `.env`
  Project-local defaults you want the app to read without `export`.
- `.env.local`
  Machine-specific overrides. This is useful when your personal RTSP endpoints, ports, or debug settings differ from the shared default file.
- Real environment variables
  Still take precedence over values from `.env` files when you need a one-off override.

Recommended workflow:

1. Copy `.env.example` to `.env`.
2. Put safe shared local defaults in `.env`.
3. Put personal overrides in `.env.local`.
4. Run `uv run vision-service` directly without extra `export` commands.

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
- `VISION_SERVICE_ROI_ENABLED`
  Enables ROI occupancy detection alongside the existing YOLO pipeline. Defaults to `false`.
- `VISION_SERVICE_YOLO_RUN_MODE`
  Controls how YOLO is scheduled when ROI is enabled. `always` keeps current behavior. `roi_triggered` keeps ROI resident and only requests shared YOLO inference while ROI occupancy is active.
- `VISION_SERVICE_SEMANTIC_CHECKER_BASE_URL`
  Optional OpenAI-compatible local VLM endpoint. When unset, ROI/VLM semantic fallback stays disabled and rule-level `key_entities` matching cannot run.
- `VISION_SERVICE_SEMANTIC_CHECKER_MODEL_NAME`
  Model name sent to the semantic checker `chat/completions` request.
- `VISION_SERVICE_SEMANTIC_CHECKER_TIMEOUT_SECONDS`
  Timeout for local VLM `chat/completions` calls. Defaults to `20`.
- `VISION_SERVICE_SEMANTIC_CHECKER_CONSECUTIVE_YOLO_FAILURES`
  Consecutive in-zone YOLO misses required before the service asks the VLM to re-check an occupied ROI.
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
- Each worker consumes the latest shared frame and can optionally run an ROI occupancy detector over its configured zone.
- With `VISION_SERVICE_ROI_ENABLED=false`, workers keep the original behavior: shared YOLO inference runs on every frame, then ByteTrack tracking and zone filtering drive dwell.
- With `VISION_SERVICE_ROI_ENABLED=true` and `VISION_SERVICE_YOLO_RUN_MODE=always`, ROI runs in parallel as an occupancy signal but does not silently extend YOLO episodes on its own.
- With `VISION_SERVICE_ROI_ENABLED=true` and `VISION_SERVICE_YOLO_RUN_MODE=roi_triggered`, ROI stays resident and shared YOLO inference is requested only while ROI occupancy is active.
- When ROI remains occupied but YOLO keeps missing and the semantic checker is configured, the worker can ask the local VLM to re-check cropped zone keyframes using the rule's entity plus optional behavior text.
- `threshold_met` is emitted once after a dwell episode ends and exceeded the configured threshold.
- Evidence is sent as `start`, `middle`, and `end` raw JPEG frames plus structured YOLO detections over the WebSocket session.
- ROI/VLM fallback crops the configured zone only for the semantic checker input; emitted evidence remains full-frame raw JPEG so Gateway/Admin can show where the episode occurred.
- When a YOLO dwell event includes `key_entities`, the worker crops the triggering tracked entity from those three evidence samples, runs one-to-many VLM matching, aggregates the per-frame votes, and includes the winning `key_entity_id` in the emitted event when available. ROI-only semantic fallback does not run key entity matching.

## Validation

Validation completed with:

- `python3 -m compileall src tests`
- `uv run pytest`

# Vision Service Integration Contract

Back to the [API index](../api.md).

This document is the downstream-facing integration contract for the external Vision Service that works with Celestia's `vision_entity_stay_zone` capability.

Important:

- The current integration is HTTP + JSON, not gRPC.
- Gateway is the source of truth for configuration.
- Gateway does not proxy RTSP, does not process frames, and does not depend on the Vision engine's implementation details.
- The Vision Service is responsible for consuming RTSP, running recognition, maintaining dwell logic, and calling Gateway back with status, structured events, and event evidence screenshots.

## Roles

### Gateway Responsibilities

- Own and persist all vision capability configuration.
- Push a normalized configuration payload to the Vision Service.
- Receive service status callbacks.
- Receive structured recognition events.
- Receive screenshot evidence tied to specific recognition events.
- Project recognition results into Gateway events and camera state for downstream automation and notification flows.

### Vision Service Responsibilities

- Accept the config pushed by Gateway.
- Treat that config as the current desired state.
- Start, stop, or update internal recognition pipelines accordingly.
- Read RTSP from the provided `rtsp_source.url`.
- Track dwell duration for each configured rule.
- Report service health and runtime status back to Gateway.
- Report threshold-crossing events back to Gateway.
- Report start / middle / end screenshots back to Gateway for persisted event evidence.

## Transport

- Protocol: HTTP/1.1 or HTTP/2 over JSON
- Encoding: UTF-8 JSON
- Auth: no auth headers are currently defined by Gateway
- Time format: RFC3339 / RFC3339Nano UTC timestamps

Current base paths:

- Gateway -> Vision Service entity catalog:
  `GET {service_url}/api/v1/capabilities/vision_entity_stay_zone/entities`
- Gateway -> Vision Service sync:
  `PUT {service_url}/api/v1/capabilities/vision_entity_stay_zone`
- Vision Service -> Gateway status callback:
  `POST {gateway_base}{status_path}`
- Vision Service -> Gateway event callback:
  `POST {gateway_base}{event_path}`
- Vision Service -> Gateway evidence callback:
  `POST {gateway_base}{evidence_path}`

The `status_path`, `event_path`, and `evidence_path` are supplied by Gateway inside the sync payload, so the Vision Service should not hardcode them beyond basic routing support.

## 0. Entity Catalog Discovery: Gateway -> Vision Service

Before Gateway configures a rule such as `cat` stay-zone detection, it first fetches the current model-supported entity catalog from the Vision Service.

### Request

`GET {service_url}/api/v1/capabilities/vision_entity_stay_zone/entities`

### Recommended Response

```json
{
  "schema_version": "celestia.vision.catalog.v1",
  "service_version": "1.2.0",
  "model_name": "yolo11m-coco",
  "fetched_at": "2026-04-08T09:18:00Z",
  "entities": [
    {
      "kind": "label",
      "value": "cat",
      "display_name": "Cat"
    },
    {
      "kind": "label",
      "value": "dog",
      "display_name": "Dog"
    }
  ]
}
```

### Field Semantics

- `schema_version`
  Current value is `celestia.vision.catalog.v1`.
- `service_version`
  Optional service/runtime version string.
- `model_name`
  Optional current recognition model identifier, for example `yolo11m-coco`.
- `fetched_at`
  Service-side timestamp for this entity catalog snapshot.
- `entities`
  Current recognizable entities for the active model.

#### Entity Fields

- `kind`
  Generic selector kind that Gateway can feed back into `entity_selector.kind`.
  Current recommended value is `label`.
- `value`
  Business-facing selector value, for example `cat`.
- `display_name`
  Optional human-friendly label for admin surfaces.

Gateway uses this catalog to populate the rule editor and validate configured entities when the catalog matches the target Vision Service address.

## 1. Config Sync: Gateway -> Vision Service

### Request

`PUT {service_url}/api/v1/capabilities/vision_entity_stay_zone`

Request body:

```json
{
  "schema_version": "celestia.vision.control.v1",
  "sent_at": "2026-04-08T09:20:00Z",
  "recognition_enabled": true,
  "callbacks": {
    "status_path": "/api/v1/capabilities/vision_entity_stay_zone/status",
    "event_path": "/api/v1/capabilities/vision_entity_stay_zone/events",
    "evidence_path": "/api/v1/capabilities/vision_entity_stay_zone/evidence"
  },
  "rules": [
    {
      "id": "feeder-zone",
      "name": "Feeder Zone",
      "enabled": true,
      "camera": {
        "device_id": "hikvision:camera:entry-1",
        "plugin_id": "hikvision",
        "vendor_device_id": "192.0.2.10:8000:ch1",
        "name": "Patio Camera",
        "entry_id": "entry-1"
      },
      "rtsp_source": {
        "url": "rtsp://user:pass@camera/stream"
      },
      "entity_selector": {
        "kind": "label",
        "value": "cat"
      },
      "zone": {
        "x": 0.12,
        "y": 0.28,
        "width": 0.34,
        "height": 0.22
      },
      "stay_threshold_seconds": 5
    }
  ]
}
```

### Field Semantics

#### Top Level

- `schema_version`
  Current value is `celestia.vision.control.v1`. The Vision Service should reject unknown incompatible versions.
- `sent_at`
  Gateway timestamp for this desired-state snapshot.
- `recognition_enabled`
  Global kill switch for the whole capability.
  If `false`, the Vision Service should stop all recognition work for this capability.
- `callbacks.status_path`
  Relative Gateway callback path for service status updates.
- `callbacks.event_path`
  Relative Gateway callback path for recognition events.
- `callbacks.evidence_path`
  Relative Gateway callback path for screenshot evidence uploads.
- `rules`
  Full desired rule set, not a patch.
  The Vision Service should reconcile its internal runtime to exactly this list.

#### Rule

- `id`
  Stable rule identifier.
  Gateway also uses this to generate projected camera state keys.
- `name`
  Human-readable label only.
- `enabled`
  Rule-level switch.
  If `false`, the Vision Service must not emit recognition events for this rule.
- `camera`
  Gateway camera identity metadata.
  This is informational for correlation and logging; `camera.device_id` is the canonical Gateway camera id.
- `rtsp_source.url`
  RTSP endpoint the Vision Service should read directly.
  Gateway will not relay or proxy this stream.
- `entity_selector.kind`
  Generic selector type.
  Current recommended value is `label`.
- `entity_selector.value`
  Generic selector target, for example `cat`.
  This is business-facing configuration, not an engine-specific class map.
- `zone`
  Normalized rectangle in source-frame coordinates.
  Range is `[0, 1]`.
  `x`, `y` are top-left origin ratios.
  `width`, `height` are size ratios.
- `stay_threshold_seconds`
  Dwell threshold that the Vision Service must use before emitting `threshold_met`.

### Desired-State Reconciliation Rules

The Vision Service should treat each sync request as the full latest state:

- Rules missing from the new payload should be stopped and removed from active processing.
- Existing rules with changed fields should be updated in place if possible, otherwise restarted.
- If `recognition_enabled=false`, the service should stop all active pipelines for this capability.

### Response

Gateway currently only checks the HTTP status code.

Contract:

- Any `2xx` status means the config sync succeeded.
- Any non-`2xx` status means Gateway marks the capability sync as degraded.

Recommended success response:

```json
{
  "ok": true
}
```

Recommended error response:

```json
{
  "error": "human-readable message"
}
```

## 2. Status Callback: Vision Service -> Gateway

### Request

`POST {gateway_base}{status_path}`

Request body:

```json
{
  "status": "healthy",
  "message": "tracking 1 stream",
  "service_version": "1.2.0",
  "reported_at": "2026-04-08T09:25:00Z",
  "runtime": {
    "active_streams": 1
  }
}
```

### Fields

- `status`
  One of:
  - `unknown`
  - `healthy`
  - `degraded`
  - `unhealthy`
  - `stopped`
- `message`
  Short human-readable runtime summary.
- `service_version`
  Optional build/runtime version string from the Vision Service.
- `reported_at`
  Service-side report timestamp.
- `runtime`
  Optional arbitrary JSON object with operational details.
  Example: active stream count, queue backlog, engine mode, worker count.

### Response

Gateway responds with the persisted status object.

The Vision Service may ignore the response body if it only needs delivery confirmation.

## 3. Event Callback: Vision Service -> Gateway

### Request

`POST {gateway_base}{event_path}`

Request body:

```json
{
  "events": [
    {
      "event_id": "vision-evt-1",
      "rule_id": "feeder-zone",
      "camera_device_id": "hikvision:camera:entry-1",
      "status": "threshold_met",
      "observed_at": "2026-04-08T09:28:11Z",
      "dwell_seconds": 6,
      "entity_value": "cat",
      "metadata": {
        "track_id": "trk-7"
      }
    }
  ]
}
```

### Fields

- `events`
  Non-empty batch of structured recognition events.

#### Event Fields

- `event_id`
  Optional event identifier from the Vision Service.
  Recommended: globally unique per emitted event.
  Required if the Vision Service later uploads screenshot evidence for this event.
- `rule_id`
  Required Gateway rule id from the current synced config.
- `camera_device_id`
  Optional if the rule uniquely determines the camera.
  If provided, it must match the camera bound to `rule_id`.
- `status`
  One of:
  - `threshold_met`
  - `cleared`
- `observed_at`
  Detection timestamp from the Vision Service.
- `dwell_seconds`
  Current or final measured dwell duration for this detection episode.
- `entity_value`
  Optional normalized entity identifier, for example `cat`.
- `metadata`
  Optional arbitrary structured details, for example track id or model confidence summaries.

## Event Emission Semantics

This is the most important behavioral contract for the downstream Vision Service.

### `threshold_met`

Emit exactly once when a tracked entity first crosses the configured dwell threshold for a rule.

Do not repeatedly emit `threshold_met` while the same entity remains in the same active dwell episode.

Reason:

- Gateway increments `vision_rule_<rule_id>_match_count` on each `threshold_met`.
- Existing automations can trigger from that state change.
- Repeated `threshold_met` spam would produce duplicate automations and notifications.

### `cleared`

Emit when a previously active threshold-met episode is no longer active for that rule.

Typical causes:

- the tracked entity left the zone
- the tracked entity no longer matches the selector
- tracking was lost after a previously active episode

`cleared` does not increment the match counter.
It clears the projected `vision_rule_<rule_id>_active` state in Gateway.

### Multiple Entities

Current contract is rule-centric, not multi-object-state replication.

The Vision Service should collapse its internal tracking into the rule-level event stream above.

If multiple matching entities exist simultaneously, the service should still emit rule-level `threshold_met` / `cleared` transitions rather than raw per-frame detections.

## 4. Evidence Callback: Vision Service -> Gateway

### Request

`POST {gateway_base}{evidence_path}`

Request body:

```json
{
  "captures": [
    {
      "capture_id": "vision-evt-1:start",
      "event_id": "vision-evt-1",
      "rule_id": "feeder-zone",
      "camera_device_id": "hikvision:camera:entry-1",
      "phase": "start",
      "captured_at": "2026-04-08T09:28:05Z",
      "content_type": "image/jpeg",
      "image_base64": "/9j/4AAQSk..."
    },
    {
      "capture_id": "vision-evt-1:middle",
      "event_id": "vision-evt-1",
      "rule_id": "feeder-zone",
      "camera_device_id": "hikvision:camera:entry-1",
      "phase": "middle",
      "captured_at": "2026-04-08T09:28:08Z",
      "content_type": "image/jpeg",
      "image_base64": "/9j/4AAQSk..."
    },
    {
      "capture_id": "vision-evt-1:end",
      "event_id": "vision-evt-1",
      "rule_id": "feeder-zone",
      "camera_device_id": "hikvision:camera:entry-1",
      "phase": "end",
      "captured_at": "2026-04-08T09:28:11Z",
      "content_type": "image/jpeg",
      "image_base64": "/9j/4AAQSk..."
    }
  ]
}
```

### Fields

- `captures`
  Non-empty batch of event screenshots.
- `capture_id`
  Optional stable screenshot identifier.
  If omitted, Gateway derives one from `event_id` and `phase`.
- `event_id`
  Required previously accepted recognition `event_id`.
- `rule_id`
  Optional redundancy field; if provided it must match the persisted event.
- `camera_device_id`
  Optional redundancy field; if provided it must match the persisted event camera.
- `phase`
  One of:
  - `start`
  - `middle`
  - `end`
- `captured_at`
  Screenshot timestamp from the Vision Service.
- `content_type`
  Optional image MIME type, for example `image/jpeg`.
- `image_base64`
  Required base64-encoded screenshot bytes.

### Behavior

- Gateway persists the screenshot bytes in Core-owned storage.
- Gateway associates each screenshot with the referenced event record.
- Gateway enforces screenshot retention using its own configured persistence window.

## 5. Gateway Side Effects

For each accepted event callback, Gateway currently does two things:

1. Appends a structured `device.event.occurred` event with `payload.capability_id = "vision_entity_stay_zone"`.
2. Projects the rule result into the camera device state and emits `device.state.changed`.

Projected camera state keys:

- `vision_rule_<rule_id>_match_count`
- `vision_rule_<rule_id>_active`
- `vision_rule_<rule_id>_last_event_at`
- `vision_rule_<rule_id>_last_entity_value`
- `vision_rule_<rule_id>_last_dwell_seconds`
- `vision_rule_<rule_id>_last_status`

Current projection rules:

- `threshold_met`
  - increments `vision_rule_<rule_id>_match_count`
  - sets `vision_rule_<rule_id>_active = true`
- `cleared`
  - sets `vision_rule_<rule_id>_active = false`

Both statuses update:

- `vision_rule_<rule_id>_last_event_at`
- `vision_rule_<rule_id>_last_entity_value`
- `vision_rule_<rule_id>_last_dwell_seconds`
- `vision_rule_<rule_id>_last_status`

For each accepted evidence callback, Gateway persists the uploaded screenshots and exposes them back on the matching event record for admin review.

## 6. Failure Handling

### Config Sync Failure

If the Vision Service returns non-`2xx` or is unreachable:

- Gateway still persists the desired config locally.
- Gateway marks the capability runtime as degraded.
- Gateway stores the sync error in `runtime.sync_error`.

The Vision Service should therefore not assume that missing sync requests mean the feature is disabled.
Its runtime state should always be derived from the most recent successful sync it accepted.

### Callback Failure

If Gateway returns non-`2xx` on status, event, or evidence callbacks:

- The Vision Service should log the failure.
- The Vision Service should retry based on its own retry policy.

Current Gateway behavior does not expose a dedicated idempotency protocol for event retries.
Practical guidance for the Vision Service:

- only emit `threshold_met` once per dwell episode
- use a unique `event_id` if your runtime needs stable correlation
- avoid blind replay storms for already-delivered events

For evidence retries:

- keep `event_id` stable
- upload at most one screenshot per phase (`start`, `middle`, `end`) unless intentionally replacing the prior upload

## 7. Minimal Downstream Checklist

To be considered compatible with the current Gateway contract, the Vision Service must:

1. Expose `GET /api/v1/capabilities/vision_entity_stay_zone/entities`.
2. Expose `PUT /api/v1/capabilities/vision_entity_stay_zone`.
3. Accept the full sync payload defined above.
4. Reconcile internal active rules to the synced desired state.
5. Read RTSP from `rtsp_source.url` directly.
6. Apply zone and dwell logic per rule.
7. POST runtime health to Gateway using the provided `status_path`.
8. POST structured `threshold_met` / `cleared` batches to Gateway using the provided `event_path`.
9. POST screenshot evidence batches to Gateway using the provided `evidence_path`.
10. Avoid repeated `threshold_met` emission for the same active stay episode.

## 8. Schema Reference

The canonical Go structs in this repository are:

- `internal/models/vision.go`
  - `VisionServiceEntityCatalog`
  - `VisionServiceSyncPayload`
  - `VisionServiceSyncCallbacks`
  - `VisionServiceRule`
  - `VisionServiceCamera`
  - `VisionServiceStatusReport`
  - `VisionServiceEvent`
  - `VisionServiceEventBatch`
  - `VisionServiceEventCapture`
  - `VisionServiceEventCaptureBatch`

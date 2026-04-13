# Vision Service WebSocket Contract

This document is the integration contract between Gateway and the standalone Vision Service control session.

## Overview

The protocol is WebSocket-only.

- Gateway connects to Vision Service over a single long-lived WebSocket session.
- Gateway pushes model selection and full desired-state sync over that session.
- Vision Service pushes runtime status, rule events, and evidence back over that same session.
- Vision Service is intentionally stateless for Gateway-owned configuration.

Implications:

- Vision Service does not persist synced rules locally.
- Vision Service does not attempt deferred evidence delivery after Gateway disconnects.
- When the WebSocket disconnects, Vision Service stops RTSP ingestion, recognition workers, and telemetry delivery immediately.
- Gateway must reconnect and resend model selection and `sync_config`.

## Endpoint

Gateway connects to:

```text
ws://{vision_service_host}:{port}/ws/control
```

The route path is configurable with `VISION_SERVICE_CONTROL_WS_PATH`.

Current server behavior:

- exactly one active Gateway session is allowed
- if a second session connects while one is active, Vision Service sends an `error` message and closes the second connection

## Session Lifecycle

On successful connection:

1. Vision Service accepts the WebSocket.
2. Vision Service sends a `hello` message.
3. Vision Service sends an initial `runtime_status` message.

On disconnect:

1. Vision Service clears the active session transport.
2. Vision Service stops all workers and RTSP streams.
3. Vision Service drops the current synced config and selected model state.

## Message Envelope

Every message uses the same outer envelope:

```json
{
  "type": "message_type",
  "request_id": "optional-correlation-id",
  "payload": {}
}
```

Field semantics:

- `type`
  Required message type string.
- `request_id`
  Optional correlation id supplied by Gateway for request/response flows.
  Vision Service echoes it on direct responses and error replies related to that request.
- `payload`
  Optional object whose schema depends on `type`.

## Server-Initiated Messages

### `hello`

Sent once immediately after the socket is accepted.

```json
{
  "type": "hello",
  "payload": {
    "schema_version": "celestia.vision.ws.v1",
    "service_version": "0.1.0",
    "connected_at": "2026-04-11T08:00:00Z"
  }
}
```

### `runtime_status`

Sent:

- once immediately after connect
- periodically while connected
- immediately after successful `sync_config`
- whenever the service reports a new runtime state snapshot

Payload schema:

```json
{
  "type": "runtime_status",
  "payload": {
    "status": "healthy",
    "message": "tracking 1 stream(s) across 2 rule(s)",
    "service_version": "0.1.0",
    "reported_at": "2026-04-11T08:00:05Z",
    "runtime": {
      "configured_rules": 2,
      "enabled_rules": 2,
      "active_streams": 1,
      "workers": []
    }
  }
}
```

### `rule_events`

Sent once after a dwell episode ends and the completed stay exceeded the configured threshold.

```json
{
  "type": "rule_events",
  "payload": {
    "events": [
      {
        "event_id": "vision-evt-123",
        "rule_id": "feeder-zone",
        "camera_device_id": "hikvision:camera:entry-1",
        "status": "threshold_met",
        "observed_at": "2026-04-11T08:00:10Z",
        "dwell_seconds": 5,
        "entity_value": "cat",
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
        ],
        "metadata": {
          "track_id": "7"
        }
      }
    ]
  }
}
```

`rule_events` entity semantics:

- `entity_value` remains the backward-compatible primary entity identifier.
- `entities` is optional, but when present it carries the complete set of recognized entities currently inside the configured zone for that emitted event.
- When Gateway syncs a rule whose `entity_selector.value == ""`, Vision Service must treat that rule as "no class filter" and must not gate detections by entity class before dwell aggregation.
- For wildcard rules, events must include every in-zone recognized entity in `entities`.
- Gateway uses `entities` for persisted history display and entity-based filtering in Admin, so Vision Service should keep the array stable, deduplicated by `kind + value`, and ordered by its primary/most relevant entity first.
- `metadata.decision` may carry source and scoring details such as `source`, `confidence_score`, `confidence_breakdown`, and semantic checker verdicts when ROI/VLM fallback was used.

### `evidence`

Sent after the matching `rule_events` message when the completed dwell event includes evidence.

```json
{
  "type": "evidence",
  "payload": {
    "captures": [
      {
        "capture_id": "vision-evt-123:start",
        "event_id": "vision-evt-123",
        "rule_id": "feeder-zone",
        "camera_device_id": "hikvision:camera:entry-1",
        "phase": "start",
        "captured_at": "2026-04-11T08:00:08Z",
        "content_type": "image/jpeg",
        "image_base64": "...",
        "metadata": {
          "annotations": {
            "image_kind": "raw",
            "coordinate_space": "normalized_xywh",
            "source": "ultralytics.boxes",
            "detections": [
              {
                "kind": "label",
                "value": "cat",
                "display_name": "Cat",
                "confidence": 0.93,
                "track_id": "7",
                "box": {
                  "x": 0.12,
                  "y": 0.24,
                  "width": 0.31,
                  "height": 0.42
                }
              }
            ]
          }
        }
      }
    ]
  }
}
```

`evidence` annotation semantics:

- Vision Service may send either a pre-rendered annotated image or a raw capture plus structured detections.
- `metadata.annotations.image_kind = "annotated"` means `image_base64` already contains rendered boxes and labels. Gateway/Admin should display that image as-is and must not draw another overlay from the same detection set.
- `metadata.annotations.image_kind = "raw"` means `image_base64` is an unannotated capture. Gateway/Admin may draw the overlay from `metadata.annotations.detections`.
- `metadata.annotations.coordinate_space` is fixed to normalized top-left origin coordinates in `[0,1]` space using `box.{x,y,width,height}`.
- Each detection should carry `display_name` and may optionally carry `kind`, `value`, `confidence`, and `track_id`.
- If Vision Service uses Ultralytics' own renderer to generate evidence, it should still keep detection ordering stable and may optionally include the same detection list for downstream structured consumers.

### `error`

Sent when Vision Service rejects a request message but keeps the session open.

```json
{
  "type": "error",
  "request_id": "req-3",
  "payload": {
    "code": "model_not_found",
    "message": "model 'missing.pt' was not found in directory /path/to/models"
  }
}
```

## Gateway-Initiated Messages

### `get_models`

Request:

```json
{
  "type": "get_models",
  "request_id": "req-1"
}
```

Response:

```json
{
  "type": "models",
  "request_id": "req-1",
  "payload": {
    "schema_version": "celestia.vision.models.v1",
    "service_version": "0.1.0",
    "current_model_name": "yolo11n.pt",
    "default_model_name": "yolo11n.pt",
    "fetched_at": "2026-04-11T08:00:01Z",
    "models": [
      {
        "name": "yolo11n.pt",
        "created_at": "2026-04-10T08:00:00Z",
        "is_selected": true,
        "is_default": true
      }
    ]
  }
}
```

### `select_model`

Request:

```json
{
  "type": "select_model",
  "request_id": "req-2",
  "payload": {
    "model_name": "custom-pets.pt"
  }
}
```

`model_name: null` resets selection to the default model.

Response:

```json
{
  "type": "model_selected",
  "request_id": "req-2",
  "payload": {
    "ok": true,
    "model_name": "custom-pets.pt",
    "changed_at": "2026-04-11T08:00:02Z"
  }
}
```

### `get_entities`

Request:

```json
{
  "type": "get_entities",
  "request_id": "req-3",
  "payload": {
    "model_name": "custom-pets.pt"
  }
}
```

`payload` may be omitted or `{}` to use the current runtime model.

Response:

```json
{
  "type": "entity_catalog",
  "request_id": "req-3",
  "payload": {
    "schema_version": "celestia.vision.catalog.v1",
    "service_version": "0.1.0",
    "model_name": "custom-pets.pt",
    "fetched_at": "2026-04-11T08:00:03Z",
    "entities": [
      {
        "kind": "label",
        "value": "cat",
        "display_name": "Cat"
      }
    ]
  }
}
```

### `sync_config`

This is the full desired-state snapshot for the active Gateway session.

Request:

```json
{
  "type": "sync_config",
  "request_id": "req-4",
  "payload": {
    "schema_version": "celestia.vision.control.ws.v1",
    "sent_at": "2026-04-11T08:00:04Z",
    "recognition_enabled": true,
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
        "behavior": "eating",
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
}
```

Response:

```json
{
  "type": "sync_applied",
  "request_id": "req-4",
  "payload": {
    "ok": true,
    "applied_at": "2026-04-11T08:00:04Z"
  }
}
```

Important semantics:

- `sync_config` replaces the full active desired state.
- rules missing from the new payload are stopped and removed.
- `recognition_enabled=false` stops all recognition work cleanly.
- rules only run while the WebSocket session remains connected.
- `entity_selector.value == ""` means the rule is intentionally wildcarded. Vision Service must still honor the rule's zone and dwell threshold, but it must skip any class-level inclusion gate for that rule.
- `behavior` is optional. When present, Vision Service may combine it with the entity selector to build a short semantic-check prompt for an optional local VLM fallback path.

## Ordering Notes

The protocol is asynchronous.

- Vision Service may send `runtime_status`, `rule_events`, or `evidence` at any time while the session is open.
- After `sync_config`, Vision Service currently emits a fresh `runtime_status` during reconciliation, so Gateway may observe `runtime_status` before `sync_applied`.
- `request_id` correlation only applies to direct request/response messages and related `error` replies.

## Failure Semantics

### Invalid request payload

Vision Service sends:

```json
{
  "type": "error",
  "request_id": "req-x",
  "payload": {
    "code": "invalid_message",
    "message": "..."
  }
}
```

The socket remains open.

### Unknown message type

Vision Service sends:

```json
{
  "type": "error",
  "request_id": "req-x",
  "payload": {
    "code": "unsupported_message_type",
    "message": "unsupported message type: ..."
  }
}
```

The socket remains open.

### Disconnect

If the WebSocket disconnects:

- Vision Service stops RTSP ingestion and workers immediately.
- unsent status, event, or evidence messages are dropped
- Gateway must reconnect and resend model selection and `sync_config`

## Implementation Summary

Gateway responsibilities:

- hold the WebSocket connection open while vision runtime should remain active
- resend session state after reconnect
- consume asynchronous `runtime_status`, `rule_events`, and `evidence`

Vision Service responsibilities:

- keep runtime strictly scoped to the live Gateway session
- reconcile rules from the latest `sync_config`
- stop all work when the session disconnects
- avoid local persistence of Gateway-owned desired state
